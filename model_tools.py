
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ╭──────────────────────── UTILIDADES GERAIS ───────────────────────╮
def load_csv(path: str) -> np.ndarray:
    """
    Load CSV without header that uses commas both as separator and decimal.
    (Note: For wine data, pd.read_csv with sep=';' and no header=None is used directly
    in main scripts, as wine data has headers and ';' separator).
    """
    return pd.read_csv(path, header=None, decimal=',').to_numpy()

def create_nonlinear_features(X: np.ndarray) -> np.ndarray:
    """
    Create nonlinear features for each column in X.
    For each j ∈ [0, D-1], append:
    - xj²
    - xj³
    - log(xj)
    - xj * xl (for each l ≠ j and l ∈ [0, D-1])
    Parameters
    ----------
    X : ndarray (N, D)
        Input feature matrix.

    Returns
    -------
    X_new : ndarray (N, D')
        Output feature matrix with added nonlinear features.
    """
    N, D = X.shape

    num_interaction_features = len(list(combinations(range(D), 2)))
    total_new_features = D + D + D + D + num_interaction_features

    X_new = np.zeros((N, total_new_features))

    for i in range(N):
        current_col = 0
        X_new[i, current_col:D] = X[i, :] # Copy original features
        current_col += D

        # Note: log(0) is undefined, so we add a small constant to avoid it
        for j in range(D):
            X_new[i][current_col + j] = X[i][j] ** 2
        current_col += D

        for j in range(D):
            X_new[i][current_col + j] = X[i][j] ** 3
        current_col += D

        for j in range(D):
            X_new[i][current_col + j] = np.log(np.abs(X[i][j]) + 1e-8)
        current_col += D

        # Add xj * xl for each j and l ≠ j
        for cols in combinations(range(D), 2):
            j, l = cols
            X_new[i][current_col] = X[i][j] * X[i][l]
            current_col += 1

    return X_new


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score each column: (x-E(x))/Var(x).

    Returns
    -------
    X_norm : ndarray
        Normalised matrix (mean 0, std 1 per feature).
    mean   : ndarray
        Feature-wise means (for inverse transform / test norm).
    std    : ndarray
        Feature-wise stds.
    """
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std

def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of 1 s → handles θ₀ (intercept)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def mse(theta: np.ndarray, Xb: np.ndarray, y: np.ndarray) -> float:
    """Mean-squared error   L(θ) = 1/N ‖y − X_b θ‖² ."""
    return float(np.mean((y - Xb @ theta) ** 2))

def gradient(theta: np.ndarray, Xb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic ∇θ L (convex LS)."""
    return (2 / Xb.shape[0]) * Xb.T @ (Xb @ theta - y)

def gd(Xb: np.ndarray, y: np.ndarray, *,
    alpha=0.05, max_iter=5_000, tol=1e-6) -> np.ndarray:
    """
    Plain batch Gradient Descent.

    Stops when ‖∇L‖ < tol **or** max_iter reached.
    """
    theta = np.zeros(Xb.shape[1])
    prev_loss = mse(theta, Xb, y)

    for _ in range(max_iter):
      g = gradient(theta, Xb, y)
      if np.linalg.norm(g) < tol:
          break

      theta -= alpha * g

      # Check relative tolerance
      current_loss = mse(theta, Xb, y)
      if abs(prev_loss - current_loss) / (abs(prev_loss) + 1e-8) < tol:
          break
      prev_loss = current_loss

    return theta
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── FITTERS (BFGS / GD) ─────────────────╮
def fit_bfgs(Xb: np.ndarray, y: np.ndarray, print_steps=False) -> np.ndarray:
    """
    Minimise LS using quasi-Newton BFGS (SciPy).

    Returns
    -------
    θ̂ : ndarray
        Optimal coefficients.
    """
    iteration = 0
    theta0 = np.zeros(Xb.shape[1])

    if print_steps:
        def print_iteration(xk):
            nonlocal iteration
            print(f"Iteration {iteration}: {xk}")
            iteration += 1
    else:
        print_iteration = None

    return minimize(mse, theta0, args=(Xb, y), method="BFGS", callback=print_iteration).x

def fit_gd(Xb: np.ndarray, y: np.ndarray, **gd_kw) -> np.ndarray:
    """
    Wrapper to run our own GD.

    Extra kwargs (alpha, max_iter, tol) are forwarded.
    """
    return gd(Xb, y, **gd_kw)
# ╰──────────────────────────────────────────────────────────────────╯


# ╭────────────────────────── VALIDADORES (k-fold) ──────────────────╮
def kfold_mse(cols, X_norm, y, *, k: int, fit_function, **fit_kw) -> float:
    """
    Generic k-fold CV.

    Parameters
    ----------
    cols    : tuple[int]
        Indices of selected features.
    X_norm  : ndarray (N, D)
        Normalised full matrix.
    y     : ndarray (N,)
        Target vector.
    k     : int
        Number of folds (e.g. 5).
    fit_function : callable
        Function that returns θ̂ given (Xb, y).
    fit_kw  : dict
        Extra kwargs forwarded to fit_function.

    Returns
    -------
    float
        Mean MSE across folds.
    """
    N = X_norm.shape[0]
    idx = np.random.default_rng(42).permutation(N)
    folds = np.array_split(idx, k)
    losses = []

    for i in range(k):
        val, train = folds[i], np.hstack(folds[:i] + folds[i+1:])
        Xtr, ytr = X_norm[train][:, cols], y[train]
        Xv,  yv  = X_norm[val ][:, cols], y[val]
        theta = fit_function(add_bias(Xtr), ytr, **fit_kw)
        losses.append(mse(theta, add_bias(Xv), yv))

    return float(np.mean(losses))
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── SELEÇÃO DE VARIÁVEIS ────────────────────╮
def best_subset_by_R(X_norm, y, R_vals=(1, 2, 3, 4), *,
                     k=5, fit_function, **fit_kw):
    """
    For each R ∈ R_vals, test all comb(5, R) feature sets and
    return the one with minimal k-fold MSE.
    """
    result = {}
    for R in R_vals:
        best = {"cols": None, "mse": np.inf}
        # Este combinations itera sobre todas as features criadas (99),
        # não apenas as 11 originais. Se R for grande, isso levará muito tempo.
        for cols in combinations(range(X_norm.shape[1]), R):
            err = kfold_mse(cols, X_norm, y, k=k, fit_function=fit_function, **fit_kw)
            if err < best["mse"]:
                best = {"cols": cols, "mse": err}
        result[R] = best
        print(f"R={R} | CV-MSE={best['mse']:<18} | cols={best['cols']}")

    return result
# ╰──────────────────────────────────────────────────────────────────╯

# ╭────────────────────── HARDCODED MODEL GENERATOR ─────────────────╮
def generate_model(R: int) -> dict:
    """
    Returns pre-computed optimal theta values for each R based on best features.
    Calculates thetas using the actual fitters with the correct features.

    Parameters
    ----------
    R : int
        Number of features (1, 2, 3, or 4)

    Returns
    -------
    dict
        Dictionary containing 'cols', 'theta'
    """
    Xy = pd.read_csv('dados/winequality-red_treino.csv', sep=';').values
    y = Xy[:, -1]
    X = create_nonlinear_features(Xy[:, :-1])
    X_norm, mean, std = normalize(X)

    # Define the best columns for each R baseado no que foi pré-calculado ou determinado
    best_cols = {
        1: (0,),
        2: (22, 25),
        3: (3, 15, 24),
        4: (9, 14, 15, 27)
    }

    if R not in best_cols:
        raise ValueError(f"R must be one of {list(best_cols.keys())}, got {R}")

    # Get the subset of features for this R
    cols = best_cols[R]
    Xb_full = add_bias(X_norm[:, cols])

    # Calculate theta using BFGS
    theta = fit_bfgs(Xb_full, y)

    # Calculate mse for validation (opcional, já que a seleção já fez isso)
    mse_value = kfold_mse(cols, X_norm, y, k=5, fit_function=fit_bfgs)

    return {
        'cols': cols,
        'theta': theta,
        'mean': mean,
        'std': std,
        'mse': mse_value
    }
# ╰──────────────────────────────────────────────────────────────────╯

class Model:
    def __init__(self, R: int):
        self.R = R
        data = generate_model(R)
        self.cols = data['cols']
        self.theta = data['theta']
        self.mean = data['mean']
        self.std = data['std']
        self.mse = data['mse']

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using the model on new data.

        Parameters
        ----------
        X_test : ndarray (N, D)
            New data to predict (features ORIGINAIS).

        Returns
        -------
        ndarray (N,)
            Predicted values.
        """
        X_test_engineered = create_nonlinear_features(X_test)
        X_test_norm = (X_test_engineered - self.mean) / self.std
        Xb_test = add_bias(X_test_norm[:, self.cols])