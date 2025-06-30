"""
Script para treinar um modelo de regressão linear (com features não lineares) sobre o dataset winequality-red.csv.
Utiliza funções utilitárias de model_tools.py para pipeline de engenharia de features, normalização, treinamento e avaliação.

Etapas:
1. Carregamento dos dados
2. Engenharia de features não lineares
3. Normalização (Z-score)
4. Treinamento do modelo (BFGS)
5. Avaliação com k-fold cross-validation
6. Salvamento dos parâmetros do modelo
"""

import numpy as np
import pandas as pd
from model_tools import load_csv, create_nonlinear_features, normalize, add_bias, fit_bfgs, kfold_mse

# 1. Carregar os dados do CSV (com header e separador ';')
df = pd.read_csv('dados/winequality-red_treino.csv', sep=';')
X = df.iloc[:, :-1].values  # Todas as colunas exceto a última
y = df.iloc[:, -1].values   # Última coluna (qualidade do vinho)

# 2. Criar features não lineares
X_nl = create_nonlinear_features(X)

# 3. Normalizar as features (Z-score)
X_norm, mean, std = normalize(X_nl)

# 4. Treinar o modelo usando todas as features (com bias)
Xb = add_bias(X_norm)
theta = fit_bfgs(Xb, y)

# 5. Avaliação com k-fold cross-validation (usando todas as features)
cols = tuple(range(X_norm.shape[1]))
mse_cv = kfold_mse(cols, X_norm, y, k=5, fit_function=fit_bfgs)

# 6. Salvar os parâmetros do modelo
np.savez('modelo_winequality_red.npz', theta=theta, mean=mean, std=std)

# 7. Exibir resultados
print("--- Treinamento do modelo winequality-red.csv ---")
print(f"Shape X original: {X.shape}")
print(f"Shape X_nl (features não lineares): {X_nl.shape}")
print(f"Shape X_norm: {X_norm.shape}")
print(f"Shape theta: {theta.shape}")
print(f"MSE médio (5-fold CV): {mse_cv:.4f}")
print("Parâmetros salvos em 'modelo_winequality_red.npz'")

# 8. Avaliação no conjunto de teste externo
# Carregar dados de teste
try:
    df_teste = pd.read_csv('dados/winequality-red_teste.csv', sep=';')
    X_teste = df_teste.iloc[:, :-1].values
    y_teste = df_teste.iloc[:, -1].values

    # Engenharia de features e normalização com mean/std do treino
    X_teste_nl = create_nonlinear_features(X_teste)
    X_teste_norm = (X_teste_nl - mean) / std
    Xb_teste = add_bias(X_teste_norm)
    y_pred_teste = Xb_teste @ theta

    mse_teste = np.mean((y_teste - y_pred_teste) ** 2)
    print(f"MSE no conjunto de teste externo: {mse_teste:.4f}")
except Exception as e:
    print(f"[Atenção] Não foi possível avaliar no conjunto de teste externo: {e}")
