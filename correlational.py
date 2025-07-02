#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import seaborn as sns
from src.model_tools import load_csv, normalize, create_nonlinear_features

# --- Início do seu código modificado ---
Xy_raw = pd.read_csv('dados/winequality-red_treino.csv', sep=';').values
y_target = Xy_raw[:, -1].reshape(-1, 1)
X_original_features = Xy_raw[:, :-1]

X_engineered = create_nonlinear_features(X_original_features)

data_for_corr = np.hstack((X_engineered, y_target))

num_original_features = X_original_features.shape[1]
num_engineered_features = X_engineered.shape[1]


header = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

feature_names = []

for name in header:
    feature_names.append(name)
for name in header:
    feature_names.append(f'{name}^2')
for name in header:
    feature_names.append(f'{name}^3')
for name in header:
    feature_names.append(f'log({name})')
for i in range(len(header)):
    for j in range(i+1, len(header)):
        feature_names.append(f'{header[i]} x {header[j]}')


if len(feature_names) != num_engineered_features:
    feature_names = [f'Feature_{i+1}' for i in range(num_engineered_features)]
final_feature_names = feature_names + ['Target_y']

df_for_corr = pd.DataFrame(data_for_corr, columns=final_feature_names)
corr_matrix = df_for_corr.corr()
figsize = (max(18, int(0.25 * len(final_feature_names))), max(15, int(0.25 * len(final_feature_names))))
plt.figure(figsize=figsize)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)
plt.title("Matriz de Correlação das Features e Variável Alvo")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
Path("graficos").mkdir(parents=True, exist_ok=True)
plt.savefig("graficos/correlation_matrix_features_target.png", dpi=300)
plt.close()
X_norm, mean_norm, std_norm = normalize(X_engineered)
