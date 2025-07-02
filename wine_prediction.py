#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para treinar um modelo de regressão linear (com features não lineares) sobre o dataset winequality-red.csv.
Utiliza funções utilitárias de src/model_tools.py para pipeline de engenharia de features, normalização, treinamento e avaliação.

Etapas:
1. Carregamento dos dados
2. Engenharia de features não lineares
3. Normalização (Z-score)
4. Treinamento do modelo (BFGS)
5. Avaliação com k-fold cross-validation
6. Salvamento dos parâmetros do modelo
7. Avaliação no conjunto de teste externo e salvamento das previsões.
"""

import numpy as np
import pandas as pd
from model_tools import load_csv, create_nonlinear_features, normalize, add_bias, fit_bfgs, kfold_mse
from pathlib import Path
from logger_config import logger

df_treino = pd.read_csv('dados/winequality-red_treino.csv', sep=';')
X_treino = df_treino.iloc[:, :-1].values
y_treino = df_treino.iloc[:, -1].values

X_nl_treino = create_nonlinear_features(X_treino)

X_norm_treino, mean_norm_treino, std_norm_treino = normalize(X_nl_treino)

Xb_treino = add_bias(X_norm_treino)
theta_treino = fit_bfgs(Xb_treino, y_treino)

cols_all_features = tuple(range(X_norm_treino.shape[1]))
mse_cv = kfold_mse(cols_all_features, X_norm_treino, y_treino, k=5, fit_function=fit_bfgs)

np.savez('modelo_winequality_red.npz',
         theta=theta_treino,
         mean=mean_norm_treino,
         std=std_norm_treino)

logger.info("--- Treinamento do modelo winequality-red.csv ---")
logger.info(f"Shape X original (treino): {X_treino.shape}")
logger.info(f"Shape X_nl (features não lineares - treino): {X_nl_treino.shape}")
logger.info(f"Shape X_norm (treino): {X_norm_treino.shape}")
logger.info(f"Shape theta: {theta_treino.shape}")
logger.info(f"MSE médio (5-fold CV): {mse_cv:.4f}")
logger.info("Parâmetros do modelo salvos em 'modelo_winequality_red.npz'")

try:
    df_teste = pd.read_csv('dados/winequality-red_teste.csv', sep=';')
    X_teste = df_teste.iloc[:, :-1].values
    y_teste = df_teste.iloc[:, -1].values
    X_nl_teste = create_nonlinear_features(X_teste)
    X_norm_teste = (X_nl_teste - mean_norm_treino) / std_norm_treino
    Xb_teste = add_bias(X_norm_teste)
    y_pred_teste = Xb_teste @ theta_treino
    mse_teste = np.mean((y_teste - y_pred_teste) ** 2)
    logger.info(f"MSE no conjunto de teste externo: {mse_teste:.4f}")
    PREVISOES_DIR = Path("previsoes")
    PREVISOES_DIR.mkdir(parents=True, exist_ok=True)
    out_path_pred = PREVISOES_DIR / 'Y_pred_winequality_red_BFGS.csv'
    if len(y_pred_teste) > 0:
        np.savetxt(out_path_pred, y_pred_teste, fmt='%.6f', delimiter=',')
        logger.info(f"Arquivo de previsões salvo em: '{out_path_pred.absolute()}'")
    else:
        logger.warning("O array de previsões (y_pred_teste) está vazio. Nenhum arquivo foi salvo.")
except FileNotFoundError:
    logger.error(f"[Erro FATAL] O arquivo 'dados/winequality-red_teste.csv' não foi encontrado. Certifique-se de que ele está na pasta 'dados'.")
except Exception as e:
    logger.error(f"[Atenção] Um erro inesperado ocorreu durante a avaliação no conjunto de teste ou ao tentar salvar as previsões: {e}")