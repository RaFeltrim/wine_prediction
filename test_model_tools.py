#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from src.model_tools import create_nonlinear_features, normalize, add_bias

def test_create_nonlinear_features_shape():
    X = np.random.rand(10, 3)
    X_nl = create_nonlinear_features(X)
    assert X_nl.shape[0] == 10
    assert X_nl.shape[1] > 3  # Deve criar mais features

def test_normalize_mean_std():
    X = np.random.rand(100, 5)
    X_norm, mean, std = normalize(X)
    np.testing.assert_allclose(X_norm.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(X_norm.std(axis=0), 1, atol=1e-7)

def test_add_bias():
    X = np.ones((4, 2))
    Xb = add_bias(X)
    assert np.all(Xb[:, 0] == 1)
    assert Xb.shape[1] == 3
