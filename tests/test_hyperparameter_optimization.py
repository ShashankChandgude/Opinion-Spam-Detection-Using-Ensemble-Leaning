import numpy as np
from sklearn.linear_model import LogisticRegression
import pytest
import src.training.hyperparameter_optimization as hp


def test_optimize_hyperparameters_returns_model_and_params():
    X = np.arange(8).reshape(-1,1)
    y = np.array([0]*4 + [1]*4)
    model, params = hp.optimize_hyperparameters(
        'LR', LogisticRegression, {'C':[0.1, 1.0]}, X, y, cv_splits=2, n_iter=2
    )
    assert isinstance(model, LogisticRegression)
    assert params['C'] in [0.1, 1.0]


def test_optimize_all_classifiers_collects_results():
    X = np.arange(8).reshape(-1,1)
    y = np.array([0]*4 + [1]*4)
    base_clfs = {'LR': LogisticRegression()}
    hyperparams = {'LR': {'C':[0.5]}}
    best_models, best_params = hp.optimize_all_classifiers(
        base_clfs, hyperparams, X, y, cv_splits=2, n_iter=1
    )
    assert set(best_models) == {'LR'} and set(best_params) == {'LR'}
    assert isinstance(best_models['LR'], LogisticRegression)
    assert best_params['LR']['C'] == 0.5
