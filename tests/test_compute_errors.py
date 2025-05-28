from src.evaluation.compute_errors import compute_errors
import pytest

class DummyModel:
    def __init__(self, score): 
        self._score = score
    def score(self, X, y): 
        return self._score

X_train = X_test = y_train = y_test = object()

def _check(best_models, bagging, exp_train, exp_test):
    train_err, test_err = compute_errors(
        best_models, bagging, X_train, X_test, y_train, y_test
    )
    for key, val in exp_train.items():
        assert train_err[key] == pytest.approx(val)
    for key, val in exp_test.items():
        assert test_err[key] == pytest.approx(val)

def test_compute_errors_only_best_models():
    _check(
        {'A': DummyModel(0.8), 'B': DummyModel(1.0)}, 
        {},                                           
        {'A': 0.2, 'B': 0.0},                        
        {'A': 0.2, 'B': 0.0},                         
    )

def test_compute_errors_with_bagging():
    _check(
        {'C': DummyModel(0.75)},                      
        {'C': DummyModel(0.5)},                      
        {'C': 0.25, 'C (Bagging)': 0.5},              
        {'C': 0.25, 'C (Bagging)': 0.5},             
    )