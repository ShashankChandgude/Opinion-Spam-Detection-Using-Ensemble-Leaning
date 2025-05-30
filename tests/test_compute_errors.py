from src.evaluation.compute_errors import compute_errors
import pytest

N = 20
X_train = X_test = list(range(N))
y_train = y_test = [1] * N

class DummyModel:
    def __init__(self, score): 
        self._score = score
    def score(self, X, y): 
        return self._score
    def predict(self, X):
        n = len(X)
        correct = int(self._score * n)
        return [1] * correct + [0] * (n - correct)

def _check(best_models, bagging, stacking, exp_train, exp_test):
    train_err, test_err = compute_errors(best_models, bagging, stacking, X_train, X_test, y_train, y_test)
    
    for name, expected in exp_train.items():
        assert train_err[name]["Accuracy"] == pytest.approx(expected)
    for name, expected in exp_test.items():
        assert test_err[name]["Accuracy"] == pytest.approx(expected)

def test_compute_errors_only_best_models():
    _check(
        {'A': DummyModel(0.8), 'B': DummyModel(1.0)}, 
        {},
        None,                                           
        {'A': 0.2, 'B': 0.0},                        
        {'A': 0.2, 'B': 0.0},                         
    )

def test_compute_errors_with_bagging():
    _check(
        {'C': DummyModel(0.75)},                      
        {'C': DummyModel(0.5)},
        None,                                           
        {'C': 0.25, 'C (Bagging)': 0.5},              
        {'C': 0.25, 'C (Bagging)': 0.5},             
    )

def test_compute_errors_with_stacking():
    _check(
        {'D': DummyModel(0.6)},                      
        {},                                           
        DummyModel(0.4),                       
        {'D': 0.4, 'Stacking': 0.6},              
        {'D': 0.4, 'Stacking': 0.6},             
    )