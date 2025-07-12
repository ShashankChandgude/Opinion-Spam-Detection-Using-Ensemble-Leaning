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

def test_compute_errors_only_best_models_train_a():
    train_err, test_err = compute_errors(
        {'A': DummyModel(0.8), 'B': DummyModel(1.0)}, 
        {},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert train_err["A"]["Accuracy"] == pytest.approx(0.2)

def test_compute_errors_only_best_models_train_b():
    train_err, test_err = compute_errors(
        {'A': DummyModel(0.8), 'B': DummyModel(1.0)}, 
        {},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert train_err["B"]["Accuracy"] == pytest.approx(0.0)

def test_compute_errors_only_best_models_test_a():
    train_err, test_err = compute_errors(
        {'A': DummyModel(0.8), 'B': DummyModel(1.0)}, 
        {},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert test_err["A"]["Accuracy"] == pytest.approx(0.2)

def test_compute_errors_only_best_models_test_b():
    train_err, test_err = compute_errors(
        {'A': DummyModel(0.8), 'B': DummyModel(1.0)}, 
        {},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert test_err["B"]["Accuracy"] == pytest.approx(0.0)

def test_compute_errors_with_bagging_train_c():
    train_err, test_err = compute_errors(
        {'C': DummyModel(0.75)},                      
        {'C': DummyModel(0.5)},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert train_err["C"]["Accuracy"] == pytest.approx(0.25)

def test_compute_errors_with_bagging_train_bagging():
    train_err, test_err = compute_errors(
        {'C': DummyModel(0.75)},                      
        {'C': DummyModel(0.5)},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert train_err["C (Bagging)"]["Accuracy"] == pytest.approx(0.5)

def test_compute_errors_with_bagging_test_c():
    train_err, test_err = compute_errors(
        {'C': DummyModel(0.75)},                      
        {'C': DummyModel(0.5)},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert test_err["C"]["Accuracy"] == pytest.approx(0.25)

def test_compute_errors_with_bagging_test_bagging():
    train_err, test_err = compute_errors(
        {'C': DummyModel(0.75)},                      
        {'C': DummyModel(0.5)},
        None,                                           
        X_train, X_test, y_train, y_test
    )
    assert test_err["C (Bagging)"]["Accuracy"] == pytest.approx(0.5)

def test_compute_errors_with_stacking_train_d():
    train_err, test_err = compute_errors(
        {'D': DummyModel(0.6)},                      
        {},                                           
        DummyModel(0.4),                       
        X_train, X_test, y_train, y_test
    )
    assert train_err["D"]["Accuracy"] == pytest.approx(0.4)

def test_compute_errors_with_stacking_train_stacking():
    train_err, test_err = compute_errors(
        {'D': DummyModel(0.6)},                      
        {},                                           
        DummyModel(0.4),                       
        X_train, X_test, y_train, y_test
    )
    assert train_err["Stacking"]["Accuracy"] == pytest.approx(0.6)

def test_compute_errors_with_stacking_test_d():
    train_err, test_err = compute_errors(
        {'D': DummyModel(0.6)},                      
        {},                                           
        DummyModel(0.4),                       
        X_train, X_test, y_train, y_test
    )
    assert test_err["D"]["Accuracy"] == pytest.approx(0.4)

def test_compute_errors_with_stacking_test_stacking():
    train_err, test_err = compute_errors(
        {'D': DummyModel(0.6)},                      
        {},                                           
        DummyModel(0.4),                       
        X_train, X_test, y_train, y_test
    )
    assert test_err["Stacking"]["Accuracy"] == pytest.approx(0.6)