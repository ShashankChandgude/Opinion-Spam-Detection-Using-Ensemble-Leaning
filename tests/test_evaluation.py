import pytest
from src.evaluation.evaluation import evaluate_model, evaluate_models


class StubClassifier:
    def __init__(self, predictions):
        self.preds = predictions
    def predict(self, _):
        return self.preds


def test_evaluate_model_perfect_accuracy():
    y = [0, 1, 1, 0]
    metrics = evaluate_model(StubClassifier(y), X_test=None, y_test=y)
    assert metrics['Accuracy'] == pytest.approx(1.0)

def test_evaluate_model_perfect_precision():
    y = [0, 1, 1, 0]
    metrics = evaluate_model(StubClassifier(y), X_test=None, y_test=y)
    assert metrics['Precision'] == pytest.approx(1.0)

def test_evaluate_model_perfect_recall():
    y = [0, 1, 1, 0]
    metrics = evaluate_model(StubClassifier(y), X_test=None, y_test=y)
    assert metrics['Recall'] == pytest.approx(1.0)

def test_evaluate_model_perfect_f1_score():
    y = [0, 1, 1, 0]
    metrics = evaluate_model(StubClassifier(y), X_test=None, y_test=y)
    assert metrics['F1 Score'] == pytest.approx(1.0)

def test_evaluate_model_imperfect_accuracy():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    metrics = evaluate_model(StubClassifier(preds), None, y)
    assert metrics['Accuracy'] == pytest.approx(3/4)

def test_evaluate_model_imperfect_precision():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    metrics = evaluate_model(StubClassifier(preds), None, y)
    expected_precision = ((2/3) + 1.0) / 2
    assert metrics['Precision'] == pytest.approx(expected_precision)

def test_evaluate_model_imperfect_recall():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    metrics = evaluate_model(StubClassifier(preds), None, y)
    expected_recall = (1.0 + 0.5) / 2
    assert metrics['Recall'] == pytest.approx(expected_recall)

def test_evaluate_model_imperfect_f1_score():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    metrics = evaluate_model(StubClassifier(preds), None, y)
    expected_f1 = (0.8 + (2/3)) / 2
    assert metrics['F1 Score'] == pytest.approx(expected_f1)

def test_evaluate_model_custom_metric_funcs():
    y = [0, 1, 0, 1]
    preds = [1, 1, 0, 0]
    custom = {'CountOnes': lambda y, yp: sum(1 for p in yp if p == 1)}
    metrics = evaluate_model(StubClassifier(preds), None, y, metric_funcs=custom)
    assert metrics == {'CountOnes': 2}

def test_evaluate_models_wrapper_forwards_kwargs():
    y = [0, 1, 0]
    preds = [0, 1, 0]
    model = StubClassifier(preds)
    results = evaluate_models({'first': model, 'second': model}, None, y)
    expected = evaluate_model(model, None, y)
    assert results == {'first': expected, 'second': expected}
