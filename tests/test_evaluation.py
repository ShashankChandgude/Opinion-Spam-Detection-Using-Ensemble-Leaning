import pytest
from src.evaluation.evaluation import evaluate_model, evaluate_models


class StubClassifier:
    def __init__(self, predictions):
        self.preds = predictions
    def predict(self, _):
        return self.preds


def test_evaluate_model_perfect():
    y = [0, 1, 1, 0]
    metrics = evaluate_model(StubClassifier(y), X_test=None, y_test=y)
    assert all(val == pytest.approx(1.0) for val in metrics.values())


def test_evaluate_model_imperfect_default_metrics():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    metrics = evaluate_model(StubClassifier(preds), None, y)
    expected = {
        'Accuracy':  3/4,
        'Precision': ((2/3) + 1.0) / 2,
        'Recall':    (1.0 + 0.5) / 2,
        'F1 Score':  (0.8 + (2/3)) / 2,
    }
    for name, val in expected.items():
        assert metrics[name] == pytest.approx(val)


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
