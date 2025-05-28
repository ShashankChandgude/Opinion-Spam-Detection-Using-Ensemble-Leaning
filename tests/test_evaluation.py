import pytest
from src.evaluation.evaluation import evaluate_model, evaluate_models

class StubClassifier:
    def __init__(self, predictions):
        self.preds = predictions
    def predict(self, _):
        return self.preds

def test_evaluate_model_perfect():
    y = [0, 1, 1, 0]
    metrics = evaluate_model(StubClassifier(y), None, y)
    assert all(val == pytest.approx(1.0) for val in metrics.values())

def test_evaluate_model_imperfect():
    y = [0, 1, 1, 0]
    predictions = [0, 1, 0, 0]
    metrics = evaluate_model(StubClassifier(predictions), None, y)
    expected = {
        'Accuracy':  0.75,
        'Precision': ((2/3) + 1) / 2,
        'Recall':    (1 + 0.5) / 2,
        'F1 Score':  (0.8 + (2/3)) / 2,
    }
    assert metrics == pytest.approx(expected)

def test_evaluate_models_wrapper():
    y = [0, 1, 0]
    model = StubClassifier(y)
    results = evaluate_models({'first': model, 'second': model}, None, y)
    expected = evaluate_model(model, None, y)
    assert results == {'first': expected, 'second': expected}