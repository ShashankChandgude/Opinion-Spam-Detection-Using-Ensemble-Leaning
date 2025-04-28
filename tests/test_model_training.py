import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.model_training import train_bagging_ensemble, train_stacking_ensemble, train_all_ensembles

def check_predictions(model, X, y):
    preds = model.predict(X)
    assert preds.shape == y.shape, "Prediction shape mismatch."
    assert set(preds).issubset({0, 1}), "Predicted labels should be 0 or 1."

def test_train_bagging_ensemble_predicts_labels():
    X = np.arange(6).reshape(-1,1)
    y = np.array([0,1,0,1,0,1])
    bag = train_bagging_ensemble('Decision Tree', DecisionTreeClassifier, {}, X, y)
    check_predictions(bag, X, y)

def test_train_stacking_ensemble_predicts_labels():
    X = np.arange(10).reshape(-1,1)
    y = np.array([0]*5 + [1]*5)
    stack = train_stacking_ensemble({'lr': LogisticRegression()}, X, y, meta_estimator=DecisionTreeClassifier())
    check_predictions(stack, X, y)

def test_train_all_ensembles_returns_models():
    X = np.arange(10).reshape(-1,1)
    y = np.array([0]*5 + [1]*5)
    bag, stack = train_all_ensembles({'A': DecisionTreeClassifier()}, {'A': {}}, X, y, {'A': DecisionTreeClassifier()})
    # Verify that ensemble dictionaries and stacking model behave correctly.
    assert 'A' in bag
    check_predictions(bag['A'], X, y)
    check_predictions(stack, X, y)

def test_svm_bagging_sets_probability_true():
    class FakeSVC:
        def __init__(self, **kwargs): self.params = kwargs
        def fit(self, X, y): return self
        def predict(self, X): return [0] * len(X)
        def get_params(self, deep=True): return self.params.copy()
        def set_params(self, **p): self.params.update(p); return self

    X = np.arange(4).reshape(-1,1)
    y = np.array([0,1,0,1])
    bag = train_bagging_ensemble('Support Vector Machine', FakeSVC, {}, X, y)
    assert bag.estimator.get_params().get('probability') is True