import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
import pytest

from src.training.cross_validation import (cross_validate_models, plot_cv_results, run_cv_and_plot)

plt.show = lambda: None

def test_cross_validate_models_perfect():
    X = np.arange(6).reshape(-1, 1)
    y = pd.Series([1] * 6)
    clf = DummyClassifier(strategy='constant', constant=1)
    scores = cross_validate_models(X, y, {'c': clf}, n_splits=3)['c']
    assert all(v == pytest.approx(1.0) for v in scores.values())

def test_plot_cv_results_bars_and_labels():
    data = {'M': {'Accuracy': 0.1, 'Precision': 0.2, 'Recall': 0.3, 'F1 Score': 0.4}}
    plt.close('all')
    plot_cv_results(data)
    ax = plt.gca()
    assert len(ax.patches) == 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ['M']

def test_run_cv_and_plot_uses_cv_and_plot(monkeypatch):
    X = np.arange(4).reshape(-1, 1)
    y = pd.Series([0, 1, 0, 1])
    calls = []

    def fake_cv(Xv, yv, cls, ns):
        calls.append('cv')
        return {'x': {'Accuracy': 0.5, 'Precision': 0.5, 'Recall': 0.5, 'F1 Score': 0.5}}

    def fake_plot(results, save_path=None):
        calls.append(('plot', results))

    monkeypatch.setattr(
        'src.training.cross_validation.cross_validate_models',
        fake_cv
    )
    monkeypatch.setattr(
        'src.training.cross_validation.plot_cv_results',
        fake_plot
    )

    output = run_cv_and_plot(X, y, {}, n_splits=2)
    assert calls == ['cv', ('plot', output)]
    assert output == {'x': {'Accuracy': 0.5, 'Precision': 0.5, 'Recall': 0.5, 'F1 Score': 0.5}}
