import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from src.evaluation.evaluation_visualization import (create_results_dataframe, plot_error_curves, compute_confusion_values, plot_confusion_matrix)

def test_results_dataframe():
    data = {
        'A': {'Accuracy': 0.9, 'Precision': 0.8},
        'B': {'Accuracy': 0.7, 'Precision': 0.6}
    }
    df = create_results_dataframe(data, sort_by='Accuracy').set_index('Classifier')
    assert df['Accuracy'].tolist() == pytest.approx([0.9, 0.7])
    assert df['Precision'].tolist() == pytest.approx([0.8, 0.6])

def test_results_dataframe_sorting():
    data = {'X': {'F1 Score': 0.5}, 'Y': {'F1 Score': 0.9}}
    df = create_results_dataframe(data, sort_by='F1 Score').set_index('Classifier')
    assert list(df.index) == ['Y', 'X']

def test_plot_error_curves_default_axes():
    plt.close('all')
    plot_error_curves({'A': 0.1, 'B': 0.2}, {'A': 0.15, 'B': 0.25})
    ax = plt.gca()
    labels = {line.get_label() for line in ax.get_lines()}
    assert labels == {'Train Error', 'Test Error'}
    xt = [t.get_text() for t in ax.get_xticklabels()]
    assert 'A' in xt and 'B' in xt

def test_plot_error_curves_with_supplied_ax():
    fig, ax = plt.subplots()
    fig_out, ax_out = plot_error_curves({'M': 0.3}, {'M': 0.4}, ax=ax)
    assert ax_out is ax
    lines = ax.get_lines()
    assert len(lines) == 2
    assert {l.get_label() for l in lines} == {'Train Error', 'Test Error'}

def test_compute_confusion_values_raw():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    cm = compute_confusion_values(y_true, y_pred)
    expected = np.array([[0, 2], [2, 0]])
    assert np.array_equal(cm, expected)

def test_compute_confusion_values_with_labels_and_normalize():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    cm = compute_confusion_values(y_true, y_pred, labels=[1, 0], normalize=True)
    expected = np.array([[0.5, 0.5], [0.0, 1.0]])
    assert np.allclose(cm, expected)

def test_plot_confusion_matrix_basic_and_labels():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    ax = plot_confusion_matrix(y_true, y_pred)
    assert ax.get_xlabel() == 'Predicted'
    assert ax.get_ylabel() == 'True'
    cm = compute_confusion_values(y_true, y_pred)
    mesh = ax.collections[0]
    arr = mesh.get_array().data if isinstance(mesh.get_array(), np.ma.MaskedArray) else mesh.get_array()
    arr = arr.reshape(cm.shape)
    assert np.array_equal(arr, cm)

    ax2 = plot_confusion_matrix(y_true, y_pred, labels=[1, 0])
    xt = [t.get_text() for t in ax2.get_xticklabels()]
    yt = [t.get_text() for t in ax2.get_yticklabels()]
    assert xt == ['1', '0']
    assert yt == ['1', '0']

def test_plot_confusion_matrix_normalized_and_cmap():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    fig, ax = plt.subplots()
    ax_out = plot_confusion_matrix(
        y_true, y_pred,
        labels=[0, 1],
        normalize=True,
        ax=ax,
        cmap="viridis"
    )

    assert ax_out is ax
    mesh = ax.collections[0]
    arr = mesh.get_array().data if isinstance(mesh.get_array(), np.ma.MaskedArray) else mesh.get_array()
    arr = arr.reshape((2, 2))
    row_sums = np.round(arr.sum(axis=1), 6)
    assert np.all(row_sums == 1.0)
