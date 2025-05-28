import pytest
import matplotlib.pyplot as plt
from src.evaluation.evaluation_visualization import create_results_dataframe, plot_error_curves

def test_results_dataframe():
    data = {'A': {'Accuracy': .9, 'Precision': .8},
            'B': {'Accuracy': .7, 'Precision': .6}}
    df = create_results_dataframe(data).set_index('Classifier')
    assert df['Accuracy'].tolist()  == pytest.approx([.9, .7])
    assert df['Precision'].tolist() == pytest.approx([.8, .6])

def test_plot_error_curves():
    plt.close()
    plot_error_curves({'A': .1, 'B': .2}, {'A': .15, 'B': .25})
    labels = {l.get_label() for l in plt.gca().get_lines()}
    assert labels == {'Train Error', 'Test Error'}