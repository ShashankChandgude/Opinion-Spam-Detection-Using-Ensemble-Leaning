import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_error_curves(train_errors: dict, test_errors: dict, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure

    names = list(train_errors.keys())

    def _to_series(d):
        vals = list(d.values())
        if isinstance(vals[0], (dict,)):
            return [list(m.values())[0] for m in vals]
        return vals

    d_train = _to_series(train_errors)
    d_test  = _to_series(test_errors)

    ax.plot(names, d_train, marker='o', label='Train Error')
    ax.plot(names, d_test,  marker='o', label='Test Error')
    ax.set_xlabel('Model')
    ax.set_ylabel('Error')
    ax.set_title('Train vs. Test Error')
    ax.legend()
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def create_results_dataframe(results_dict: dict, sort_by: str = None, ascending: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(results_dict).T
    df.index.name = 'Classifier'
    df.reset_index(inplace=True)
    if sort_by and sort_by in df.columns:
        df.sort_values(by=sort_by, ascending=ascending, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def compute_confusion_values(y_true, y_pred, labels=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    return cm

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, ax=None, cmap="Blues"):
    cm = compute_confusion_values(y_true, y_pred, labels=labels, normalize=normalize)

    if ax is None:
        fig, ax = plt.subplots()
    
    hm_kwargs = {}
    if labels is not None:
        hm_kwargs['xticklabels'] = labels
        hm_kwargs['yticklabels'] = labels

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        ax=ax,
        cmap=cmap,
        **hm_kwargs
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    return ax
