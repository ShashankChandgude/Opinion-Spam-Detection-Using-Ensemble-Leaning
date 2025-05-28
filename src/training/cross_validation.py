import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def cross_validate_models(X_vec, y, classifiers: dict, n_splits=5) -> dict:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for name, clf in classifiers.items():
        accs, precs, recs, f1s = [], [], [], []
        for train_idx, test_idx in kf.split(X_vec):
            X_train_cv, X_test_cv = X_vec[train_idx], X_vec[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
            clf.fit(X_train_cv, y_train_cv)
            y_pred = clf.predict(X_test_cv)
            accs.append(accuracy_score(y_test_cv, y_pred))
            precs.append(precision_score(y_test_cv, y_pred, average='weighted'))
            recs.append(recall_score(y_test_cv, y_pred, average='weighted'))
            f1s.append(f1_score(y_test_cv, y_pred, average='weighted'))
        results[name] = {
            "Accuracy": np.mean(accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "F1 Score": np.mean(f1s)
        }
    return results

def plot_cv_results(results: dict) -> None:
    names = list(results.keys())
    metrics = list(results.values())
    d_acc = [m['Accuracy'] for m in metrics]
    d_prec = [m['Precision'] for m in metrics]
    d_rec = [m['Recall'] for m in metrics]
    d_f1  = [m['F1 Score'] for m in metrics]

    fig, ax = plt.subplots(figsize=(10,6))
    bar_width = 0.2
    idx = np.arange(len(names))
    ax.bar(idx, d_acc, bar_width, label='Accuracy')
    ax.bar(idx + bar_width, d_prec, bar_width, label='Precision')
    ax.bar(idx + 2*bar_width, d_rec, bar_width, label='Recall')
    ax.bar(idx + 3*bar_width, d_f1, bar_width, label='F1 Score')
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Classifier')
    ax.set_xticks(idx + 1.5*bar_width)
    ax.set_xticklabels(names)
    ax.legend()
    plt.tight_layout()
    plt.show()

def run_cv_and_plot(X_vec, y, classifiers: dict, n_splits=5) -> dict:
    results = cross_validate_models(X_vec, y, classifiers, n_splits)
    plot_cv_results(results)
    return results
