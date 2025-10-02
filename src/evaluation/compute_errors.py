from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def compute_errors(best_models: dict, 
                   bagging_ensembles: dict = None, 
                   stacking_ensemble=None,
                   X_train=None, X_test=None,
                   y_train=None, y_test=None,
                   metrics: dict = None):
    if metrics is None:
        metrics = {'Accuracy': accuracy_score}
    
    train_errors = {}
    test_errors = {}

    def _compute(name, model):
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)
        train_errors[name] = {m_name: 1 - m_func(y_train, y_pred_train)
                              for m_name, m_func in metrics.items()}
        test_errors[name]  = {m_name: 1 - m_func(y_test,  y_pred_test)
                              for m_name, m_func in metrics.items()}

    for name, model in best_models.items():
        _compute(name, model)

    if bagging_ensembles:
        for name, model in bagging_ensembles.items():
            _compute(f"{name} (Bagging)", model)

    if stacking_ensemble is not None:
        _compute("Stacking", stacking_ensemble)

    return train_errors, test_errors


def compute_confusion_matrix(model, X_test, y_test, labels=None):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred, labels=labels)