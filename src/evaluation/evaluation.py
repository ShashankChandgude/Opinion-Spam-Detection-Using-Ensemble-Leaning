from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test, metric_funcs=None, average='weighted') -> dict:
    y_pred = model.predict(X_test)
    if metric_funcs is None:
        metric_funcs = {
            "Accuracy": accuracy_score,
            "Precision": lambda y, yp: precision_score(y, yp, average=average),
            "Recall":    lambda y, yp: recall_score(y, yp, average=average),
            "F1 Score":  lambda y, yp: f1_score(y, yp, average=average)
        }
    return {name: func(y_test, y_pred) for name, func in metric_funcs.items()}


def evaluate_models(models_dict: dict, X_test, y_test, **kwargs):
    return {
        name: evaluate_model(model, X_test, y_test, **kwargs)
        for name, model in models_dict.items()
    }
