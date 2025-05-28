def compute_errors(best_models, bagging_ensembles, X_train, X_test, y_train, y_test):
    train_errors = {}
    test_errors = {}
    for name, model in best_models.items():
        train_errors[name] = 1 - model.score(X_train, y_train)
        test_errors[name] = 1 - model.score(X_test, y_test)
    for name, model in bagging_ensembles.items():
        train_errors[f"{name} (Bagging)"] = 1 - model.score(X_train, y_train)
        test_errors[f"{name} (Bagging)"] = 1 - model.score(X_test, y_test)
    return train_errors, test_errors