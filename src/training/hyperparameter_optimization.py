import numpy as np
import random
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.utils.config import config
from src.utils.logging_config import get_logger

HYPERPARAMS = {
    'Logistic Regression': {
        'C': np.linspace(0.1, 100, num=10),
        'solver': ['lbfgs', 'liblinear']
    },
    'Decision Tree': {
        'max_depth': np.arange(3, 6),
        'splitter': ['best', 'random'],
        'criterion': ['gini', 'entropy']
    },
    'Random Forest': {
        'n_estimators': np.arange(100, 501, 50),
        'max_depth': np.arange(3, 6),
        'criterion': ['gini', 'entropy']
    },
    'K Nearest Neighbors': {
        'n_neighbors': range(1, 11),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'brute']
    },
    'Multinomial Naive Bayes': {
        'alpha': np.linspace(0.1, 1, num=10),
        'fit_prior': [True, False]
    },
    'Support Vector Machine': {
        'kernel': ['rbf', 'linear'],
        'C': np.linspace(0.1, 100, num=10),
        'degree': np.arange(3, 6),
        'gamma': ['scale', 'auto'],
        'probability': [True]
    },
    'Multilayer Perceptron': {
        'max_iter': [600, 800, 1000],
        'hidden_layer_sizes': [(random.randint(2, 5), random.randint(5, 30)) for _ in range(5)],
        'activation': ['relu', 'tanh']
    }
}

def optimize_hyperparameters(classifier_name: str, classifier_class, hyperparams: dict, X_train, y_train, cv_splits=None, n_iter=None):
    cv_splits = cv_splits or config.CV_SPLITS
    n_iter = n_iter or config.N_ITER
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=classifier_class(),
        param_distributions=hyperparams,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def optimize_all_classifiers(base_classifiers: dict, hyperparams: dict, X_train, y_train, cv_splits=None, n_iter=None):
    logger = get_logger(__name__)
    best_models = {}
    best_params_dict = {}
    
    for name, clf in base_classifiers.items():
        best_model, best_params = optimize_hyperparameters(
            name,
            type(clf),
            hyperparams[name],
            X_train,
            y_train,
            cv_splits,
            n_iter
        )
        best_models[name] = best_model
        best_params_dict[name] = best_params
        logger.info(f"Optimized {name}: {best_params}")
    
    return best_models, best_params_dict
