from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier

# Define base classifiers as a constant
BASE_CLASSIFIERS = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K Nearest Neighbors': KNeighborsClassifier(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(),
    'Multilayer Perceptron': MLPClassifier()
}

def train_bagging_ensemble(classifier_name: str, classifier_class, best_params: dict, X_train, y_train):
    if classifier_name == 'Support Vector Machine':
        best_params['probability'] = True
    ensemble = BaggingClassifier(estimator=classifier_class(**best_params),
                                 n_estimators=10, random_state=42)
    ensemble.fit(X_train, y_train)
    return ensemble

def train_stacking_ensemble(best_models: dict, X_train, y_train, meta_estimator=None, n_splits: int = 5):
    if meta_estimator is None:
        meta_estimator = LogisticRegression()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    estimators = [(name, model) for name, model in best_models.items()]
    stacking = StackingClassifier(estimators=estimators, final_estimator=meta_estimator, cv=cv, n_jobs=-1)
    stacking.fit(X_train, y_train)
    return stacking

def train_all_ensembles(base_classifiers: dict, best_params_dict: dict, X_train, y_train, best_models: dict):
    bagging_ensembles = {}
    for name, clf in base_classifiers.items():
        if name in best_params_dict:
            bagging_ensembles[name] = train_bagging_ensemble(name, type(clf), best_params_dict[name], X_train, y_train)
    stacking_ensemble = train_stacking_ensemble(best_models, X_train, y_train)
    return bagging_ensembles, stacking_ensemble
