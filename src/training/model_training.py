from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from src.training.classifier_config import classifier_registry
from src.utils.config import config
from src.utils.logging_config import get_logger

class ModelTrainer:
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def train_bagging_ensemble(self, classifier_name: str, best_params: dict, X_train, y_train):
        base_classifier = classifier_registry.create_classifier_with_params(classifier_name, best_params)
        
        ensemble = BaggingClassifier(
            estimator=base_classifier,
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE
        )
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def train_stacking_ensemble(self, best_models: dict, X_train, y_train, meta_estimator=None, n_splits: int = None):
        if meta_estimator is None:
            meta_estimator = LogisticRegression()
        if n_splits is None:
            n_splits = config.CV_SPLITS
            
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        estimators = [(name, model) for name, model in best_models.items()]
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_estimator,
            cv=cv,
            n_jobs=-1
        )
        stacking.fit(X_train, y_train)
        return stacking
    
    def train_all_ensembles(self, best_params_dict: dict, X_train, y_train, best_models: dict):
        bagging_ensembles = {}
        
        for name in best_params_dict.keys():
            bagging_ensembles[name] = self.train_bagging_ensemble(
                name, best_params_dict[name], X_train, y_train
            )
        
        stacking_ensemble = self.train_stacking_ensemble(best_models, X_train, y_train)
        
        return bagging_ensembles, stacking_ensemble

model_trainer = ModelTrainer()
def train_bagging_ensemble(classifier_name: str, classifier_class, best_params: dict, X_train, y_train):
    return model_trainer.train_bagging_ensemble(classifier_name, best_params, X_train, y_train)

def train_stacking_ensemble(best_models: dict, X_train, y_train, meta_estimator=None, n_splits: int = 5):
    return model_trainer.train_stacking_ensemble(best_models, X_train, y_train, meta_estimator, n_splits)

def train_all_ensembles(base_classifiers: dict, best_params_dict: dict, X_train, y_train, best_models: dict):
    return model_trainer.train_all_ensembles(best_params_dict, X_train, y_train, best_models)

BASE_CLASSIFIERS = classifier_registry.get_classifier_instances()
