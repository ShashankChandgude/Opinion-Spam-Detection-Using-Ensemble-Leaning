#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
from typing import Type, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

@dataclass
class ClassifierConfig:
    name: str
    classifier_class: Type[BaseEstimator]
    requires_probability: bool = False
    default_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_params is None:
            self.default_params = {}

class ClassifierRegistry:
    def __init__(self):
        self._classifiers = {}
        self._register_default_classifiers()
    
    def _register_default_classifiers(self):
        """Register default classifier configurations."""
        classifiers = [
            ClassifierConfig(
                name='Logistic Regression',
                classifier_class=LogisticRegression,
                requires_probability=False
            ),
            ClassifierConfig(
                name='Decision Tree',
                classifier_class=DecisionTreeClassifier,
                requires_probability=False
            ),
            ClassifierConfig(
                name='Random Forest',
                classifier_class=RandomForestClassifier,
                requires_probability=False
            ),
            ClassifierConfig(
                name='K Nearest Neighbors',
                classifier_class=KNeighborsClassifier,
                requires_probability=False
            ),
            ClassifierConfig(
                name='Multinomial Naive Bayes',
                classifier_class=MultinomialNB,
                requires_probability=False
            ),
            ClassifierConfig(
                name='Support Vector Machine',
                classifier_class=SVC,
                requires_probability=True
            ),
            ClassifierConfig(
                name='Multilayer Perceptron',
                classifier_class=MLPClassifier,
                requires_probability=False
            )
        ]
        
        for classifier in classifiers:
            self.register_classifier(classifier)
    
    def register_classifier(self, config: ClassifierConfig):
        """Register a classifier configuration."""
        self._classifiers[config.name] = config
    
    def get_classifier_config(self, name: str) -> ClassifierConfig:
        """Get classifier configuration by name."""
        if name not in self._classifiers:
            raise ValueError(f"Unknown classifier: {name}")
        return self._classifiers[name]
    
    def get_all_classifiers(self) -> Dict[str, ClassifierConfig]:
        """Get all registered classifier configurations."""
        return self._classifiers.copy()
    
    def get_classifier_instances(self) -> Dict[str, BaseEstimator]:
        """Get instances of all classifiers with default parameters."""
        instances = {}
        for name, config in self._classifiers.items():
            instances[name] = config.classifier_class(**config.default_params)
        return instances
    
    def create_classifier_with_params(self, name: str, params: Dict[str, Any]) -> BaseEstimator:
        """Create a classifier instance with the given parameters."""
        config = self.get_classifier_config(name)
        
        if config.requires_probability and 'probability' not in params:
            params['probability'] = True
        
        return config.classifier_class(**params)

classifier_registry = ClassifierRegistry() 