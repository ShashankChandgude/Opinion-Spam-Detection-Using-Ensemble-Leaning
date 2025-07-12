import pytest
from unittest.mock import Mock, patch
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.training.classifier_config import (
    ClassifierConfig, 
    ClassifierRegistry, 
    classifier_registry
)


class TestClassifierConfig:
    
    def test_classifier_config_name(self):
        config = ClassifierConfig(
            name="Test Classifier",
            classifier_class=DecisionTreeClassifier,
            requires_probability=True,
            default_params={"max_depth": 5}
        )
        assert config.name == "Test Classifier"
    
    def test_classifier_config_classifier_class(self):
        config = ClassifierConfig(
            name="Test Classifier",
            classifier_class=DecisionTreeClassifier,
            requires_probability=True,
            default_params={"max_depth": 5}
        )
        assert config.classifier_class == DecisionTreeClassifier
    
    def test_classifier_config_requires_probability(self):
        config = ClassifierConfig(
            name="Test Classifier",
            classifier_class=DecisionTreeClassifier,
            requires_probability=True,
            default_params={"max_depth": 5}
        )
        assert config.requires_probability is True
    
    def test_classifier_config_default_params(self):
        config = ClassifierConfig(
            name="Test Classifier",
            classifier_class=DecisionTreeClassifier,
            requires_probability=True,
            default_params={"max_depth": 5}
        )
        assert config.default_params == {"max_depth": 5}
    
    def test_classifier_config_default_params_none(self):
        config = ClassifierConfig(
            name="Test Classifier",
            classifier_class=DecisionTreeClassifier
        )
        assert config.default_params == {}
    
    def test_classifier_config_post_init(self):
        config = ClassifierConfig(
            name="Test Classifier",
            classifier_class=DecisionTreeClassifier,
            default_params=None
        )
        assert config.default_params == {}


class TestClassifierRegistry:
    
    def test_classifier_registry_initialization(self):
        registry = ClassifierRegistry()
        assert isinstance(registry._classifiers, dict)
        assert len(registry._classifiers) > 0
    
    def test_register_default_classifiers_logistic_regression(self):
        registry = ClassifierRegistry()
        classifiers = registry.get_all_classifiers()
        assert 'Logistic Regression' in classifiers
        assert isinstance(classifiers['Logistic Regression'], ClassifierConfig)
    
    def test_register_default_classifiers_decision_tree(self):
        registry = ClassifierRegistry()
        classifiers = registry.get_all_classifiers()
        assert 'Decision Tree' in classifiers
        assert isinstance(classifiers['Decision Tree'], ClassifierConfig)
    
    def test_register_default_classifiers_random_forest(self):
        registry = ClassifierRegistry()
        classifiers = registry.get_all_classifiers()
        assert 'Random Forest' in classifiers
        assert isinstance(classifiers['Random Forest'], ClassifierConfig)
    
    def test_register_default_classifiers_knn(self):
        registry = ClassifierRegistry()
        classifiers = registry.get_all_classifiers()
        assert 'K Nearest Neighbors' in classifiers
        assert isinstance(classifiers['K Nearest Neighbors'], ClassifierConfig)
    
    def test_register_default_classifiers_naive_bayes(self):
        registry = ClassifierRegistry()
        classifiers = registry.get_all_classifiers()
        assert 'Multinomial Naive Bayes' in classifiers
        assert isinstance(classifiers['Multinomial Naive Bayes'], ClassifierConfig)
    
    def test_register_default_classifiers_svm(self):
        registry = ClassifierRegistry()
        classifiers = registry.get_all_classifiers()
        assert 'Support Vector Machine' in classifiers
        assert isinstance(classifiers['Support Vector Machine'], ClassifierConfig)
    
    def test_register_default_classifiers_mlp(self):
        registry = ClassifierRegistry()
        classifiers = registry.get_all_classifiers()
        assert 'Multilayer Perceptron' in classifiers
        assert isinstance(classifiers['Multilayer Perceptron'], ClassifierConfig)
    
    def test_register_classifier(self):
        registry = ClassifierRegistry()
        custom_config = ClassifierConfig(
            name="Custom Classifier",
            classifier_class=DecisionTreeClassifier,
            requires_probability=False
        )
        registry.register_classifier(custom_config)
        assert "Custom Classifier" in registry._classifiers
        assert registry._classifiers["Custom Classifier"] == custom_config
    
    def test_get_classifier_config_existing_name(self):
        registry = ClassifierRegistry()
        config = registry.get_classifier_config("Decision Tree")
        assert config.name == "Decision Tree"
    
    def test_get_classifier_config_existing_class(self):
        registry = ClassifierRegistry()
        config = registry.get_classifier_config("Decision Tree")
        assert config.classifier_class == DecisionTreeClassifier
    
    def test_get_classifier_config_existing_probability(self):
        registry = ClassifierRegistry()
        config = registry.get_classifier_config("Decision Tree")
        assert config.requires_probability is False
    
    def test_get_classifier_config_nonexistent(self):
        registry = ClassifierRegistry()
        with pytest.raises(ValueError, match="Unknown classifier: NonExistent"):
            registry.get_classifier_config("NonExistent")
    
    def test_get_all_classifiers_returns_copy(self):
        registry = ClassifierRegistry()
        classifiers1 = registry.get_all_classifiers()
        classifiers2 = registry.get_all_classifiers()
        assert classifiers1 is not classifiers2
        assert classifiers1 == classifiers2
    
    def test_get_classifier_instances_returns_dict(self):
        registry = ClassifierRegistry()
        instances = registry.get_classifier_instances()
        assert isinstance(instances, dict)
        assert len(instances) > 0
    
    def test_get_classifier_instances_has_fit_method(self):
        registry = ClassifierRegistry()
        instances = registry.get_classifier_instances()
        for name, instance in instances.items():
            assert hasattr(instance, 'fit')
    
    def test_get_classifier_instances_has_predict_method(self):
        registry = ClassifierRegistry()
        instances = registry.get_classifier_instances()
        for name, instance in instances.items():
            assert hasattr(instance, 'predict')
    
    def test_create_classifier_with_params_basic_type(self):
        registry = ClassifierRegistry()
        params = {"max_depth": 5, "random_state": 42}
        classifier = registry.create_classifier_with_params("Decision Tree", params)
        assert isinstance(classifier, DecisionTreeClassifier)
    
    def test_create_classifier_with_params_basic_max_depth(self):
        registry = ClassifierRegistry()
        params = {"max_depth": 5, "random_state": 42}
        classifier = registry.create_classifier_with_params("Decision Tree", params)
        assert classifier.max_depth == 5
    
    def test_create_classifier_with_params_basic_random_state(self):
        registry = ClassifierRegistry()
        params = {"max_depth": 5, "random_state": 42}
        classifier = registry.create_classifier_with_params("Decision Tree", params)
        assert classifier.random_state == 42
    
    def test_create_classifier_with_params_svm_probability(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0, "kernel": "rbf"}
        classifier = registry.create_classifier_with_params("Support Vector Machine", params)
        assert isinstance(classifier, SVC)
        assert classifier.probability is True
    
    def test_create_classifier_with_params_svm_c_value(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0, "kernel": "rbf"}
        classifier = registry.create_classifier_with_params("Support Vector Machine", params)
        assert classifier.C == 1.0
    
    def test_create_classifier_with_params_svm_kernel(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0, "kernel": "rbf"}
        classifier = registry.create_classifier_with_params("Support Vector Machine", params)
        assert classifier.kernel == "rbf"
    
    def test_create_classifier_with_params_svm_probability_already_set(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0, "probability": False}
        classifier = registry.create_classifier_with_params("Support Vector Machine", params)
        assert isinstance(classifier, SVC)
        assert classifier.probability is False
    
    def test_create_classifier_with_params_svm_c_value_already_set(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0, "probability": False}
        classifier = registry.create_classifier_with_params("Support Vector Machine", params)
        assert classifier.C == 1.0
    
    def test_create_classifier_with_params_non_probability_classifier_type(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0}
        classifier = registry.create_classifier_with_params("Logistic Regression", params)
        assert isinstance(classifier, LogisticRegression)
    
    def test_create_classifier_with_params_non_probability_classifier_c_value(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0}
        classifier = registry.create_classifier_with_params("Logistic Regression", params)
        assert classifier.C == 1.0
    
    def test_create_classifier_with_params_non_probability_classifier_no_probability_attr(self):
        registry = ClassifierRegistry()
        params = {"C": 1.0}
        classifier = registry.create_classifier_with_params("Logistic Regression", params)
        assert not hasattr(classifier, 'probability')


class TestGlobalClassifierRegistry:
    
    def test_global_registry_exists(self):
        assert classifier_registry is not None
        assert isinstance(classifier_registry, ClassifierRegistry)
    
    def test_global_registry_has_classifiers(self):
        classifiers = classifier_registry.get_all_classifiers()
        assert len(classifiers) > 0
    
    def test_global_registry_has_logistic_regression(self):
        assert 'Logistic Regression' in classifier_registry._classifiers
    
    def test_global_registry_has_decision_tree(self):
        assert 'Decision Tree' in classifier_registry._classifiers
    
    def test_global_registry_has_random_forest(self):
        assert 'Random Forest' in classifier_registry._classifiers
    
    def test_global_registry_has_knn(self):
        assert 'K Nearest Neighbors' in classifier_registry._classifiers
    
    def test_global_registry_has_naive_bayes(self):
        assert 'Multinomial Naive Bayes' in classifier_registry._classifiers
    
    def test_global_registry_has_svm(self):
        assert 'Support Vector Machine' in classifier_registry._classifiers
    
    def test_global_registry_has_mlp(self):
        assert 'Multilayer Perceptron' in classifier_registry._classifiers
    
    def test_global_registry_svm_requires_probability(self):
        svm_config = classifier_registry.get_classifier_config("Support Vector Machine")
        assert svm_config.requires_probability is True
    
    def test_global_registry_logistic_regression_no_probability(self):
        config = classifier_registry.get_classifier_config("Logistic Regression")
        assert config.requires_probability is False
    
    def test_global_registry_decision_tree_no_probability(self):
        config = classifier_registry.get_classifier_config("Decision Tree")
        assert config.requires_probability is False
    
    def test_global_registry_random_forest_no_probability(self):
        config = classifier_registry.get_classifier_config("Random Forest")
        assert config.requires_probability is False
    
    def test_global_registry_knn_no_probability(self):
        config = classifier_registry.get_classifier_config("K Nearest Neighbors")
        assert config.requires_probability is False
    
    def test_global_registry_naive_bayes_no_probability(self):
        config = classifier_registry.get_classifier_config("Multinomial Naive Bayes")
        assert config.requires_probability is False
    
    def test_global_registry_mlp_no_probability(self):
        config = classifier_registry.get_classifier_config("Multilayer Perceptron")
        assert config.requires_probability is False 