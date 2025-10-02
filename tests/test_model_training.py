#!/usr/bin/env python
# coding: utf-8

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.training.model_training import (
    ModelTrainer, 
    train_bagging_ensemble, 
    train_stacking_ensemble, 
    train_all_ensembles,
    BASE_CLASSIFIERS
)
from src.utils.config import config


class TestModelTrainer:
    """Test the ModelTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
        self.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y_train = np.array([0, 1, 0, 1])
        self.best_params = {'max_depth': 3, 'random_state': 42}
    
    @patch('src.training.model_training.classifier_registry')
    def test_train_bagging_ensemble(self, mock_registry):
        """Test bagging ensemble training."""
        # Mock the classifier creation
        mock_classifier = Mock()
        mock_registry.create_classifier_with_params.return_value = mock_classifier
        
        # Mock the BaggingClassifier
        with patch('src.training.model_training.BaggingClassifier') as mock_bagging:
            mock_ensemble = Mock()
            mock_bagging.return_value = mock_ensemble
            
            result = self.trainer.train_bagging_ensemble(
                'DecisionTreeClassifier', self.best_params, self.X_train, self.y_train
            )
            
            # Verify classifier was created with correct parameters
            mock_registry.create_classifier_with_params.assert_called_once_with(
                'DecisionTreeClassifier', self.best_params
            )
            
            # Verify BaggingClassifier was created with correct parameters
            mock_bagging.assert_called_once_with(
                estimator=mock_classifier,
                n_estimators=config.N_ESTIMATORS,
                random_state=config.RANDOM_STATE
            )
            
            # Verify ensemble was fitted
            mock_ensemble.fit.assert_called_once_with(self.X_train, self.y_train)
            
            # Verify result
            assert result == mock_ensemble
    
    def test_train_stacking_ensemble_default_params(self):
        """Test stacking ensemble training with default parameters."""
        best_models = {
            'LogisticRegression': Mock(),
            'DecisionTreeClassifier': Mock()
        }
        
        with patch('src.training.model_training.StackingClassifier') as mock_stacking:
            mock_ensemble = Mock()
            mock_stacking.return_value = mock_ensemble
            
            result = self.trainer.train_stacking_ensemble(
                best_models, self.X_train, self.y_train
            )
            
            # Verify StackingClassifier was created with correct parameters
            expected_estimators = [('LogisticRegression', best_models['LogisticRegression']),
                                 ('DecisionTreeClassifier', best_models['DecisionTreeClassifier'])]
            
            mock_stacking.assert_called_once()
            call_args = mock_stacking.call_args
            
            assert call_args[1]['estimators'] == expected_estimators
            assert isinstance(call_args[1]['final_estimator'], LogisticRegression)
            assert call_args[1]['cv'] is not None  # StratifiedKFold instance
            assert call_args[1]['n_jobs'] == -1
            
            # Verify ensemble was fitted
            mock_ensemble.fit.assert_called_once_with(self.X_train, self.y_train)
            
            # Verify result
            assert result == mock_ensemble
    
    def test_train_stacking_ensemble_custom_params(self):
        """Test stacking ensemble training with custom parameters."""
        best_models = {
            'LogisticRegression': Mock(),
            'DecisionTreeClassifier': Mock()
        }
        custom_meta_estimator = DecisionTreeClassifier()
        custom_n_splits = 3
        
        with patch('src.training.model_training.StackingClassifier') as mock_stacking:
            mock_ensemble = Mock()
            mock_stacking.return_value = mock_ensemble
            
            result = self.trainer.train_stacking_ensemble(
                best_models, self.X_train, self.y_train, 
                meta_estimator=custom_meta_estimator, n_splits=custom_n_splits
            )
            
            # Verify StackingClassifier was created with custom parameters
            call_args = mock_stacking.call_args
            assert call_args[1]['final_estimator'] == custom_meta_estimator
            
            # Verify custom CV was used
            cv_instance = call_args[1]['cv']
            assert cv_instance.n_splits == custom_n_splits
            
            # Verify ensemble was fitted
            mock_ensemble.fit.assert_called_once_with(self.X_train, self.y_train)
            
            # Verify result
            assert result == mock_ensemble
    
    @patch.object(ModelTrainer, 'train_bagging_ensemble')
    @patch.object(ModelTrainer, 'train_stacking_ensemble')
    def test_train_all_ensembles(self, mock_train_stacking, mock_train_bagging):
        """Test training all ensembles."""
        best_params_dict = {
            'LogisticRegression': {'C': 1.0},
            'DecisionTreeClassifier': {'max_depth': 3}
        }
        best_models = {
            'LogisticRegression': Mock(),
            'DecisionTreeClassifier': Mock()
        }
        
        # Mock bagging ensemble returns
        mock_bagging_lr = Mock()
        mock_bagging_dt = Mock()
        mock_train_bagging.side_effect = [mock_bagging_lr, mock_bagging_dt]
        
        # Mock stacking ensemble return
        mock_stacking = Mock()
        mock_train_stacking.return_value = mock_stacking
        
        bagging_ensembles, stacking_ensemble = self.trainer.train_all_ensembles(
            best_params_dict, self.X_train, self.y_train, best_models
        )
        
        # Verify bagging ensembles were trained for each classifier
        assert mock_train_bagging.call_count == 2
        
        # Check first call
        first_call = mock_train_bagging.call_args_list[0]
        assert first_call[0][0] == 'LogisticRegression'
        assert first_call[0][1] == {'C': 1.0}
        assert np.array_equal(first_call[0][2], self.X_train)
        assert np.array_equal(first_call[0][3], self.y_train)
        
        # Check second call
        second_call = mock_train_bagging.call_args_list[1]
        assert second_call[0][0] == 'DecisionTreeClassifier'
        assert second_call[0][1] == {'max_depth': 3}
        assert np.array_equal(second_call[0][2], self.X_train)
        assert np.array_equal(second_call[0][3], self.y_train)
        
        # Verify stacking ensemble was trained
        mock_train_stacking.assert_called_once_with(
            best_models, self.X_train, self.y_train
        )
        
        # Verify results
        expected_bagging = {
            'LogisticRegression': mock_bagging_lr,
            'DecisionTreeClassifier': mock_bagging_dt
        }
        assert bagging_ensembles == expected_bagging
        assert stacking_ensemble == mock_stacking


class TestModelTrainingFunctions:
    """Test the module-level functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y_train = np.array([0, 1, 0, 1])
        self.best_params = {'max_depth': 3, 'random_state': 42}
        self.best_models = {'LogisticRegression': Mock()}
    
    @patch('src.training.model_training.model_trainer')
    def test_train_bagging_ensemble_function(self, mock_trainer):
        """Test the train_bagging_ensemble function."""
        mock_result = Mock()
        mock_trainer.train_bagging_ensemble.return_value = mock_result
        
        result = train_bagging_ensemble(
            'DecisionTreeClassifier', None, self.best_params, self.X_train, self.y_train
        )
        
        mock_trainer.train_bagging_ensemble.assert_called_once_with(
            'DecisionTreeClassifier', self.best_params, self.X_train, self.y_train
        )
        assert result == mock_result
    
    @patch('src.training.model_training.model_trainer')
    def test_train_stacking_ensemble_function(self, mock_trainer):
        """Test the train_stacking_ensemble function."""
        mock_result = Mock()
        mock_trainer.train_stacking_ensemble.return_value = mock_result
        
        result = train_stacking_ensemble(
            self.best_models, self.X_train, self.y_train
        )
        
        mock_trainer.train_stacking_ensemble.assert_called_once_with(
            self.best_models, self.X_train, self.y_train, None, 5
        )
        assert result == mock_result
    
    @patch('src.training.model_training.model_trainer')
    def test_train_all_ensembles_function(self, mock_trainer):
        """Test the train_all_ensembles function."""
        best_params_dict = {'LogisticRegression': {'C': 1.0}}
        base_classifiers = {'LogisticRegression': Mock()}
        
        mock_bagging = {'LogisticRegression': Mock()}
        mock_stacking = Mock()
        mock_trainer.train_all_ensembles.return_value = (mock_bagging, mock_stacking)
        
        bagging_ensembles, stacking_ensemble = train_all_ensembles(
            base_classifiers, best_params_dict, self.X_train, self.y_train, self.best_models
        )
        
        mock_trainer.train_all_ensembles.assert_called_once_with(
            best_params_dict, self.X_train, self.y_train, self.best_models
        )
        assert bagging_ensembles == mock_bagging
        assert stacking_ensemble == mock_stacking
    
    def test_base_classifiers_constant(self):
        """Test that BASE_CLASSIFIERS is properly defined."""
        from src.training.model_training import BASE_CLASSIFIERS
        
        # BASE_CLASSIFIERS should be a dictionary of classifier instances
        assert isinstance(BASE_CLASSIFIERS, dict)
        assert len(BASE_CLASSIFIERS) > 0
        
        # Each value should be a classifier instance
        for name, classifier in BASE_CLASSIFIERS.items():
            assert hasattr(classifier, 'fit')
            assert hasattr(classifier, 'predict')
            assert isinstance(name, str)


class TestModelTrainingIntegration:
    """Integration tests for model training."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use larger dataset for integration tests to avoid CV issues
        self.X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
        self.y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.best_params = {'max_depth': 2, 'random_state': 42}
    
    @patch('src.training.model_training.classifier_registry')
    def test_bagging_ensemble_integration(self, mock_registry):
        """Test bagging ensemble with real classifier."""
        # Use a real DecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        mock_registry.create_classifier_with_params.return_value = DecisionTreeClassifier(
            max_depth=2, random_state=42
        )
        
        trainer = ModelTrainer()
        
        # This should work with real sklearn objects
        ensemble = trainer.train_bagging_ensemble(
            'DecisionTreeClassifier', self.best_params, self.X_train, self.y_train
        )
        
        assert isinstance(ensemble, BaggingClassifier)
        assert ensemble.n_estimators == config.N_ESTIMATORS
        assert ensemble.random_state == config.RANDOM_STATE
    
    def test_stacking_ensemble_integration(self):
        """Test stacking ensemble with real classifiers."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        best_models = {
            'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=2, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        trainer = ModelTrainer()
        
        # This should work with real sklearn objects
        ensemble = trainer.train_stacking_ensemble(
            best_models, self.X_train, self.y_train
        )
        
        assert isinstance(ensemble, StackingClassifier)
        assert len(ensemble.estimators) == 2
        assert isinstance(ensemble.final_estimator, LogisticRegression)
