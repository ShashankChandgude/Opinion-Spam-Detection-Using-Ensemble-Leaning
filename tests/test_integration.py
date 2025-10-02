#!/usr/bin/env python
# coding: utf-8

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock
from src.utils.pipeline_orchestrator import PipelineOrchestrator
from src.pipeline import ModelPipeline


class TestIntegration:
    """Integration tests that verify the complete pipeline functionality using mocking."""
    
    def test_data_flow_integrity_fast(self):
        """Fast test that verifies data flows correctly through the pipeline."""
        from src.data.data_cleaning import DataCleaner
        from src.data.preprocessing import DataPreprocessor
        
        # Test data cleaning
        cleaner = DataCleaner()
        cleaned_data = cleaner.process()
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        preprocessed_data = preprocessor.process(cleaned_data)
        
        # Verify data integrity through the pipeline
        assert len(preprocessed_data) == len(cleaned_data)
        assert 'review_text' in preprocessed_data.columns
        assert 'label' in preprocessed_data.columns
        assert len(preprocessed_data.columns) >= len(cleaned_data.columns)
    
    @patch('src.pipeline.load_and_vectorize_data')
    @patch('src.pipeline.cross_validate_models')
    @patch('src.pipeline.plot_cv_results')
    @patch('src.pipeline.optimize_all_classifiers')
    @patch('src.pipeline.train_all_ensembles')
    @patch('src.pipeline.evaluate_models')
    @patch('src.pipeline.evaluate_model')
    @patch('src.pipeline.create_results_dataframe')
    @patch('src.pipeline.plot_confusion_matrix')
    @patch('src.pipeline.compute_errors')
    @patch('src.pipeline.plot_error_curves')
    def test_pipeline_data_flow_with_mocks(self, mock_plot_error_curves, mock_compute_errors,
                                         mock_plot_confusion_matrix, mock_create_results_dataframe,
                                         mock_eval_model, mock_eval_models, 
                                         mock_train_ensembles, mock_optimize, 
                                         mock_plot_cv, mock_cv, mock_vectorize):
        """Test that preprocessed data flows correctly through the pipeline using mocks."""
        
        # Create sample data
        sample_preprocessed_data = pd.DataFrame({
            'review_text': ['great hotel', 'terrible service', 'amazing view', 'poor quality'],
            'label': [0, 1, 0, 1]
        })
        
        # Mock the vectorization to return our test data
        mock_vectorizer = MagicMock()
        mock_X_train = MagicMock()
        mock_X_test = MagicMock()
        mock_y_train = pd.Series([0, 1])
        mock_y_test = pd.Series([0, 1])
        
        mock_vectorize.return_value = (mock_vectorizer, mock_X_train, mock_X_test, 
                                     mock_y_train, mock_y_test, sample_preprocessed_data)
        
        # Mock other expensive operations - make sure cross_validate_models is completely mocked
        mock_cv.return_value = {
            'Logistic Regression': {
                'Accuracy': 0.85,
                'Precision': 0.84,
                'Recall': 0.86,
                'F1-Score': 0.85
            }
        }
        # Create a proper mock model with predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1]  # Return actual predictions
        
        # Create a proper mock stacking ensemble with predict method
        mock_stacking_ensemble = MagicMock()
        mock_stacking_ensemble.predict.return_value = [0, 1]  # Return actual predictions
        
        mock_optimize.return_value = ({'Logistic Regression': mock_model}, 
                                    {'Logistic Regression': {'C': 1.0}})
        mock_train_ensembles.return_value = ({'Logistic Regression': mock_model}, 
                                            mock_stacking_ensemble)
        mock_eval_models.return_value = {'Logistic Regression': {'Accuracy': 0.85}}
        mock_eval_model.return_value = {'Accuracy': 0.87}
        mock_plot_cv.return_value = (MagicMock(), MagicMock())  # fig, ax
        
        # Mock additional functions
        mock_create_results_dataframe.return_value = pd.DataFrame({
            'Model': ['Logistic Regression'],
            'Accuracy': [0.85],
            'Precision': [0.84],
            'Recall': [0.86],
            'F1-Score': [0.85]
        })
        mock_plot_confusion_matrix.return_value = None
        mock_compute_errors.return_value = ({'Logistic Regression': 0.1}, {'Logistic Regression': 0.15})
        mock_plot_error_curves.return_value = (MagicMock(), MagicMock())  # fig, ax
        
        # Test the pipeline
        pipeline = ModelPipeline()
        results = pipeline.run(
            preprocessed_data=sample_preprocessed_data,
            test_size=0.5,
            vectorizer_type='count'
        )
        
        # Verify that vectorization was called with our preprocessed data
        mock_vectorize.assert_called_once()
        call_args = mock_vectorize.call_args
        assert call_args[1]['preprocessed_data'] is sample_preprocessed_data
        
        # Verify results structure
        assert 'stacking_ensemble' in results
        assert 'stack_result' in results
        assert 'best_models' in results
    
    @patch('src.pipeline.ModelPipeline.run')
    @patch('src.data.preprocessing.DataPreprocessor.process')
    @patch('src.data.data_cleaning.DataCleaner.process')
    def test_orchestrator_data_flow_with_mocks(self, mock_cleaner, mock_preprocessor, mock_pipeline):
        """Test that orchestrator passes data correctly between stages using mocks."""
        
        # Create mock data
        mock_cleaned_data = pd.DataFrame({
            'review_text': ['great hotel', 'terrible service'],
            'label': [0, 1]
        })
        
        mock_preprocessed_data = pd.DataFrame({
            'review_text': ['great hotel', 'terrible service'],
            'label': [0, 1],
            'tokens': [['great', 'hotel'], ['terrible', 'service']]
        })
        
        mock_model_results = {
            'stacking_ensemble': MagicMock(),
            'stack_result': {'Stacking': {'Accuracy': 0.87}}
        }
        
        # Configure mocks
        mock_cleaner.return_value = mock_cleaned_data
        mock_preprocessor.return_value = mock_preprocessed_data
        mock_pipeline.return_value = mock_model_results
        
        # Test orchestrator
        orchestrator = PipelineOrchestrator()
        results = orchestrator.run(test_size=0.1)
        
        # Verify data flow
        mock_cleaner.assert_called_once()
        mock_preprocessor.assert_called_once_with(mock_cleaned_data)
        mock_pipeline.assert_called_once()
        
        # Verify pipeline was called with preprocessed data
        pipeline_call_args = mock_pipeline.call_args
        assert 'preprocessed_data' in pipeline_call_args[1]
        assert pipeline_call_args[1]['preprocessed_data'] is mock_preprocessed_data
        
        # Verify results
        assert results['status'] == 'success'
        assert results['cleaned_data'] is mock_cleaned_data
        assert results['preprocessed_data'] is mock_preprocessed_data
        assert results['model_results'] is mock_model_results
    
    @patch('src.pipeline.ModelPipeline.run')
    @patch('src.data.preprocessing.DataPreprocessor.process')
    @patch('src.data.data_cleaning.DataCleaner.process')
    def test_orchestrator_error_handling_with_mocks(self, mock_cleaner, mock_preprocessor, mock_pipeline):
        """Test that orchestrator handles errors gracefully using mocks."""
        
        # Test data cleaning failure
        mock_cleaner.side_effect = Exception("Data cleaning failed")
        
        orchestrator = PipelineOrchestrator()
        results = orchestrator.run(test_size=0.1)
        
        assert results['status'] == 'error'
        assert 'error' in results
        
        # Reset mocks
        mock_cleaner.reset_mock()
        mock_cleaner.side_effect = None
        mock_cleaner.return_value = pd.DataFrame({'review_text': ['test'], 'label': [0]})
        
        # Test preprocessing failure
        mock_preprocessor.side_effect = Exception("Preprocessing failed")
        
        results = orchestrator.run(test_size=0.1)
        
        assert results['status'] == 'error'
        assert 'error' in results
        
        # Reset mocks
        mock_preprocessor.reset_mock()
        mock_preprocessor.side_effect = None
        mock_preprocessor.return_value = pd.DataFrame({'review_text': ['test'], 'label': [0]})
        
        # Test pipeline failure
        mock_pipeline.side_effect = Exception("Pipeline failed")
        
        results = orchestrator.run(test_size=0.1)
        
        assert results['status'] == 'error'
        assert 'error' in results
    
    @pytest.mark.slow
    def test_pipeline_orchestrator_complete_flow(self):
        """Test that the complete pipeline orchestrator works end-to-end."""
        orchestrator = PipelineOrchestrator()
        
        # Test with very small dataset for speed
        results = orchestrator.run(test_size=0.8)  # Use 80% for test, 20% for train
        
        # Verify pipeline completed successfully
        assert results['status'] == 'success'
        assert 'cleaned_data' in results
        assert 'preprocessed_data' in results
        assert 'model_results' in results
        
        # Verify data processing worked
        assert isinstance(results['cleaned_data'], pd.DataFrame)
        assert isinstance(results['preprocessed_data'], pd.DataFrame)
        assert len(results['cleaned_data']) > 0
        assert len(results['preprocessed_data']) > 0
        
        # Verify model results
        model_results = results['model_results']
        assert 'stacking_ensemble' in model_results
        assert 'stack_result' in model_results
        
        # Verify performance metrics exist
        stack_result = model_results['stack_result']
        assert 'Stacking' in stack_result
        assert 'Accuracy' in stack_result['Stacking']
        assert stack_result['Stacking']['Accuracy'] > 0.5  # Reasonable minimum
    
    @pytest.mark.slow
    def test_model_pipeline_with_preprocessed_data(self):
        """Test that ModelPipeline correctly uses preprocessed data."""
        # Create sample preprocessed data
        sample_data = pd.DataFrame({
            'review_text': ['great hotel', 'terrible service', 'amazing view', 'poor quality'],
            'label': [0, 1, 0, 1]
        })
        
        pipeline = ModelPipeline()
        
        # Test that pipeline accepts and uses preprocessed data
        results = pipeline.run(
            preprocessed_data=sample_data,
            test_size=0.5,  # Small test size for quick execution
            vectorizer_type='count'
        )
        
        # Verify results structure
        assert 'stacking_ensemble' in results
        assert 'stack_result' in results
        assert 'best_models' in results
        
        # Verify performance metrics
        stack_result = results['stack_result']
        assert 'Stacking' in stack_result
        assert 'Accuracy' in stack_result['Stacking']
    
    @pytest.mark.slow
    def test_data_flow_integrity(self):
        """Test that data flows correctly through the pipeline."""
        orchestrator = PipelineOrchestrator()
        
        # Run pipeline
        results = orchestrator.run(test_size=0.1)
        
        # Verify data integrity through the pipeline
        cleaned_data = results['cleaned_data']
        preprocessed_data = results['preprocessed_data']
        
        # Preprocessed data should have more columns (tokens, etc.)
        assert len(preprocessed_data.columns) >= len(cleaned_data.columns)
        
        # Both should have the same number of rows (no data loss)
        assert len(preprocessed_data) == len(cleaned_data)
        
        # Required columns should exist
        assert 'review_text' in preprocessed_data.columns
        assert 'label' in preprocessed_data.columns
    
    @pytest.mark.slow
    def test_pipeline_reproducibility(self):
        """Test that pipeline results are reproducible with same random state."""
        orchestrator = PipelineOrchestrator()
        
        # Run pipeline twice with same parameters
        results1 = orchestrator.run(test_size=0.1, random_state=42)
        results2 = orchestrator.run(test_size=0.1, random_state=42)
        
        # Results should be identical (reproducible)
        assert results1['model_results']['stack_result']['Stacking']['Accuracy'] == \
               results2['model_results']['stack_result']['Stacking']['Accuracy']
    
    @pytest.mark.slow
    def test_pipeline_error_handling(self):
        """Test that pipeline handles errors gracefully."""
        pipeline = ModelPipeline()
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'review_text': [None, '', 'valid text'],
            'label': [0, 1, 0]
        })
        
        # Should handle gracefully without crashing
        try:
            results = pipeline.run(
                preprocessed_data=invalid_data,
                test_size=0.5
            )
            # If it succeeds, verify structure
            assert 'stacking_ensemble' in results
        except Exception as e:
            # If it fails, it should be a meaningful error
            assert "data" in str(e).lower() or "empty" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])
