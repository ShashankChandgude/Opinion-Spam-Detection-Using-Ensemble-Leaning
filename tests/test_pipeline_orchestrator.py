import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.utils.pipeline_orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator:
    def test_has_logger(self):
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, 'logger')

    def test_has_root(self):
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, 'root')

    def test_has_data_cleaner(self):
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, 'data_cleaner')

    def test_has_data_preprocessor(self):
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, 'data_preprocessor')

    def test_has_model_pipeline(self):
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, 'model_pipeline')

    def test_root_is_str(self):
        orchestrator = PipelineOrchestrator()
        assert isinstance(orchestrator.root, str)

    def test_has_run(self):
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, 'run')

    def test_run_is_callable(self):
        orchestrator = PipelineOrchestrator()
        assert callable(orchestrator.run)

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_successful_pipeline_status(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85, 'precision': 0.82}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(test_param='value')
        assert result['status'] == 'success'

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_successful_pipeline_cleaned_data(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85, 'precision': 0.82}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(test_param='value')
        assert result['cleaned_data'].equals(cleaned_data)

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_successful_pipeline_preprocessed_data(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85, 'precision': 0.82}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(test_param='value')
        assert result['preprocessed_data'].equals(preprocessed_data)

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_successful_pipeline_model_results(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85, 'precision': 0.82}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(test_param='value')
        assert result['model_results'] == model_results

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_successful_pipeline_cleaner_called(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85, 'precision': 0.82}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        orchestrator.run(test_param='value')
        mock_cleaner_instance.process.assert_called_once()

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_successful_pipeline_preprocessor_called(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85, 'precision': 0.82}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        orchestrator.run(test_param='value')
        mock_preprocessor_instance.process.assert_called_once_with(cleaned_data)

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_successful_pipeline_model_called(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85, 'precision': 0.82}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        orchestrator.run(test_param='value')
        mock_pipeline_instance.run.assert_called_once_with(preprocessed_data=preprocessed_data, test_param='value')

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_data_cleaner_failure_status(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.process.side_effect = Exception("Data cleaning failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert result['status'] == 'error'

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_data_cleaner_failure_error_present(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.process.side_effect = Exception("Data cleaning failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert 'error' in result

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_data_cleaner_failure_error_message(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.process.side_effect = Exception("Data cleaning failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert 'Data cleaning failed' in result['error']

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_data_cleaner_failure_preprocessor_called(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.process.side_effect = Exception("Data cleaning failed")
        orchestrator = PipelineOrchestrator()
        orchestrator.run()
        mock_data_preprocessor.assert_called_once()

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_data_cleaner_failure_model_called(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_cleaner_instance.process.side_effect = Exception("Data cleaning failed")
        orchestrator = PipelineOrchestrator()
        orchestrator.run()
        mock_model_pipeline.assert_called_once()

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_preprocessor_failure_status(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.side_effect = Exception("Preprocessing failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert result['status'] == 'error'

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_preprocessor_failure_error_present(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.side_effect = Exception("Preprocessing failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert 'error' in result

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_preprocessor_failure_error_message(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.side_effect = Exception("Preprocessing failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert 'Preprocessing failed' in result['error']

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_preprocessor_failure_model_called(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.side_effect = Exception("Preprocessing failed")
        orchestrator = PipelineOrchestrator()
        orchestrator.run()
        mock_model_pipeline.assert_called_once()

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_model_failure_status(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.side_effect = Exception("Model training failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert result['status'] == 'error'

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_model_failure_error_present(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.side_effect = Exception("Model training failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert 'error' in result

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_model_failure_error_message(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.side_effect = Exception("Model training failed")
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run()
        assert 'Model training failed' in result['error']

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_with_kwargs_status(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(test_size=0.3, random_state=42, cv_folds=10)
        assert result['status'] == 'success'

    @patch('src.utils.pipeline_orchestrator.ModelPipeline')
    @patch('src.utils.pipeline_orchestrator.DataPreprocessor')
    @patch('src.utils.pipeline_orchestrator.DataCleaner')
    def test_run_pipeline_with_kwargs_model_called(self, mock_data_cleaner, mock_data_preprocessor, mock_model_pipeline):
        mock_cleaner_instance = Mock()
        mock_preprocessor_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_data_cleaner.return_value = mock_cleaner_instance
        mock_data_preprocessor.return_value = mock_preprocessor_instance
        mock_model_pipeline.return_value = mock_pipeline_instance
        cleaned_data = pd.DataFrame({'col1': [1, 2, 3]})
        preprocessed_data = pd.DataFrame({'col1': [1, 2, 3], 'processed': [True, True, True]})
        model_results = {'accuracy': 0.85}
        mock_cleaner_instance.process.return_value = cleaned_data
        mock_preprocessor_instance.process.return_value = preprocessed_data
        mock_pipeline_instance.run.return_value = model_results
        orchestrator = PipelineOrchestrator()
        orchestrator.run(test_size=0.3, random_state=42, cv_folds=10)
        mock_pipeline_instance.run.assert_called_once_with(preprocessed_data=preprocessed_data, test_size=0.3, random_state=42, cv_folds=10)

    def test_get_project_root_type(self):
        orchestrator = PipelineOrchestrator()
        root = orchestrator._get_project_root()
        assert isinstance(root, str)

    def test_get_project_root_format(self):
        orchestrator = PipelineOrchestrator()
        root = orchestrator._get_project_root()
        assert root.startswith('/') or ':' in root

    def test_get_project_root_content(self):
        orchestrator = PipelineOrchestrator()
        root = orchestrator._get_project_root()
        assert 'src' in root or 'utils' in root 