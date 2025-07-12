import pytest
import pandas as pd
from unittest.mock import Mock
from abc import ABC
from src.utils.interfaces import (
    DataProcessor, DataLoader, DataSaver, Vectorizer, 
    ModelTrainer, ModelEvaluator, PipelineOrchestrator
)

class TestInterfaces:
    def test_data_processor_abc(self):
        assert issubclass(DataProcessor, ABC)

    def test_data_processor_has_process(self):
        assert hasattr(DataProcessor, 'process')

    def test_data_processor_instantiation_error(self):
        with pytest.raises(TypeError):
            DataProcessor()

    def test_data_loader_abc(self):
        assert issubclass(DataLoader, ABC)

    def test_data_loader_has_load(self):
        assert hasattr(DataLoader, 'load')

    def test_data_loader_instantiation_error(self):
        with pytest.raises(TypeError):
            DataLoader()

    def test_data_saver_abc(self):
        assert issubclass(DataSaver, ABC)

    def test_data_saver_has_save(self):
        assert hasattr(DataSaver, 'save')

    def test_data_saver_instantiation_error(self):
        with pytest.raises(TypeError):
            DataSaver()

    def test_vectorizer_abc(self):
        assert issubclass(Vectorizer, ABC)

    def test_vectorizer_has_vectorize(self):
        assert hasattr(Vectorizer, 'vectorize')

    def test_vectorizer_instantiation_error(self):
        with pytest.raises(TypeError):
            Vectorizer()

    def test_model_trainer_abc(self):
        assert issubclass(ModelTrainer, ABC)

    def test_model_trainer_has_train(self):
        assert hasattr(ModelTrainer, 'train')

    def test_model_trainer_instantiation_error(self):
        with pytest.raises(TypeError):
            ModelTrainer()

    def test_model_evaluator_abc(self):
        assert issubclass(ModelEvaluator, ABC)

    def test_model_evaluator_has_evaluate(self):
        assert hasattr(ModelEvaluator, 'evaluate')

    def test_model_evaluator_instantiation_error(self):
        with pytest.raises(TypeError):
            ModelEvaluator()

    def test_pipeline_orchestrator_abc(self):
        assert issubclass(PipelineOrchestrator, ABC)

    def test_pipeline_orchestrator_has_run(self):
        assert hasattr(PipelineOrchestrator, 'run')

    def test_pipeline_orchestrator_instantiation_error(self):
        with pytest.raises(TypeError):
            PipelineOrchestrator()

class TestConcreteImplementations:
    def test_data_processor_concrete_type(self):
        class ConcreteDataProcessor(DataProcessor):
            def process(self, data: pd.DataFrame) -> pd.DataFrame:
                return data.copy()
        processor = ConcreteDataProcessor()
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        result = processor.process(test_data)
        assert isinstance(result, pd.DataFrame)

    def test_data_processor_concrete_equals(self):
        class ConcreteDataProcessor(DataProcessor):
            def process(self, data: pd.DataFrame) -> pd.DataFrame:
                return data.copy()
        processor = ConcreteDataProcessor()
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        result = processor.process(test_data)
        assert result.equals(test_data)

    def test_data_loader_concrete_type(self):
        class ConcreteDataLoader(DataLoader):
            def load(self) -> pd.DataFrame:
                return pd.DataFrame({'col': [1, 2, 3]})
        loader = ConcreteDataLoader()
        result = loader.load()
        assert isinstance(result, pd.DataFrame)

    def test_data_loader_concrete_shape(self):
        class ConcreteDataLoader(DataLoader):
            def load(self) -> pd.DataFrame:
                return pd.DataFrame({'col': [1, 2, 3]})
        loader = ConcreteDataLoader()
        result = loader.load()
        assert result.shape == (3, 1)

    def test_data_saver_concrete_implementation(self):
        class ConcreteDataSaver(DataSaver):
            def save(self, data: pd.DataFrame, path: str) -> None:
                pass
        saver = ConcreteDataSaver()
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        saver.save(test_data, "test_path.csv")

    def test_vectorizer_concrete_implementation(self):
        class ConcreteVectorizer(Vectorizer):
            def vectorize(self, X_train, X_test):
                return Mock(), X_train, X_test
        vectorizer = ConcreteVectorizer()
        X_train = pd.DataFrame({'text': ['a', 'b']})
        X_test = pd.DataFrame({'text': ['c', 'd']})
        result = vectorizer.vectorize(X_train, X_test)
        assert len(result) == 3

    def test_model_trainer_concrete_type(self):
        class ConcreteModelTrainer(ModelTrainer):
            def train(self, X_train, y_train):
                return {'model1': Mock(), 'model2': Mock()}
        trainer = ConcreteModelTrainer()
        X_train = pd.DataFrame({'feature': [1, 2, 3]})
        y_train = pd.Series([0, 1, 0])
        result = trainer.train(X_train, y_train)
        assert isinstance(result, dict)

    def test_model_trainer_concrete_length(self):
        class ConcreteModelTrainer(ModelTrainer):
            def train(self, X_train, y_train):
                return {'model1': Mock(), 'model2': Mock()}
        trainer = ConcreteModelTrainer()
        X_train = pd.DataFrame({'feature': [1, 2, 3]})
        y_train = pd.Series([0, 1, 0])
        result = trainer.train(X_train, y_train)
        assert len(result) == 2

    def test_model_evaluator_concrete_type(self):
        class ConcreteModelEvaluator(ModelEvaluator):
            def evaluate(self, models, X_test, y_test):
                return {
                    'model1': {'accuracy': 0.8, 'precision': 0.7},
                    'model2': {'accuracy': 0.9, 'precision': 0.8}
                }
        evaluator = ConcreteModelEvaluator()
        models = {'model1': Mock(), 'model2': Mock()}
        X_test = pd.DataFrame({'feature': [1, 2]})
        y_test = pd.Series([0, 1])
        result = evaluator.evaluate(models, X_test, y_test)
        assert isinstance(result, dict)

    def test_model_evaluator_concrete_model1(self):
        class ConcreteModelEvaluator(ModelEvaluator):
            def evaluate(self, models, X_test, y_test):
                return {
                    'model1': {'accuracy': 0.8, 'precision': 0.7},
                    'model2': {'accuracy': 0.9, 'precision': 0.8}
                }
        evaluator = ConcreteModelEvaluator()
        models = {'model1': Mock(), 'model2': Mock()}
        X_test = pd.DataFrame({'feature': [1, 2]})
        y_test = pd.Series([0, 1])
        result = evaluator.evaluate(models, X_test, y_test)
        assert 'model1' in result

    def test_model_evaluator_concrete_model2(self):
        class ConcreteModelEvaluator(ModelEvaluator):
            def evaluate(self, models, X_test, y_test):
                return {
                    'model1': {'accuracy': 0.8, 'precision': 0.7},
                    'model2': {'accuracy': 0.9, 'precision': 0.8}
                }
        evaluator = ConcreteModelEvaluator()
        models = {'model1': Mock(), 'model2': Mock()}
        X_test = pd.DataFrame({'feature': [1, 2]})
        y_test = pd.Series([0, 1])
        result = evaluator.evaluate(models, X_test, y_test)
        assert 'model2' in result

    def test_pipeline_orchestrator_concrete_type(self):
        class ConcretePipelineOrchestrator(PipelineOrchestrator):
            def run(self, **kwargs):
                return {'status': 'success', 'data': 'processed'}
        orchestrator = ConcretePipelineOrchestrator()
        result = orchestrator.run(test_param='value')
        assert isinstance(result, dict)

    def test_pipeline_orchestrator_concrete_status(self):
        class ConcretePipelineOrchestrator(PipelineOrchestrator):
            def run(self, **kwargs):
                return {'status': 'success', 'data': 'processed'}
        orchestrator = ConcretePipelineOrchestrator()
        result = orchestrator.run(test_param='value')
        assert result['status'] == 'success' 