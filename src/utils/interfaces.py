#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.base import BaseEstimator

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass # pragma: no cover

class DataLoader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass # pragma: no cover

class DataSaver(ABC):
    @abstractmethod
    def save(self, data: pd.DataFrame, path: str) -> None:
        pass # pragma: no cover

class Vectorizer(ABC):
    @abstractmethod
    def vectorize(self, X_train, X_test) -> Tuple[BaseEstimator, Any, Any]:
        pass # pragma: no cover

class ModelTrainer(ABC):
    @abstractmethod
    def train(self, X_train, y_train) -> Dict[str, BaseEstimator]:
        pass # pragma: no cover

class ModelEvaluator(ABC):
    @abstractmethod
    def evaluate(self, models: Dict[str, BaseEstimator], X_test, y_test) -> Dict[str, Dict[str, float]]:
        pass # pragma: no cover

class PipelineOrchestrator(ABC):
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        pass # pragma: no cover