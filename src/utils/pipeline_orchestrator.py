#!/usr/bin/env python
# coding: utf-8

import os
from typing import Dict, Any
from functools import lru_cache
from src.utils.interfaces import PipelineOrchestrator as IPipelineOrchestrator
from src.utils.config import config
from src.utils.logging_config import get_logger
from src.data.data_cleaning import DataCleaner
from src.data.preprocessing import DataPreprocessor
from src.pipeline import ModelPipeline
from src.utils.logging_config import get_logger, setup_logging

class PipelineOrchestrator(IPipelineOrchestrator):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.root = self._get_project_root()
        self.data_cleaner = DataCleaner()
        self.data_preprocessor = DataPreprocessor()
        self.model_pipeline = ModelPipeline()
    
    @lru_cache(maxsize=1)
    def _get_project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def run(self, **kwargs) -> Dict[str, Any]:
        setup_logging(config.get_log_file_path(self._get_project_root()))
        self.logger.info("Starting Opinion Spam Detection Pipeline")
        try:
            self.logger.info("Stage 1: Data Cleaning")
            cleaned_data = self.data_cleaner.process()
            
            self.logger.info("Stage 2: Data Preprocessing")
            preprocessed_data = self.data_preprocessor.process(cleaned_data)
            
            self.logger.info("Stage 3: Model Training and Evaluation")
            model_results = self.model_pipeline.run(
                preprocessed_data=preprocessed_data,
                **kwargs
            )
            
            self.logger.info("Pipeline completed successfully!")
            return {
                'cleaned_data': cleaned_data,
                'preprocessed_data': preprocessed_data,
                'model_results': model_results,
                'status': 'success'
            }
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            } 