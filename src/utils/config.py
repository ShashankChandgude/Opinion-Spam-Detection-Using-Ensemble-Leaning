#!/usr/bin/env python
# coding: utf-8

import os
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class Config:
    RAW_DATA_FILE: str = "deceptive-opinion-corpus.csv"
    CLEANED_DATA_FILE: str = "cleaned_data.csv"
    PREPROCESSED_DATA_FILE: str = "preprocessed_data.csv"
    
    TEXT_COLUMN: str = "review_text"
    LABEL_COLUMN: str = "label"
    DECEPTIVE_COLUMN: str = "deceptive"
    ORIGINAL_TEXT_COLUMN: str = "text"
    
    COLUMNS_TO_DROP: List[str] = field(default_factory=lambda: ["hotel", "source", "polarity"])
    
    RANDOM_STATE: int = 42
    N_ESTIMATORS: int = 10
    TEST_SIZE: float = 0.2
    CV_SPLITS: int = 5
    N_ITER: int = 50
    
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    OUTPUT_DIR: str = "output"
    PLOTS_DIR: str = "output/plots"
    PREPROCESSING_DIR: str = "output/data_preprocessing"
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s %(levelname)s: %(message)s"
    LOG_FILE: str = "output/log.txt"
    
    DEFAULT_VECTORIZER: str = "tfidf"
    STOPWORDS_LANGUAGE: str = "english"

    def get_raw_data_path(self, root: str) -> str:
        return os.path.join(root, self.RAW_DATA_DIR, self.RAW_DATA_FILE)
    
    def get_cleaned_data_path(self, root: str) -> str:
        return os.path.join(root, self.PROCESSED_DATA_DIR, self.CLEANED_DATA_FILE)
    
    def get_preprocessed_data_path(self, root: str) -> str:
        return os.path.join(root, self.PROCESSED_DATA_DIR, self.PREPROCESSED_DATA_FILE)
    
    def get_plots_dir(self, root: str) -> str:
        return os.path.join(root, self.PLOTS_DIR)
    
    def get_preprocessing_dir(self, root: str) -> str:
        return os.path.join(root, self.PREPROCESSING_DIR)
    
    def get_log_file_path(self, root: str) -> str:
        return os.path.join(root, self.LOG_FILE)
    
    def get_results_dir(self, root: str) -> str:
        return os.path.join(root, self.OUTPUT_DIR, "runs")

config = Config() 