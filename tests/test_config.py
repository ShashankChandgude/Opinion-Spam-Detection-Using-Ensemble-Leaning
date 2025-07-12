import os
import pytest
from src.utils.config import config, Config


class TestConfig:
    
    def test_config_raw_data_file(self):
        assert config.RAW_DATA_FILE == "deceptive-opinion-corpus.csv"
    
    def test_config_cleaned_data_file(self):
        assert config.CLEANED_DATA_FILE == "cleaned_data.csv"
    
    def test_config_preprocessed_data_file(self):
        assert config.PREPROCESSED_DATA_FILE == "preprocessed_data.csv"
    
    def test_config_text_column(self):
        assert config.TEXT_COLUMN == "review_text"
    
    def test_config_label_column(self):
        assert config.LABEL_COLUMN == "label"
    
    def test_config_deceptive_column(self):
        assert config.DECEPTIVE_COLUMN == "deceptive"
    
    def test_config_original_text_column(self):
        assert config.ORIGINAL_TEXT_COLUMN == "text"
    
    def test_config_random_state(self):
        assert config.RANDOM_STATE == 42
    
    def test_config_n_estimators(self):
        assert config.N_ESTIMATORS == 10
    
    def test_config_test_size(self):
        assert config.TEST_SIZE == 0.2
    
    def test_config_cv_splits(self):
        assert config.CV_SPLITS == 5
    
    def test_config_n_iter(self):
        assert config.N_ITER == 50
    
    def test_config_default_vectorizer(self):
        assert config.DEFAULT_VECTORIZER == "tfidf"
    
    def test_config_stopwords_language(self):
        assert config.STOPWORDS_LANGUAGE == "english"
    
    def test_columns_to_drop_default(self):
        expected_columns = ["hotel", "source", "polarity"]
        assert config.COLUMNS_TO_DROP == expected_columns
    
    def test_get_raw_data_path(self):
        test_root = "/test/root"
        raw_path = config.get_raw_data_path(test_root)
        expected_raw = os.path.join(test_root, "data", "raw", "deceptive-opinion-corpus.csv")
        assert raw_path.replace('\\', '/') == expected_raw.replace('\\', '/')
    
    def test_get_cleaned_data_path(self):
        test_root = "/test/root"
        cleaned_path = config.get_cleaned_data_path(test_root)
        expected_cleaned = os.path.join(test_root, "data", "processed", "cleaned_data.csv")
        assert cleaned_path.replace('\\', '/') == expected_cleaned.replace('\\', '/')
    
    def test_get_preprocessed_data_path(self):
        test_root = "/test/root"
        preprocessed_path = config.get_preprocessed_data_path(test_root)
        expected_preprocessed = os.path.join(test_root, "data", "processed", "preprocessed_data.csv")
        assert preprocessed_path.replace('\\', '/') == expected_preprocessed.replace('\\', '/')
    
    def test_get_plots_dir(self):
        test_root = "/test/root"
        plots_dir = config.get_plots_dir(test_root)
        expected_plots = os.path.join(test_root, "output", "plots")
        assert plots_dir.replace('\\', '/') == expected_plots.replace('\\', '/')
    
    def test_get_preprocessing_dir(self):
        test_root = "/test/root"
        preprocessing_dir = config.get_preprocessing_dir(test_root)
        expected_preprocessing = os.path.join(test_root, "output", "data_preprocessing")
        assert preprocessing_dir.replace('\\', '/') == expected_preprocessing.replace('\\', '/')
    
    def test_get_log_file_path(self):
        test_root = "/test/root"
        log_path = config.get_log_file_path(test_root)
        expected_log = os.path.join(test_root, "output", "log.txt")
        assert log_path.replace('\\', '/') == expected_log.replace('\\', '/')
    
    def test_custom_config_raw_data_file(self):
        custom_config = Config(RAW_DATA_FILE="custom.csv")
        assert custom_config.RAW_DATA_FILE == "custom.csv"
    
    def test_custom_config_random_state(self):
        custom_config = Config(RANDOM_STATE=123)
        assert custom_config.RANDOM_STATE == 123
    
    def test_custom_config_test_size(self):
        custom_config = Config(TEST_SIZE=0.3)
        assert custom_config.TEST_SIZE == 0.3
    
    def test_custom_config_inherits_defaults(self):
        custom_config = Config(RAW_DATA_FILE="custom.csv")
        assert custom_config.CLEANED_DATA_FILE == "cleaned_data.csv"
        assert custom_config.DEFAULT_VECTORIZER == "tfidf"
    
    def test_config_immutability_random_state(self):
        assert config.RANDOM_STATE == 42
    
    def test_config_immutability_test_size(self):
        assert config.TEST_SIZE == 0.2
    
    def test_config_immutability_default_vectorizer(self):
        assert config.DEFAULT_VECTORIZER == "tfidf" 