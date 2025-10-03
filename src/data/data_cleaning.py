#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
from functools import lru_cache
from src.utils.interfaces import DataProcessor, DataLoader, DataSaver
from src.utils.config import config
from src.utils.logging_config import get_logger
from src.utils.helpers import get_project_root
from src.data.data_io import write_csv_file
from src.data.load_data import FileDataLoader

class DataCleaner(DataProcessor):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.root = get_project_root()
        self.data_loader = FileDataLoader()
        self.data_saver = FileDataSaver()
    def drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=config.COLUMNS_TO_DROP, errors='ignore')
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            config.DECEPTIVE_COLUMN: config.LABEL_COLUMN,
            config.ORIGINAL_TEXT_COLUMN: config.TEXT_COLUMN
        }
        return df.rename(columns=column_mapping)
    def drop_duplicated_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        self.logger.info("Dropped %d duplicate rows", before - len(df_cleaned))
        return df_cleaned
    def recode_truthful_deceptive(self, df: pd.DataFrame) -> pd.DataFrame:
        label_mapping = {'truthful': 0, 'deceptive': 1}
        df_copy = df.copy()
        df_copy[config.LABEL_COLUMN] = df_copy[config.LABEL_COLUMN].map(label_mapping)
        return df_copy
    def process(self, data: pd.DataFrame = None) -> pd.DataFrame:
        if data is None:
            self.logger.info("Starting data cleaning phase")
            data = self.data_loader.load()
            self.logger.info("Loaded raw data: %d rows × %d cols", data.shape[0], data.shape[1])
        cleaned = (data
                  .pipe(self.drop_irrelevant_columns)
                  .pipe(self.rename_columns)
                  .pipe(self.drop_duplicated_rows)
                  .pipe(self.recode_truthful_deceptive))
        self.logger.info("Cleaned data: %d rows × %d cols", cleaned.shape[0], cleaned.shape[1])
        out_path = config.get_cleaned_data_path(self.root)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self.data_saver.save(cleaned, out_path)
        self.logger.info("Data cleaning completed successfully, saved to %s", out_path)
        return cleaned

class FileDataSaver(DataSaver):
    def save(self, data: pd.DataFrame, path: str) -> None:
        write_csv_file(data, path)

def pipeline() -> None:
    cleaner = DataCleaner() # pragma: no cover
    cleaner.process() # pragma: no cover

if __name__ == "__main__":
    pipeline() # pragma: no cover
