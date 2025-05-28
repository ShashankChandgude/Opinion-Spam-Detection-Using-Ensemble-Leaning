#!/usr/bin/env python
# coding: utf-8

from src.utils.helpers import os, pd, get_project_root
from src.data.data_io import  write_csv_file
from src.data.load_data import load_data
from src.utils.logging import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["hotel", "source","polarity"], errors='ignore')

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"deceptive": "label", "text": "review_text"})

def drop_duplicated_rows(df: pd.DataFrame) -> pd:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logging.info("Dropped %d duplicate rows", before - len(df))
    return df

def recode_truthful_deceptive(df: pd.DataFrame) -> pd.DataFrame:
    df['label'] = df['label'].map({'truthful': 0, 'deceptive': 1})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_irrelevant_columns(df)
    df = rename_columns(df)
    df = drop_duplicated_rows(df)
    df = recode_truthful_deceptive(df)
    return df


def pipeline() -> None:
    root = get_project_root()
    logging.info("🔹 Starting data cleaning phase")

    raw_df = load_data(root)
    logging.info("Loaded raw data: %d rows × %d cols", raw_df.shape[0], raw_df.shape[1])

    cleaned = clean_data(raw_df)
    logging.info("Cleaned data: %d rows × %d cols", cleaned.shape[0], cleaned.shape[1])

    out_path = os.path.join(root, "data", "processed", "cleaned_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_csv_file(cleaned, out_path)
    logging.info("✅ Data cleaning done, saved to %s", out_path)


if __name__ == "__main__":
    pipeline() # pragma: no cover
