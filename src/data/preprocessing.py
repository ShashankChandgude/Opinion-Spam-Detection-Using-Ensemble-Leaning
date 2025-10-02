#!/usr/bin/env python
# coding: utf-8

import re
import string
import pandas as pd
from pathlib import Path
from functools import lru_cache
from src.utils.interfaces import DataProcessor
from src.utils.config import config
from src.utils.logging_config import get_logger
from src.utils.helpers import get_project_root, WordCloud, stopwords, word_tokenize, PorterStemmer
from src.data.data_io import load_csv_file, write_csv_file
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
STOPWORDS = set(stopwords.words(config.STOPWORDS_LANGUAGE))
PUNCTUATION_SET = set(string.punctuation)
STEMMER = PorterStemmer()

class DataPreprocessor(DataProcessor):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.root = get_project_root()
    def load_cleaned_data(self) -> pd.DataFrame:
        path = config.get_cleaned_data_path(self.root)
        df = load_csv_file(path)
        self.logger.info("Loaded cleaned data: %d rows × %d cols", *df.shape)
        return df
    def save_preprocessed_data(self, df: pd.DataFrame) -> None:
        out_path = config.get_preprocessed_data_path(self.root)
        write_csv_file(df, out_path)
        self.logger.info("Saved preprocessed data to %s", out_path)
    def compute_text_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        text_col = df_copy[config.TEXT_COLUMN]
        df_copy['total_words'] = text_col.str.split().str.len()
        df_copy['total_characters'] = text_col.str.len()
        df_copy['total_stopwords'] = text_col.str.split().apply(lambda toks: len(set(toks) & STOPWORDS))
        df_copy['total_punctuations'] = text_col.apply(lambda t: sum(1 for ch in t if ch in PUNCTUATION_SET))
        df_copy['total_uppercases'] = text_col.str.findall(r'[A-Z]').str.len()
        return df_copy
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        def _clean(text: str) -> str:
            cleaned = re.sub(r'[^a-zA-Z]', ' ', str(text))
            tokens = cleaned.lower().split()
            filtered = [tok for tok in tokens if tok not in STOPWORDS]
            return ' '.join(STEMMER.stem(tok) for tok in filtered)
        df_copy = df.copy()
        df_copy[config.TEXT_COLUMN] = df_copy[config.TEXT_COLUMN].apply(_clean)
        return df_copy
    def log_token_stats(self, df: pd.DataFrame) -> None:
        all_tokens = ' '.join(df[config.TEXT_COLUMN]).split()
        counts = pd.Series(all_tokens).value_counts()
        top10 = ', '.join(counts.head(10).index)
        bot10 = ', '.join(counts.tail(10).index)
        self.logger.info("10 most common tokens: %s", top10)
        self.logger.info("10 least common tokens: %s", bot10)
    def create_wordcloud(self, df: pd.DataFrame, out_folder: str, fname: str) -> None:
        text = ' '.join(df[config.TEXT_COLUMN])
        try:
            wc = WordCloud(
                width=700, height=700,
                background_color='white', min_font_size=10,
                collocations=False
            ).generate(text)
            plt.figure(figsize=(5,5))
            plt.imshow(wc)
            plt.axis('off')
            path = Path(out_folder) / fname
            plt.savefig(path)
            self.logger.info("Saved wordcloud: %s", path)
            plt.close()
        except Exception as e:
            self.logger.warning(f"WordCloud generation failed: {e}. Skipping wordcloud creation.")
    def tokenize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        tokens = df_copy[config.TEXT_COLUMN].apply(word_tokenize)
        self.logger.info("Tokenized (first 5 rows):\n%s", tokens.head().to_string())
        df_copy['tokens'] = tokens
        return df_copy
    def process(self, data: pd.DataFrame = None) -> pd.DataFrame:
        if data is None:
            self.logger.info("Starting preprocessing")
            data = self.load_cleaned_data()
        out_folder = config.get_preprocessing_dir(self.root)
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        final_df = (data
                   .pipe(self.compute_text_stats)
                   .pipe(self.clean_text)
                   .pipe(self.tokenize_text))
        self.log_token_stats(final_df)
        self.create_wordcloud(final_df, out_folder, "wordcloud_after.png")
        self.save_preprocessed_data(final_df)
        self.logger.info("Preprocessing completed successfully")
        return final_df

def pipeline() -> None:
    preprocessor = DataPreprocessor()
    preprocessor.process()

if __name__ == "__main__":
    pipeline()
