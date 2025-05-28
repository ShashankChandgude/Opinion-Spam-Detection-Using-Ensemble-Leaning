#!/usr/bin/env python
# coding: utf-8

from src.utils.helpers import (os, pd, plt, sns, get_project_root, WordCloud, stopwords, word_tokenize, PorterStemmer)
from src.data.data_io import load_csv_file, write_csv_file
from src.utils.logging import logging, configure_logging
import re
import string

sns.set_theme(style="darkgrid")
STOPWORDS = set(stopwords.words('english'))

def load_cleaned_data(root: str) -> pd.DataFrame:
    path = os.path.join(root, "data", "processed", "cleaned_data.csv")
    df = load_csv_file(path)
    logging.info("Loaded cleaned data: %d rows × %d cols", *df.shape)
    return df

def save_preprocessed_data(df: pd.DataFrame, root: str) -> None:
    out_path = os.path.join(root, "data", "processed", "preprocessed_data.csv")
    write_csv_file(df, out_path)
    logging.info("Saved preprocessed data to %s", out_path)

def setup_logging(root) -> None:
    log_file = os.path.join(root, "output", "log.txt")
    configure_logging(log_file)

def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['total_words'] = df['review_text'].apply(lambda t: len(str(t).split()))
    df['total_characters'] = df['review_text'].str.len()
    df['total_stopwords'] = df['review_text'].str.split() \
        .apply(lambda toks: len(set(toks) & STOPWORDS))
    df['total_punctuations'] = df['review_text'] \
        .apply(lambda t: sum(1 for ch in t if ch in string.punctuation))
    df['total_uppercases'] = df['review_text'].str.findall(r'[A-Z]').str.len()
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    def _clean(text: str) -> str:
        cleaned = re.sub(r'[^a-zA-Z]', ' ', str(text))
        tokens = cleaned.lower().split()
        stemmer = PorterStemmer()
        filtered = [tok for tok in tokens if tok not in STOPWORDS]
        return ' '.join(stemmer.stem(tok) for tok in filtered)

    out = df.copy()
    out['review_text'] = out['review_text'].apply(_clean)
    return out

def log_token_stats(df: pd.DataFrame) -> None:
    all_tokens = ' '.join(df['review_text']).split()
    counts = pd.Series(all_tokens).value_counts()
    top10 = ', '.join(counts.head(10).index)
    bot10 = ', '.join(counts.tail(10).index)
    logging.info("10 most common tokens: %s", top10)
    logging.info("10 least common tokens: %s", bot10)

def create_wordcloud(df: pd.DataFrame, out_folder: str, fname: str) -> None:
    text = ' '.join(df['review_text'])
    wc = WordCloud(
        width=700, height=700,
        background_color='white', min_font_size=10
    ).generate(text)

    plt.figure(figsize=(5,5))
    plt.imshow(wc)
    plt.axis('off')
    path = os.path.join(out_folder, fname)
    plt.savefig(path)
    logging.info("Saved wordcloud: %s", path)
    plt.close()

def tokenize_df(df: pd.DataFrame) -> pd.Series:
    df_text = df.copy()
    tokens = df['review_text'].apply(word_tokenize)
    logging.info("Tokenized (first 5 rows):\n%s", tokens.head().to_string())
    df_text['tokens'] = tokens
    return df_text

def plot_wordcloud(df: pd.DataFrame, out_folder: str) -> None:
    create_wordcloud(df,  out_folder, "wordcloud_after.png")

def pipeline() -> None:
    root      = get_project_root()
    out_folder = os.path.join(root, "output", "data_preprocessing")
    os.makedirs(out_folder, exist_ok=True)
    setup_logging(root)
    logging.info("🔹 Starting preprocessing")

    final_df = load_cleaned_data(root).pipe(compute_text_stats).pipe(clean_df).pipe(tokenize_df)

    log_token_stats(final_df)
    plot_wordcloud(final_df, out_folder)
    save_preprocessed_data(final_df, root)

    logging.info("✅ Preprocessing done")

if __name__ == "__main__":
    pipeline()  # pragma: no cover
