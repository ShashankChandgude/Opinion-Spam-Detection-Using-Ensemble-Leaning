#!/usr/bin/env python
# coding: utf-8

from src.utils.helpers import os, pd, plt, sns, get_project_root, WordCloud, plot_verified_purchase_distribution, plot_review_length_comparison
from src.data.data_io import load_csv_file, write_csv_file
from src.utils.logging import logging, configure_logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

sns.set_theme(style="darkgrid")
STOPWORDS = set(stopwords.words('english'))

def drop_unwanted_cols(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(["Unnamed: 0", "review_title", "review_date"], axis=1, errors='ignore')

def remove_duplications(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates().reset_index(drop=True)

def compute_text_stats(data: pd.DataFrame) -> pd.DataFrame:
    data['total_words'] = data['review_text'].str.split().str.len()
    data['total_characters'] = data['review_text'].str.len()
    data['total_stopwords'] = data['review_text'].str.split().apply(lambda toks: len(set(toks) & STOPWORDS))
    data['total_punctuations'] = data['review_text'].apply(lambda txt: sum(1 for ch in txt if ch in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    data['total_uppercases'] = data['review_text'].str.count(r'[A-Z]')
    return data

def drop_text_stats_cols(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(["total_words", "total_characters", "total_stopwords",
                      "total_punctuations", "total_uppercases", "review_rating"],
                     axis=1, errors='ignore')

def apply_text_cleaning(text: str) -> str:
    cleaned = ''.join(ch if ch.isalpha() else ' ' for ch in text).lower()
    tokens = [w for w in cleaned.split() if w not in STOPWORDS]
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(w) for w in tokens)

def preprocess_text(data: pd.DataFrame) -> pd.DataFrame:
    data['review_text'] = data['review_text'].apply(apply_text_cleaning)
    return data

def log_rare_words(data: pd.DataFrame) -> None:
    all_words = ' '.join(data['review_text']).split()
    counts = pd.Series(all_words).value_counts()
    rare = counts.nsmallest(10).index.tolist()
    logging.info("10 least common tokens: %s", ', '.join(rare))

def create_wordcloud(data: pd.DataFrame, out_folder: str, filename: str) -> None:
    all_text = " ".join(data["review_text"])
    wc = WordCloud(width=700, height=700, background_color='white', min_font_size=10).generate(all_text)
    plt.figure(figsize=(5,5))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    path = os.path.join(out_folder, filename)
    plt.savefig(path)
    logging.info("Saved wordcloud: %s", path)
    plt.close()

def remove_common_rare_words(data: pd.DataFrame) -> pd.DataFrame:
    words = " ".join(data['review_text']).split()
    counts = pd.Series(words).value_counts()
    top3 = set(counts.nlargest(3).index)
    bot3 = set(counts.nsmallest(3).index)
    data['review_text'] = data['review_text'].apply(
        lambda txt: ' '.join(w for w in txt.split() if w not in top3 and w not in bot3)
    )
    return data

def tokenize_text(data: pd.DataFrame) -> None:
    tokens = data['review_text'].apply(word_tokenize)
    logging.info("Tokenized (first 5 rows):\n%s", tokens.head().to_string())
    return tokens

def pipeline() -> None:
    root = get_project_root()
    processed_in = os.path.join(root, "data", "processed", "updated_data.csv")
    processed_out = os.path.join(root, "data", "processed", "cleaned_data.csv")
    out_folder = os.path.join(root, "output", "data_preprocessing")
    os.makedirs(out_folder, exist_ok=True)
    log_file = os.path.join(root, "output", "log.txt")
    configure_logging(log_file)

    logging.info("🔹 Starting preprocessing")

    data = load_csv_file(processed_in)
    data = drop_unwanted_cols(data)
    data = remove_duplications(data)
    data = compute_text_stats(data)

    plot_verified_purchase_distribution(data, out_folder, "verified_purchase_distribution.png")
    plot_review_length_comparison(data, out_folder, "review_length_by_verification.png")

    data = drop_text_stats_cols(data)
    data = preprocess_text(data)
    log_rare_words(data)

    create_wordcloud(data, out_folder, "wordcloud_before.png")
    data = remove_common_rare_words(data)
    create_wordcloud(data, out_folder, "wordcloud_after.png")

    tokenize_text(data)
    write_csv_file(data, processed_out)

    logging.info("✅ Preprocessing done")

if __name__ == "__main__":
    pipeline() # pragma: no cover

