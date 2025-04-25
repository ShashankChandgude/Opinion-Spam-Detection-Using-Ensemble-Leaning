#!/usr/bin/env python
# coding: utf-8

from src.utils import os, re, string, pd, plt, sns, nltk, get_project_root, WordCloud, TextBlob, stopwords, word_tokenize, PorterStemmer, plot_verified_purchase_distribution, plot_review_length_comparison
from src.data_io import load_csv_file, write_csv_file
from src.logging import logging, configure_logging

sns.set_theme(style="darkgrid")
STOPWORDS = set(stopwords.words('english'))

def drop_unwanted_cols(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(["Unnamed: 0", "review_title", "review_date"], axis=1, inplace=True, errors='ignore')
    return data

def remove_duplications(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates().reset_index(drop=True)

def compute_text_stats(data: pd.DataFrame) -> pd.DataFrame:
    data['total_words'] = data['review_text'].apply(lambda text: len(str(text).split()))
    data['total_characters'] = data['review_text'].str.len()
    data['total_stopwords'] = data['review_text'].str.split().apply(lambda tokens: len(set(tokens) & STOPWORDS))
    count_punct = lambda text: sum(1 for ch in text if ch in string.punctuation)
    data['total_punctuations'] = data['review_text'].apply(count_punct)
    data['total_uppercases'] = data['review_text'].str.findall(r'[A-Z]').str.len()
    return data

def drop_text_stats_cols(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(["total_words", "total_characters", "total_stopwords",
               "total_punctuations", "total_uppercases", "review_rating"],
              axis=1, inplace=True, errors='ignore')
    return data

def apply_text_cleaning(text: str) -> str:
    cleaned = re.sub("[^a-zA-Z]", " ", str(text))
    tokens = cleaned.lower().split()
    stemmer = PorterStemmer()
    filtered = [token for token in tokens if token not in STOPWORDS]
    stemmed = [stemmer.stem(token) for token in filtered]
    return " ".join(stemmed)

def preprocess_text(data: pd.DataFrame) -> pd.DataFrame:
    data['review_text'] = data['review_text'].apply(apply_text_cleaning)
    return data

def log_rare_words(data: pd.DataFrame) -> None:
    all_words = " ".join(data['review_text']).split()
    counts = pd.Series(all_words).value_counts()
    rare = counts[-10:]
    logging.info("Rare words:\n%s", rare.to_string())

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
    all_words = " ".join(data['review_text']).split()
    counts = pd.Series(all_words).value_counts()
    common = list(counts[:3].index)
    rare = list(counts[-3:].index)
    data['review_text'] = data['review_text'].apply(
        lambda text: " ".join(word for word in text.split() if word not in common and word not in rare)
    )
    return data

def tokenize_text(data: pd.DataFrame) -> None:
    tokens = data['review_text'].apply(word_tokenize)
    logging.info("Tokenized (first 5):\n%s", tokens.head().to_string())
    return tokens

def pipeline() -> None:
    root = get_project_root()
    input_file = os.path.join(root, "data", "processed", "updated_data.csv")
    output_file = os.path.join(root, "data", "processed", "cleaned_data.csv")
    out_folder = os.path.join(root, "output", "data_preprocessing")
    os.makedirs(out_folder, exist_ok=True)
    log_file = os.path.join(root, "output", "log.txt")
    configure_logging(log_file)
    logging.info("Starting preprocessing pipeline")
    data = load_csv_file(input_file)
    data = drop_unwanted_cols(data)
    data = remove_duplications(data)
    data = compute_text_stats(data)
    # Use common plotting functions from utils
    plot_verified_purchase_distribution(data, out_folder, "verified_purchase_distribution.png")
    plot_review_length_comparison(data, out_folder, "review_length_by_verification.png")
    data = drop_text_stats_cols(data)
    # data = correct_spelling(data)  # Optional, may be slow
    data = preprocess_text(data)
    log_rare_words(data)
    create_wordcloud(data, out_folder, "wordcloud_before.png")
    data = remove_common_rare_words(data)
    create_wordcloud(data, out_folder, "wordcloud_after.png")
    tokenize_text(data)
    write_csv_file(data, output_file)
    logging.info("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    pipeline()
