# src/features/vectorization.py

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.data.data_io import load_csv_file
from src.utils.helpers import get_project_root
import os

def vectorize_count(X_train, X_test):
    vec = CountVectorizer()
    return vec, vec.fit_transform(X_train), vec.transform(X_test)

def vectorize_tfidf(X_train, X_test):
    vec = TfidfVectorizer()
    return vec, vec.fit_transform(X_train), vec.transform(X_test)

_VECTORIZE_STRATEGIES = {
    "count": vectorize_count,
    "tfidf": vectorize_tfidf,
}

def vectorize_train_test(X_train, X_test, vectorizer_type="tfidf"):
    try:
        strategy = _VECTORIZE_STRATEGIES[vectorizer_type]
    except KeyError:
        raise ValueError(f"Unknown vectorizer_type: {vectorizer_type}")
    return strategy(X_train, X_test)

def load_and_vectorize_data(vectorizer_type="tfidf", test_size=0.2, random_state=42):
    root = get_project_root()
    processed_file = os.path.join(root, "data", "processed", "preprocessed_data.csv")
    data = load_csv_file(processed_file)
    data = data.dropna(subset=["review_text"])
    X = data["review_text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    vectorizer, X_train_vec, X_test_vec = vectorize_train_test(
        X_train, X_test, vectorizer_type
    )

    return vectorizer, X_train_vec, X_test_vec, y_train, y_test, data
