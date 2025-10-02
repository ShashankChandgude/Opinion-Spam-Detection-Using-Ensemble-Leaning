from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
import pandas as pd
from typing import Tuple, Optional, Union
from functools import lru_cache
from src.data.data_io import load_csv_file
from src.utils.helpers import get_project_root
from src.utils.config import config
import os

def vectorize_count(X_train: pd.Series, X_test: pd.Series) -> Tuple[CountVectorizer, csr_matrix, csr_matrix]:
    vec = CountVectorizer()
    return vec, vec.fit_transform(X_train), vec.transform(X_test)

def vectorize_tfidf(X_train: pd.Series, X_test: pd.Series) -> Tuple[TfidfVectorizer, csr_matrix, csr_matrix]:
    vec = TfidfVectorizer()
    return vec, vec.fit_transform(X_train), vec.transform(X_test)

_VECTORIZE_STRATEGIES = {
    "count": vectorize_count,
    "tfidf": vectorize_tfidf,
}

def vectorize_train_test(X_train: pd.Series, X_test: pd.Series, vectorizer_type: Optional[str] = None) -> Tuple[Union[CountVectorizer, TfidfVectorizer], csr_matrix, csr_matrix]:
    vectorizer_type = vectorizer_type or config.DEFAULT_VECTORIZER
    
    try:
        strategy = _VECTORIZE_STRATEGIES[vectorizer_type]
    except KeyError:
        raise ValueError(f"Unknown vectorizer_type: {vectorizer_type}")
    return strategy(X_train, X_test)

def load_and_vectorize_data(preprocessed_data: Optional[pd.DataFrame] = None, vectorizer_type: Optional[str] = None, test_size: Optional[float] = None, random_state: Optional[int] = None) -> Tuple[Union[CountVectorizer, TfidfVectorizer], csr_matrix, csr_matrix, pd.Series, pd.Series, pd.DataFrame]:
    vectorizer_type = vectorizer_type or config.DEFAULT_VECTORIZER
    test_size = test_size or config.TEST_SIZE
    random_state = random_state or config.RANDOM_STATE
    
    # Use provided preprocessed data or load from file
    if preprocessed_data is not None:
        data = preprocessed_data.copy()
    else:
        root = get_project_root()
        processed_file = config.get_preprocessed_data_path(root)
        data = load_csv_file(processed_file)
    
    data = data.dropna(subset=[config.TEXT_COLUMN])
    X = data[config.TEXT_COLUMN]
    y = data[config.LABEL_COLUMN]

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
