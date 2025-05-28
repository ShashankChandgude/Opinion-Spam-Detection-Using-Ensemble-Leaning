from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.data.data_io import load_csv_file
from src.utils.helpers import get_project_root
import os

def vectorize_train_test(X_train, X_test, vectorizer_type="count"):
    if vectorizer_type == "count":
        vectorizer = CountVectorizer(stop_words='english')
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        raise ValueError("Unknown vectorizer type")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec

def load_and_vectorize_data(test_size=0.2, vectorizer_type="count", random_state=42):
    root = get_project_root()
    processed_file = os.path.join(root, "data", "processed", "cleaned_data.csv")
    data = load_csv_file(processed_file)
    # Ensure there are no NaNs in review_text
    data = data.dropna(subset=['review_text'])
    X = data['review_text']
    y = data['verified_purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    vectorizer, X_train_vec, X_test_vec = vectorize_train_test(X_train, X_test, vectorizer_type=vectorizer_type)
    return vectorizer, X_train_vec, X_test_vec, y_train, y_test, data
