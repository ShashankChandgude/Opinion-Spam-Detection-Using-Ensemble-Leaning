import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import src.features.vectorization as vz

def test_vectorize_count_produces_count_vectorizer():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["spam ham test", "ham test case"], 
        ["spam ham test"]
    )
    assert isinstance(vect, CountVectorizer)

def test_vectorize_count_train_shape():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["spam ham test", "ham test case"], 
        ["spam ham test"]
    )
    assert train_vec.shape[0] == 2

def test_vectorize_count_test_shape():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["spam ham test", "ham test case"], 
        ["spam ham test"]
    )
    assert test_vec.shape[0] == 1

def test_vectorize_count_feature_count():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["spam ham test", "ham test case"], 
        ["spam ham test"]
    )
    assert train_vec.shape[1] > 0
    assert test_vec.shape[1] == train_vec.shape[1]

def test_vectorize_tfidf_produces_tfidf_vectorizer():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["alpha beta", "beta gamma"],
        ["alpha beta"],
        vectorizer_type="tfidf"
    )
    assert isinstance(vect, TfidfVectorizer)

def test_vectorize_tfidf_feature_count():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["alpha beta", "beta gamma"],
        ["alpha beta"],
        vectorizer_type="tfidf"
    )
    feat_count = len(vect.get_feature_names_out())
    assert feat_count > 0

def test_vectorize_tfidf_train_feature_count():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["alpha beta", "beta gamma"],
        ["alpha beta"],
        vectorizer_type="tfidf"
    )
    feat_count = len(vect.get_feature_names_out())
    assert train_vec.shape[1] == feat_count

def test_vectorize_tfidf_test_feature_count():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ["alpha beta", "beta gamma"],
        ["alpha beta"],
        vectorizer_type="tfidf"
    )
    feat_count = len(vect.get_feature_names_out())
    assert test_vec.shape[1] == feat_count

def test_unknown_vectorizer_raises_value_error():
    with pytest.raises(ValueError):
        vz.vectorize_train_test(["foo"], ["bar"], vectorizer_type="bogus")

def test_load_and_vectorize_data_train_shape(tmp_path, monkeypatch):
    root = tmp_path
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    file = proc / "preprocessed_data.csv"
    df = pd.DataFrame({
        "review_text": ["spam spam", "ham ham", "spam cheese", "ham biscuit"],
        "label": [1, 0, 1, 0]
    })
    df.to_csv(file, index=False)
    monkeypatch.setattr(vz, "get_project_root", lambda: str(root))
    vect, Xtr, Xte, ytr, yte, data = vz.load_and_vectorize_data(vectorizer_type="count", test_size=0.5, random_state=123)
    assert Xtr.shape[0] == 2

def test_load_and_vectorize_data_test_shape(tmp_path, monkeypatch):
    root = tmp_path
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    file = proc / "preprocessed_data.csv"
    df = pd.DataFrame({
        "review_text": ["spam spam", "ham ham", "spam cheese", "ham biscuit"],
        "label": [1, 0, 1, 0]
    })
    df.to_csv(file, index=False)
    monkeypatch.setattr(vz, "get_project_root", lambda: str(root))
    vect, Xtr, Xte, ytr, yte, data = vz.load_and_vectorize_data(vectorizer_type="count", test_size=0.5, random_state=123)
    assert Xte.shape[0] == 2

def test_load_and_vectorize_data_returns_dataframe(tmp_path, monkeypatch):
    root = tmp_path
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    file = proc / "preprocessed_data.csv"
    df = pd.DataFrame({
        "review_text": ["spam spam", "ham ham", "spam cheese", "ham biscuit"],
        "label": [1, 0, 1, 0]
    })
    df.to_csv(file, index=False)
    monkeypatch.setattr(vz, "get_project_root", lambda: str(root))
    vect, Xtr, Xte, ytr, yte, data = vz.load_and_vectorize_data(vectorizer_type="count", test_size=0.5, random_state=123)
    pd.testing.assert_frame_equal(data.reset_index(drop=True), df.reset_index(drop=True))

def test_load_and_vectorize_data_train_labels(tmp_path, monkeypatch):
    root = tmp_path
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    file = proc / "preprocessed_data.csv"
    df = pd.DataFrame({
        "review_text": ["spam spam", "ham ham", "spam cheese", "ham biscuit"],
        "label": [1, 0, 1, 0]
    })
    df.to_csv(file, index=False)
    monkeypatch.setattr(vz, "get_project_root", lambda: str(root))
    vect, Xtr, Xte, ytr, yte, data = vz.load_and_vectorize_data(vectorizer_type="count", test_size=0.5, random_state=123)
    assert set(ytr.tolist()) == {0, 1}

def test_load_and_vectorize_data_test_labels(tmp_path, monkeypatch):
    root = tmp_path
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    file = proc / "preprocessed_data.csv"
    df = pd.DataFrame({
        "review_text": ["spam spam", "ham ham", "spam cheese", "ham biscuit"],
        "label": [1, 0, 1, 0]
    })
    df.to_csv(file, index=False)
    monkeypatch.setattr(vz, "get_project_root", lambda: str(root))
    vect, Xtr, Xte, ytr, yte, data = vz.load_and_vectorize_data(vectorizer_type="count", test_size=0.5, random_state=123)
    assert set(yte.tolist()) == {0, 1}

def test_load_and_vectorize_data_vectorizer_type(tmp_path, monkeypatch):
    root = tmp_path
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    file = proc / "preprocessed_data.csv"
    df = pd.DataFrame({
        "review_text": ["spam spam", "ham ham", "spam cheese", "ham biscuit"],
        "label": [1, 0, 1, 0]
    })
    df.to_csv(file, index=False)
    monkeypatch.setattr(vz, "get_project_root", lambda: str(root))
    vect, Xtr, Xte, ytr, yte, data = vz.load_and_vectorize_data(vectorizer_type="count", test_size=0.5, random_state=123)
    assert isinstance(vect, CountVectorizer)
