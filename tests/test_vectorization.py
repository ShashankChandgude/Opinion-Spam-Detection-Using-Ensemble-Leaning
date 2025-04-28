import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import src.vectorization as vz


def test_count_vectorizer_splits():
    vect, train_vec, test_vec = vz.vectorize_train_test(['aa bb', 'bb cc'], ['aa bb'])
    assert isinstance(vect, CountVectorizer)
    assert train_vec.shape[0] == 2 and test_vec.shape[0] == 1


def test_tfidf_vectorizer_dimensions():
    vect, train_vec, test_vec = vz.vectorize_train_test(
        ['xx yy', 'yy zz'], ['xx yy'], vectorizer_type='tfidf'
    )
    features = vect.get_feature_names_out()
    assert isinstance(vect, TfidfVectorizer)
    assert train_vec.shape[1] == len(features) and test_vec.shape[1] == len(features)


def test_invalid_vectorizer_raises():
    with pytest.raises(ValueError):
        vz.vectorize_train_test(['aa'], ['bb'], vectorizer_type='bad')


def test_load_and_vectorize_splits(tmp_path, monkeypatch):
    root = tmp_path
    proc = root / 'data' / 'processed'
    proc.mkdir(parents=True)
    df = pd.DataFrame({
        'review_text': ['foo foo', 'bar bar', 'baz baz', 'qux qux'],
        'verified_purchase': [True, False, True, False]
    })
    file = proc / 'cleaned_data.csv'
    df.to_csv(file, index=False)
    monkeypatch.setattr(vz, 'get_project_root', lambda: str(root))

    vect, Xtr, Xte, ytr, yte, data = vz.load_and_vectorize_data(
        test_size=0.5, vectorizer_type='count', random_state=1
    )
    assert Xtr.shape[0] == 2 and Xte.shape[0] == 2 and data.equals(df)