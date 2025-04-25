import pandas as pd
import numpy as np
from src.preprocessing import (
    drop_unwanted_cols,
    remove_duplications,
    compute_text_stats,
    drop_text_stats_cols,
    apply_text_cleaning,
    remove_common_rare_words,
)

def test_drop_unwanted_cols_removes_extra():
    df = pd.DataFrame({
        "Unnamed: 0": [1],
        "review_title": ["t"],
        "review_date": ["d"],
        "review_text": ["x"],
    })
    cols = drop_unwanted_cols(df).columns
    assert list(cols) == ["review_text"]

def test_remove_duplications_eliminates_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    out = remove_duplications(df)
    assert out.shape[0] == 2
    assert out.reset_index(drop=True).equals(out)

def test_compute_text_stats_creates_correct_metrics():
    txt = "Hello, WORLD!!"
    df = pd.DataFrame({"review_text": [txt]})
    out = compute_text_stats(df)
    assert out.at[0, "total_words"] == 2
    assert out.at[0, "total_characters"] == len(txt)
    assert out.at[0, "total_punctuations"] == sum(ch in set("!.,?") for ch in txt)
    assert out.at[0, "total_uppercases"] == sum(ch.isupper() for ch in txt)

def test_drop_text_stats_cols_removes_stats_and_rating():
    cols = [
        "total_words", "total_characters", "total_stopwords",
        "total_punctuations", "total_uppercases", "review_rating", "review_text"
    ]
    df = pd.DataFrame({c: [0] for c in cols})
    remaining = drop_text_stats_cols(df).columns
    assert "review_text" in remaining
    assert not any(c.startswith("total_") or c == "review_rating" for c in remaining)

def test_apply_text_cleaning_filters_and_stems():
    raw = "This is a TEST! Running 123"
    cleaned = apply_text_cleaning(raw)
    assert cleaned.split() == ["test", "run"]

def test_remove_common_rare_words_filters_top_and_bottom():
    # 'c' and 'f' appear 3x (common), 'b' 2x (common), 'a','d','e' once (rare)
    sentence = " ".join(["a","b","b","c","c","c","d","e","f","f","f"])
    df = pd.DataFrame({"review_text": [sentence]})
    out = remove_common_rare_words(df)
    assert out.at[0, "review_text"] == ""