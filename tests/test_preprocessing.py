import pandas as pd
import numpy as np
import logging
import os
import pytest

import src.data.preprocessing as pp


def test_drop_unwanted_cols_removes_extra():
    df = pd.DataFrame({
        "Unnamed: 0": [1],
        "review_title": ["t"],
        "review_date": ["d"],
        "review_text": ["x"],
    })
    assert pp.drop_unwanted_cols(df).columns.tolist() == ["review_text"]


def test_remove_duplications_eliminates_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    out = pp.remove_duplications(df)
    assert out.shape[0] == 2
    assert list(out.index) == [0, 1]


def test_compute_text_stats_creates_correct_metrics():
    txt = "Hello, WORLD!!"
    df = pd.DataFrame({"review_text": [txt]})
    out = pp.compute_text_stats(df.copy())
    assert out.at[0, "total_words"] == 2
    assert out.at[0, "total_characters"] == len(txt)
    # punctuation count just checks that the column exists
    assert "total_punctuations" in out.columns
    # uppercase count
    assert out.at[0, "total_uppercases"] == sum(ch.isupper() for ch in txt)


def test_drop_text_stats_cols_removes_stats_and_rating():
    cols = [
        "total_words", "total_characters", "total_stopwords",
        "total_punctuations", "total_uppercases", "review_rating", "review_text"
    ]
    df = pd.DataFrame({c: [0] for c in cols})
    remaining = pp.drop_text_stats_cols(df).columns
    assert "review_text" in remaining
    for c in remaining:
        assert not c.startswith("total_") or c == "review_text"


def test_apply_text_cleaning_filters_and_stems():
    raw = "This is a TEST! Running 123"
    cleaned = pp.apply_text_cleaning(raw)
    # should remove non-alpha, lowercase, remove stopwords, stem
    assert set(cleaned.split()) == {"test", "run"}


def test_remove_common_rare_words_filters_top_and_bottom():
    sentence = "a b b c c c d e f f f"
    df = pd.DataFrame({"review_text": [sentence]})
    out = pp.remove_common_rare_words(df.copy())
    # all words were either too common or too rare
    assert out.at[0, "review_text"] == ""


def test_log_rare_words_outputs_least_frequent(caplog):
    caplog.set_level(logging.INFO)
    df = pd.DataFrame({"review_text": ["x x y z"]})
    pp.log_rare_words(df)
    log = caplog.text
    # only check for the new phrase
    assert "least common tokens:" in log


def test_tokenize_text_returns_list_and_logs(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    df = pd.DataFrame({"review_text": ["x y"]})
    monkeypatch.setattr(pp, "word_tokenize", lambda t: t.split())
    tokens = pp.tokenize_text(df.copy())
    assert isinstance(tokens.iloc[0], list)
    # updated logging header
    assert "Tokenized (first 5 rows)" in caplog.text


def test_pipeline_executes_all_steps_and_logs_completion(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    # stub project structure and I/O
    monkeypatch.setattr(pp, "get_project_root", lambda: str(tmp_path))
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "updated_data.csv").write_text("ignore")

    monkeypatch.setattr(pp, "load_csv_file", lambda p: pd.DataFrame({
        "review_text": ["test"], "review_rating": [1]
    }))
    monkeypatch.setattr(pp, "drop_unwanted_cols", lambda d: d)
    monkeypatch.setattr(pp, "remove_duplications", lambda d: d)
    monkeypatch.setattr(pp, "compute_text_stats", lambda d: d)
    monkeypatch.setattr(pp, "plot_verified_purchase_distribution", lambda d, o, f: None)
    monkeypatch.setattr(pp, "plot_review_length_comparison", lambda d, o, f: None)
    monkeypatch.setattr(pp, "drop_text_stats_cols", lambda d: d)
    monkeypatch.setattr(pp, "preprocess_text", lambda d: d)
    monkeypatch.setattr(pp, "log_rare_words", lambda d: None)
    monkeypatch.setattr(pp, "create_wordcloud", lambda d, o, f: None)
    monkeypatch.setattr(pp, "remove_common_rare_words", lambda d: d)
    monkeypatch.setattr(pp, "tokenize_text", lambda d: None)
    monkeypatch.setattr(pp, "write_csv_file", lambda d, p: None)
    monkeypatch.setattr(pp, "configure_logging", lambda p: None)

    pp.pipeline()
    # updated completion marker
    assert "✅ Preprocessing done" in caplog.text
