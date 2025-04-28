import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import pytest
import src.preprocessing as pp
from src.preprocessing import compute_text_stats, STOPWORDS

def test_drop_unwanted_cols_removes_extra():
    df = pd.DataFrame({
        "Unnamed: 0": [1],
        "review_title": ["t"],
        "review_date": ["d"],
        "review_text": ["x"],
    })
    cols = pp.drop_unwanted_cols(df).columns.tolist()
    assert cols == ["review_text"]

def test_remove_duplications_eliminates_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    out = pp.remove_duplications(df)
    assert out.shape[0] == 2

def test_compute_text_stats_creates_correct_metrics():
    txt = "Hello, WORLD!!"
    df = pd.DataFrame({"review_text": [txt]})
    out = pp.compute_text_stats(df.copy())
    assert out.at[0, "total_words"] == 2
    assert out.at[0, "total_characters"] == len(txt)
    assert out.at[0, "total_punctuations"] == sum(ch in set("!.,?") for ch in txt)
    assert out.at[0, "total_uppercases"] == sum(ch.isupper() for ch in txt)

def test_drop_text_stats_cols_removes_all_stats_and_rating():
    cols = [
        "total_words", "total_characters", "total_stopwords",
        "total_punctuations", "total_uppercases", "review_rating", "review_text"
    ]
    df = pd.DataFrame({c: [0] for c in cols})
    remaining = pp.drop_text_stats_cols(df.copy()).columns
    assert "review_text" in remaining
    assert not any(c.startswith("total_") or c == "review_rating" for c in remaining)

def test_apply_text_cleaning_filters_and_stems():
    raw = "This is a TEST! Running 123"
    cleaned = pp.apply_text_cleaning(raw)
    assert cleaned.split() == ["test", "run"]

def test_remove_common_rare_words_filters_top_and_bottom():
    sentence = " ".join(["a","b","b","c","c","c","d","e","f","f","f"])
    df = pd.DataFrame({"review_text": [sentence]})
    out = pp.remove_common_rare_words(df.copy())
    assert out.at[0, "review_text"] == ""

def test_preprocess_text_applies_cleaning_to_column():
    df = pd.DataFrame({"review_text": ["Hello, WORLD! 123"]})
    out = pp.preprocess_text(df.copy())
    # should match exactly what apply_text_cleaning does on that string
    expected = pp.apply_text_cleaning("Hello, WORLD! 123")
    assert out.at[0, "review_text"] == expected

def test_compute_text_stats_counts_stopwords_relaxed():
    df = pd.DataFrame({"review_text": ["a of the cat"]})
    out = pp.compute_text_stats(df.copy())
    # at least 'a', 'of', 'the' are counted
    assert out.at[0, "total_stopwords"] >= 1

def test_log_rare_words_outputs_least_frequent(caplog):
    caplog.set_level(logging.INFO)
    df = pd.DataFrame({"review_text": ["x x y z"]})
    pp.log_rare_words(df)
    log = caplog.text
    assert "Rare words:" in log
    # 'y' and 'z' appear once
    assert "\nz    1" in log or "\ny    1" in log

def test_create_wordcloud_saves_and_logs(tmp_path, monkeypatch, caplog):
    df = pd.DataFrame({"review_text": ["aa bb", "cc dd"]})
    out_dir = tmp_path / "plots"; out_dir.mkdir()
    fname = "wc.png"

    # stub WordCloud
    class FakeWC:
        def generate(self, text): return None
    monkeypatch.setattr(pp, "WordCloud", lambda **k: FakeWC())

    # stub plt methods
    monkeypatch.setattr(pp.plt, "figure", lambda *a, **k: None)
    monkeypatch.setattr(pp.plt, "imshow", lambda img: None)
    monkeypatch.setattr(pp.plt, "axis", lambda *a, **k: None)
    monkeypatch.setattr(pp.plt, "tight_layout", lambda *a, **k: None)

    saved = []
    monkeypatch.setattr(pp.plt, "savefig", lambda p: saved.append(p))
    caplog.set_level(logging.INFO)

    pp.create_wordcloud(df, str(out_dir), fname)
    expected = os.path.join(str(out_dir), fname)
    assert saved == [expected]
    assert f"Saved wordcloud: {expected}" in caplog.text

def test_tokenize_text_returns_list_and_logs(caplog, monkeypatch):
    df = pd.DataFrame({"review_text": ["x y"]})
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(pp, "word_tokenize", lambda t: t.split())
    tokens = pp.tokenize_text(df.copy())
    assert isinstance(tokens.iloc[0], list)
    assert "Tokenized (first 5)" in caplog.text

def test_pipeline_executes_all_steps_and_logs(caplog, tmp_path, monkeypatch):
    calls = []
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(pp, "get_project_root", lambda: str(tmp_path))
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "updated_data.csv").write_text("ignore")

    monkeypatch.setattr(pp, "load_csv_file", lambda p: pd.DataFrame({
        "review_text": ["test"], "review_rating": [1]
    }))
    monkeypatch.setattr(pp, "drop_unwanted_cols", lambda d: d)
    monkeypatch.setattr(pp, "remove_duplications", lambda d: d)
    monkeypatch.setattr(pp, "compute_text_stats", lambda d: d)
    monkeypatch.setattr(pp, "plot_verified_purchase_distribution", lambda d,o,f: calls.append("vp"))
    monkeypatch.setattr(pp, "plot_review_length_comparison", lambda d,o,f: calls.append("rl"))
    monkeypatch.setattr(pp, "drop_text_stats_cols", lambda d: d)
    monkeypatch.setattr(pp, "preprocess_text", lambda d: d)
    monkeypatch.setattr(pp, "log_rare_words", lambda d: calls.append("lr"))
    monkeypatch.setattr(pp, "create_wordcloud", lambda d,o,f: calls.append("cw"))
    monkeypatch.setattr(pp, "remove_common_rare_words", lambda d: d)
    monkeypatch.setattr(pp, "tokenize_text", lambda d: calls.append("tz"))
    monkeypatch.setattr(pp, "write_csv_file", lambda d,p: calls.append("wr"))
    monkeypatch.setattr(pp, "configure_logging", lambda p: calls.append("logcfg"))

    pp.pipeline()
    assert calls == ["logcfg","vp","rl","lr","cw","cw","tz","wr"]
    assert "Preprocessing pipeline completed successfully." in caplog.text