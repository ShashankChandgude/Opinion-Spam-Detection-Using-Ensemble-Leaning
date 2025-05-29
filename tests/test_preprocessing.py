import os
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from src.data import preprocessing as pp

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "review_text": [
            "Hello, WORLD!! This is a Test.",
            "Short text..."
        ]
    })

def test_compute_text_stats_adds_expected_columns(simple_df):
    df = pp.compute_text_stats(simple_df)
    assert "review_text" in df.columns
    for col in ["total_words", "total_characters", "total_stopwords",
                "total_punctuations", "total_uppercases"]:
        assert col in df.columns
    assert df.at[0, "total_words"] == 6
    assert df.at[0, "total_characters"] == len(simple_df.at[0, "review_text"])
    assert df.at[0, "total_punctuations"] >= 3
    assert df.at[0, "total_uppercases"] >= 7

def test_clean_df_filters_non_alpha_and_stems():
    df = pd.DataFrame({
        "review_text": ["Running!! Running requires energized runners..."]
    })
    cleaned = pp.clean_df(df)
    txt = cleaned.at[0, "review_text"]
    assert "run" in txt
    assert all(ch.isalpha() or ch.isspace() for ch in txt)

def test_log_token_stats_emits_both_logs(caplog):
    caplog.set_level("INFO")
    df = pd.DataFrame({"review_text": ["a a b c", "b c c d"]})
    pp.log_token_stats(df)
    text = caplog.text
    assert "10 most common tokens:" in text
    assert "10 least common tokens:" in text

def test_tokenize_df_creates_tokens_column_and_logs(caplog, monkeypatch):
    caplog.set_level("INFO")
    monkeypatch.setattr(pp, "word_tokenize", lambda txt: txt.split())
    df = pd.DataFrame({"review_text": ["one two", "three"]})
    out = pp.tokenize_df(df)
    assert "tokens" in out.columns
    assert out.at[0, "tokens"] == ["one", "two"]
    assert "Tokenized (first 5 rows):" in caplog.text

def test_load_and_save_preprocessed_data(tmp_path, caplog):
    root = tmp_path
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "cleaned_data.csv").write_text("review_text\nalpha\nbeta\n")
    caplog.set_level("INFO")
    df = pp.load_cleaned_data(str(root))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 1)
    assert "Loaded cleaned data: 2 rows × 1 cols" in caplog.text
    calls = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv 
    monkeypatch.setattr(pp, "write_csv_file", lambda df_, path: calls.append(path))
    pp.save_preprocessed_data(df, str(root))
    monkeypatch.undo()
    assert calls, "Expected write_csv_file to be called"
    assert calls[0].endswith(os.path.join("data","processed","preprocessed_data.csv"))
    assert "Saved preprocessed data to" in caplog.text

def test_pipeline_end_to_end(tmp_path, caplog, monkeypatch):
    caplog.set_level("INFO")
    monkeypatch.setattr(pp, "setup_logging", lambda root: None)

    monkeypatch.setattr(pp, "get_project_root", lambda: str(tmp_path))

    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "cleaned_data.csv").write_text("review_text\nspam spam ham\nfoo bar baz\n")

    monkeypatch.setattr(pp, "create_wordcloud", lambda df, of, fn: None)

    pp.pipeline()

    log = caplog.text
    assert "🔹 Starting preprocessing" in log
    assert "Loaded cleaned data: 2 rows × 1 cols" in log
    assert "✅ Preprocessing done" in log

    out_csv = tmp_path / "data" / "processed" / "preprocessed_data.csv"
    assert out_csv.exists()
    df_out = pd.read_csv(out_csv)

    assert "review_text" in df_out.columns
    assert "total_words" in df_out.columns
    assert "tokens" in df_out.columns

