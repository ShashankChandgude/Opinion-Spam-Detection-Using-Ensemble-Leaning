import os
import pandas as pd
import pytest
from src.data.data_cleaning import (drop_irrelevant_columns, rename_columns, drop_duplicated_rows, recode_truthful_deceptive, clean_data, pipeline,)


def test_drop_irrelevant_columns_drops_hotel_source_polarity():
    df = pd.DataFrame({
        'hotel': [1], 'source': [2], 'polarity': [3], 'keep': [4]
    })
    out = drop_irrelevant_columns(df)
    assert 'hotel' not in out.columns
    assert 'source' not in out.columns
    assert 'polarity' not in out.columns


def test_rename_columns_renames_deceptive_and_text():
    df = pd.DataFrame({'deceptive': [True], 'text': ['foo']})
    out = rename_columns(df)
    assert list(out.columns) == ['label', 'review_text']
    assert out['label'].iloc[0] == True
    assert out['review_text'].iloc[0] == 'foo'


def test_drop_duplicated_rows_logs_and_drops(caplog):
    caplog.set_level('INFO')
    df = pd.DataFrame({'a': [1, 1, 2]})
    assert drop_duplicated_rows(df).shape[0] == 2
    assert "Dropped 1 duplicate rows" in caplog.text


def test_recode_truthful_deceptive_maps_values():
    df = pd.DataFrame({'label': ['truthful', 'deceptive', 'truthful']})
    out = recode_truthful_deceptive(df.copy())
    assert out['label'].tolist() == [0, 1, 0]


def test_clean_data_combines_all_steps(monkeypatch):
    raw = pd.DataFrame({
        'hotel': [1],
        'source': ['x'],
        'polarity': [0],
        'deceptive': ['truthful'],
        'text': ['hello world']
    })
    cleaned = clean_data(raw)
    assert list(cleaned.columns) == ['label', 'review_text']
    assert cleaned['label'].iloc[0] == 0
    assert cleaned['review_text'].iloc[0] == 'hello world'


def test_pipeline_entrypoint(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr('src.data.data_cleaning.get_project_root', lambda: str(tmp_path))
    raw_dir = tmp_path / 'data' / 'raw'
    raw_dir.mkdir(parents=True)
    fake = raw_dir / 'deceptive-opinion-corpus.csv'
    fake.write_text("deceptive,hotel,source,polarity,text\ntruthful,x,x,0,hi\n")
    calls = []
    monkeypatch.setattr('src.data.data_cleaning.write_csv_file',
                        lambda df, p: calls.append(p))
    caplog.set_level('INFO')

    pipeline()

    log = caplog.text
    assert "🔹 Starting data cleaning phase" in log
    assert "Loaded raw data: 1 rows × 5 cols" in log
    # After drop_irrelevant, only 2 cols remain
    assert "Cleaned data: 1 rows × 2 cols" in log
    assert "✅ Data cleaning done, saved to" in log
    # Ensure we wrote out to the processed folder
    out_p = calls[0]
    assert out_p.endswith(os.path.join('data', 'processed', 'cleaned_data.csv'))
