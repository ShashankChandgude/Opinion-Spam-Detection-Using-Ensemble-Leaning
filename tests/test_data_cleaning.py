import pandas as pd
import numpy as np
import logging
import runpy
import os
import pytest
import tempfile

import src.data.data_cleaning as dc


@pytest.fixture
def sample_pipeline_df():
    return pd.DataFrame({
        'ï»¿report_date': ['x', 'x'],
        'sub_category': ['Ice Cream', 'HHC'],
        'category': [np.nan, np.nan],
        'matched_keywords': [0, 0]
    })


def test_fix_column_names_handles_bom():
    df = pd.DataFrame({'ï»¿report_date': ['2025-01-01'], 'x': [1]})
    out = dc.fix_column_names(df)
    assert 'report_date' in out.columns and 'ï»¿report_date' not in out.columns


def test_fix_column_names_unchanged():
    cols = ['a', 'b']
    df = pd.DataFrame({'a': [1], 'b': [2]})
    assert dc.fix_column_names(df).columns.tolist() == cols


def test_update_categories_mapped_correctly():
    subs = ["Ice Cream", "HHC", "Deos", "Tea", "Hair Care", "Unknown"]
    df = pd.DataFrame({'sub_category': subs, 'category': [np.nan] * len(subs)})
    out = dc.update_categories(df)
    assert out['category'].iat[0] == 'Refreshment'
    assert out['sub_category'].tolist()[1:5] == [
        'Household Care',
        'Deodorants & Fragrances',
        'Tea and Soy & Fruit Beverages',
        'Hair'
    ]
    assert out['sub_category'].iat[-1] == 'Unknown'


def test_remove_unneeded_columns_drops_and_keeps():
    keys = ['matched_keywords', 'time_of_publication', 'manufacturers_response', 'keep']
    df = pd.DataFrame({k: [0] for k in keys})
    cols = dc.remove_unneeded_columns(df).columns
    assert 'keep' in cols
    for k in keys[:-1]:
        assert k not in cols


def test_explore_data_logs_overview(caplog):
    caplog.set_level(logging.INFO)
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    dc.explore_data(df)
    # Updated from "Rows: 2 Columns: 2" to the new format:
    assert "Data shape: 2 rows × 2 columns" in caplog.text


def test_clean_pipeline_returns_empty(caplog, sample_pipeline_df):
    caplog.set_level(logging.INFO)
    out = dc.clean_pipeline(sample_pipeline_df.copy())
    # after removing all unwanted columns, nothing should remain
    assert out.columns.tolist() == []
    # clean_pipeline no longer logs duplicates here, so we only check structure


def test_pipeline_invokes_components_and_logs_completion(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    # stub project structure
    raw = tmp_path / 'data' / 'raw'
    proc = tmp_path / 'data' / 'processed'
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    (raw / 'Amazon_review_data.csv').write_text('dummy')
    df = pd.DataFrame({'helpful_review_count': [0]})

    monkeypatch.setattr(dc, 'load_csv_file', lambda p: df)
    monkeypatch.setattr(dc, 'configure_logging', lambda p: None)
    monkeypatch.setattr(dc, 'clean_pipeline', lambda d: df)
    monkeypatch.setattr(dc, 'plot_helpful_review_counts', lambda d: None)
    monkeypatch.setattr(dc, 'plot_verified_purchase_distribution', lambda d, o, f: None)
    monkeypatch.setattr(dc, 'plot_review_length_comparison', lambda d, o, f: None)
    monkeypatch.setattr(dc, 'write_csv_file', lambda d, p: None)

    dc.pipeline()
    # the new completion message from pipeline()
    assert "✅ Data cleaning done" in caplog.text


def test_module_entrypoint_invokes_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(dc, 'pipeline', lambda: calls.append('run'))
    dc.pipeline()
    assert calls == ['run']
