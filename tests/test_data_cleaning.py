import os
import pandas as pd
import numpy as np
import logging
import pytest
import runpy
import src.data_cleaning as dc

def test_explore_data_logs_overview(caplog):
    df = pd.DataFrame({'a': [1,2], 'b': ['x','y']})
    caplog.set_level(logging.INFO)
    dc.explore_data(df)
    log = caplog.text
    assert "Rows: 2 Columns: 2" in log
    assert "Head:" in log and "Numerical stats:" in log
    assert "Categorical stats:" in log

def test_fix_column_names_only_replaces_bom():
    df = pd.DataFrame({'ï»¿report_date': ['d'], 'x': [1]})
    out = dc.fix_column_names(df)
    assert 'report_date' in out.columns and 'ï»¿report_date' not in out.columns

def test_update_categories_applies_mapping_and_preserves_others():
    df = pd.DataFrame({
        'sub_category': ["Ice Cream", "Unknown"],
        'category': [np.nan, np.nan]
    })
    out = dc.update_categories(df.copy())
    assert out.loc[0, 'category'] == 'Refreshment'
    assert pd.isna(out.loc[1, 'category'])
    assert out.loc[1, 'sub_category'] == 'Unknown'

def test_remove_unneeded_columns_drops_and_keeps_correctly():
    df = pd.DataFrame({'matched_keywords': [1], 'report_date': [2], 'keep': [3]})
    out = dc.remove_unneeded_columns(df)
    assert 'matched_keywords' not in out.columns
    assert 'report_date' not in out.columns
    assert 'keep' in out.columns

def test_plot_helpful_review_counts_saves_and_logs(tmp_path, monkeypatch, caplog):
    df = pd.DataFrame({'helpful_review_count': [0,1]})
    monkeypatch.setattr(dc, 'get_project_root', lambda: str(tmp_path))
    monkeypatch.setattr(dc.plt, 'figure', lambda *a, **k: None)
    class Dummy:
        def set_title(self, _): pass
    monkeypatch.setattr(dc.sns, 'countplot', lambda **k: Dummy())
    saved = []
    monkeypatch.setattr(dc.plt, 'savefig', lambda p: saved.append(p))
    caplog.set_level(logging.INFO)

    dc.plot_helpful_review_counts(df)
    expected = os.path.join(str(tmp_path), 'output', 'data_cleaning',
                            'helpful_review_counts.png')
    assert saved == [expected]
    assert f"Saved plot: {expected}" in caplog.text

def test_clean_pipeline_returns_empty_and_logs_duplicates(caplog):
    df = pd.DataFrame({
        'ï»¿report_date': ['x','x'],
        'sub_category': ['Ice Cream','HHC'],
        'category': [np.nan, np.nan],
        'matched_keywords': [0,0]
    })
    caplog.set_level(logging.INFO)
    out = dc.clean_pipeline(df.copy())
    assert out.columns.tolist() == []
    # After update_categories the two rows differ, so duplicates = 0
    assert "Duplicates: 0" in caplog.text

def test_pipeline_invokes_components_and_logs_completion(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    # stub project structure and data
    monkeypatch.setattr(dc, 'get_project_root', lambda: str(tmp_path))
    raw = tmp_path / 'data' / 'raw'; proc = tmp_path / 'data' / 'processed'
    raw.mkdir(parents=True); proc.mkdir(parents=True)
    (raw / 'Amazon_review_data.csv').write_text('dummy')
    df = pd.DataFrame({'helpful_review_count': [0]})
    monkeypatch.setattr(dc, 'load_csv_file', lambda p: df)
    monkeypatch.setattr(dc, 'configure_logging', lambda p: None)
    monkeypatch.setattr(dc, 'clean_pipeline', lambda d: df)
    monkeypatch.setattr(dc, 'plot_helpful_review_counts', lambda d: None)
    monkeypatch.setattr(dc, 'plot_verified_purchase_distribution', lambda d,o,f: None)
    monkeypatch.setattr(dc, 'plot_review_length_comparison', lambda d,o,f: None)
    monkeypatch.setattr(dc, 'write_csv_file', lambda d,p: None)

    dc.pipeline()
    assert "Data cleaning process completed successfully." in caplog.text
