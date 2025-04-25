import pandas as pd
import numpy as np
from src.data_cleaning import fix_column_names, update_categories, remove_unneeded_columns

def test_fix_column_names_handles_bom():
    cols = fix_column_names(pd.DataFrame({'ï»¿report_date': ['2025-01-01'], 'x': [1]})).columns
    assert 'report_date' in cols and 'ï»¿report_date' not in cols

def test_fix_column_names_unchanged():
    assert fix_column_names(pd.DataFrame({'a': [1], 'b': [2]})).columns.tolist() == ['a', 'b']

def test_update_categories_mapped_correctly():
    subs = ["Ice Cream", "HHC", "Deos", "Tea", "Hair Care", "Unknown"]
    df = pd.DataFrame({'sub_category': subs, 'category': [np.nan] * len(subs)})
    out = update_categories(df)
    # Ice Cream → category, next four sub_category mappings, unknown remains
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
    cols = remove_unneeded_columns(df).columns
    assert not any(k in cols for k in keys[:-1])
    assert 'keep' in cols