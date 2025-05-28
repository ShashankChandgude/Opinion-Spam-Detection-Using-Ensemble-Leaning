import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import logging

from src.utils.helpers import (
    get_project_root,
    plot_verified_purchase_distribution,
    plot_review_length_comparison
)

def test_get_project_root_is_directory():
    root = get_project_root()
    assert os.path.isdir(root)

def test_plot_verified_purchase_distribution_saves_and_logs(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    data = pd.DataFrame({'verified_purchase': [True, False, True]})
    out_dir = str(tmp_path)
    filename = "vp.png"

    plot_verified_purchase_distribution(data, out_dir, filename)

    # file was created
    target = tmp_path / filename
    assert target.exists()

    # and we logged it
    assert "Saved verified purchase distribution plot" in caplog.text

def test_plot_review_length_comparison_saves_and_logs(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    df = pd.DataFrame({
        'verified_purchase': [True, False, True],
        'review_text': ['a', 'bb', 'ccc']
    })
    out_dir = str(tmp_path)
    filename = "rl.png"

    plot_review_length_comparison(df, out_dir, filename)

    target = tmp_path / filename
    assert target.exists()
    assert "Saved review length comparison plot" in caplog.text
