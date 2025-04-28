import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import src.utils as utils

# get_project_root

def test_get_project_root_contains_src():
    root = utils.get_project_root()
    assert os.path.isdir(root)
    assert os.path.isdir(os.path.join(root, 'src'))

# plot_verified_purchase_distribution

def test_plot_verified_purchase_distribution_saves_and_logs(tmp_path, monkeypatch, caplog):
    data = pd.DataFrame({'verified_purchase': [True, False, True]})
    out_dir = tmp_path / "plots"
    filename = "vp.png"
    saved = []
    monkeypatch.setattr(utils.plt, 'savefig', lambda path: saved.append(path))
    caplog.set_level(logging.INFO)

    utils.plot_verified_purchase_distribution(data, str(out_dir), filename)

    expected = os.path.join(str(out_dir), filename)
    assert saved and saved[0] == expected
    assert f"Saved verified purchase distribution plot: {expected}" in caplog.text

# plot_review_length_comparison

def test_plot_review_length_comparison_saves_and_logs(tmp_path, monkeypatch, caplog):
    df = pd.DataFrame({
        'verified_purchase': [True, False],
        'review_text': ['aaa', 'bb']
    })
    out_dir = tmp_path / "plots"
    filename = "rl.png"
    saved = []
    monkeypatch.setattr(utils.plt, 'savefig', lambda path: saved.append(path))
    caplog.set_level(logging.INFO)

    utils.plot_review_length_comparison(df, str(out_dir), filename)

    expected = os.path.join(str(out_dir), filename)
    assert saved and saved[0] == expected
    assert f"Saved review length comparison plot: {expected}" in caplog.text
