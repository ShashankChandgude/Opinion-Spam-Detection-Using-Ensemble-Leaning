import os
import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from unittest.mock import patch, Mock
from src.utils.helpers import (
    get_project_root,
    plot_verified_purchase_distribution,
    plot_review_length_comparison
)


class TestHelpers:
    
    def test_get_project_root(self):
        root = get_project_root()
        
        assert isinstance(root, str)
        
        assert os.path.isabs(root)
        
        assert os.path.isdir(root)
        
        assert os.path.exists(os.path.join(root, 'src'))
        assert os.path.exists(os.path.join(root, 'tests'))
    
    def test_get_project_root_returns_consistent_path(self):
        root1 = get_project_root()
        root2 = get_project_root()
        
        assert root1 == root2
    
    def test_plot_verified_purchase_distribution(self, tmp_path, caplog):
        caplog.set_level('INFO')
        
        data = pd.DataFrame({
            'verified_purchase': [True, False, True, True, False]
        })
        
        out_folder = str(tmp_path)
        filename = "vp_distribution.png"
        
        plot_verified_purchase_distribution(data, out_folder, filename)
        
        filepath = os.path.join(out_folder, filename)
        assert os.path.exists(filepath)
        
        assert "Saved verified purchase distribution plot" in caplog.text
        assert filepath in caplog.text
    
    def test_plot_verified_purchase_distribution_with_empty_data(self, tmp_path, caplog):
        caplog.set_level('INFO')
        
        data = pd.DataFrame(columns=['verified_purchase'])
        
        out_folder = str(tmp_path)
        filename = "empty_vp.png"
        
        plot_verified_purchase_distribution(data, out_folder, filename)
        
        filepath = os.path.join(out_folder, filename)
        assert os.path.exists(filepath)
    
    def test_plot_verified_purchase_distribution_with_single_value(self, tmp_path, caplog):
        caplog.set_level('INFO')
        
        data = pd.DataFrame({
            'verified_purchase': [True, True, True]
        })
        
        out_folder = str(tmp_path)
        filename = "single_vp.png"
        
        plot_verified_purchase_distribution(data, out_folder, filename)
        
        filepath = os.path.join(out_folder, filename)
        assert os.path.exists(filepath)
    
    def test_plot_review_length_comparison(self, tmp_path, caplog):
        caplog.set_level('INFO')
        
        data = pd.DataFrame({
            'verified_purchase': [True, False, True, False, True],
            'review_text': ['short', 'longer text here', 'medium', 'very long text for testing', 'tiny']
        })
        
        out_folder = str(tmp_path)
        filename = "review_length.png"
        
        plot_review_length_comparison(data, out_folder, filename)
        
        filepath = os.path.join(out_folder, filename)
        assert os.path.exists(filepath)
        
        assert "Saved review length comparison plot" in caplog.text
        assert filepath in caplog.text
    
    def test_plot_review_length_comparison_with_empty_data(self, tmp_path, caplog):
        caplog.set_level('INFO')
        
        data = pd.DataFrame(columns=['verified_purchase', 'review_text'])
        
        out_folder = str(tmp_path)
        filename = "empty_length.png"
        
        plot_review_length_comparison(data, out_folder, filename)
        
        filepath = os.path.join(out_folder, filename)
        assert os.path.exists(filepath)
    
    def test_plot_review_length_comparison_with_missing_verified_purchase(self, tmp_path, caplog):
        caplog.set_level('INFO')
        
        data = pd.DataFrame({
            'verified_purchase': [True, True, True],
            'review_text': ['short', 'medium', 'long']
        })
        
        out_folder = str(tmp_path)
        filename = "missing_false.png"
        
        plot_review_length_comparison(data, out_folder, filename)
        
        filepath = os.path.join(out_folder, filename)
        assert os.path.exists(filepath)
    
    def test_plot_review_length_comparison_with_missing_true(self, tmp_path, caplog):
        caplog.set_level('INFO')
        
        data = pd.DataFrame({
            'verified_purchase': [False, False, False],
            'review_text': ['short', 'medium', 'long']
        })
        
        out_folder = str(tmp_path)
        filename = "missing_true.png"
        
        plot_review_length_comparison(data, out_folder, filename)
        
        filepath = os.path.join(out_folder, filename)
        assert os.path.exists(filepath)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_verified_purchase_distribution_handles_save_error(self, mock_savefig, tmp_path, caplog):
        caplog.set_level('INFO')
        
        mock_savefig.side_effect = Exception("Save error")
        
        data = pd.DataFrame({
            'verified_purchase': [True, False, True]
        })
        
        out_folder = str(tmp_path)
        filename = "error_test.png"
        
        with pytest.raises(Exception, match="Save error"):
            plot_verified_purchase_distribution(data, out_folder, filename)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_review_length_comparison_handles_save_error(self, mock_savefig, tmp_path, caplog):
        caplog.set_level('INFO')
        
        mock_savefig.side_effect = Exception("Save error")
        
        data = pd.DataFrame({
            'verified_purchase': [True, False, True],
            'review_text': ['short', 'long', 'medium']
        })
        
        out_folder = str(tmp_path)
        filename = "error_test.png"
        
        with pytest.raises(Exception, match="Save error"):
            plot_review_length_comparison(data, out_folder, filename) 