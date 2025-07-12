import os
import pytest
import pandas as pd
from unittest.mock import patch, Mock
from src.data.load_data import FileDataLoader, load_data


class TestFileDataLoader:
    def test_has_root(self):
        loader = FileDataLoader()
        assert hasattr(loader, 'root')

    def test_root_is_str(self):
        loader = FileDataLoader()
        assert isinstance(loader.root, str)

    def test_root_is_dir(self):
        loader = FileDataLoader()
        assert os.path.isdir(loader.root)

    def test_has_load(self):
        loader = FileDataLoader()
        assert hasattr(loader, 'load')

    def test_load_is_callable(self):
        loader = FileDataLoader()
        assert callable(loader.load)

    @patch('src.data.load_data.load_csv_file')
    def test_load_existing_file_type(self, mock_load_csv):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        result = loader.load()
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.load_data.load_csv_file')
    def test_load_existing_file_equals(self, mock_load_csv):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        result = loader.load()
        assert result.equals(test_data)

    @patch('src.data.load_data.load_csv_file')
    def test_load_existing_file_mock_called(self, mock_load_csv):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        loader.load()
        mock_load_csv.assert_called_once()

    @patch('src.data.load_data.load_csv_file')
    def test_load_nonexistent_file_raises(self, mock_load_csv):
        mock_load_csv.side_effect = FileNotFoundError("File not found")
        loader = FileDataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load()

    @patch('src.data.load_data.load_csv_file')
    def test_load_uses_correct_file_path_mock_called(self, mock_load_csv):
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        loader.load()
        mock_load_csv.assert_called_once()

    @patch('src.data.load_data.load_csv_file')
    def test_load_uses_correct_file_path_args(self, mock_load_csv):
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        loader.load()
        call_args = mock_load_csv.call_args[0]
        assert len(call_args) > 0

    @patch('src.data.load_data.load_csv_file')
    def test_load_uses_correct_file_path_filename(self, mock_load_csv):
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        loader.load()
        file_path = mock_load_csv.call_args[0][0]
        assert 'deceptive-opinion-corpus.csv' in str(file_path)

    @patch('src.data.load_data.load_csv_file')
    def test_load_returns_dataframe_type(self, mock_load_csv):
        test_data = pd.DataFrame({'deceptive': ['truthful', 'deceptive', 'truthful'], 'text': ['review 1', 'review 2', 'review 3']})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        result = loader.load()
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.load_data.load_csv_file')
    def test_load_returns_dataframe_shape(self, mock_load_csv):
        test_data = pd.DataFrame({'deceptive': ['truthful', 'deceptive', 'truthful'], 'text': ['review 1', 'review 2', 'review 3']})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        result = loader.load()
        assert result.shape == (3, 2)

    @patch('src.data.load_data.load_csv_file')
    def test_load_returns_dataframe_columns(self, mock_load_csv):
        test_data = pd.DataFrame({'deceptive': ['truthful', 'deceptive', 'truthful'], 'text': ['review 1', 'review 2', 'review 3']})
        mock_load_csv.return_value = test_data
        loader = FileDataLoader()
        result = loader.load()
        assert list(result.columns) == ['deceptive', 'text']


class TestLegacyLoadData:
    def setup_method(self):
        load_data.cache_clear()

    @patch('src.data.load_data.FileDataLoader')
    def test_legacy_load_data_function_type(self, mock_loader_class):
        mock_loader = Mock()
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_loader.load.return_value = test_data
        mock_loader_class.return_value = mock_loader
        result = load_data("test_root")
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.load_data.FileDataLoader')
    def test_legacy_load_data_function_equals(self, mock_loader_class):
        mock_loader = Mock()
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_loader.load.return_value = test_data
        mock_loader_class.return_value = mock_loader
        result = load_data("test_root")
        assert result.equals(test_data)

    @patch('src.data.load_data.FileDataLoader')
    def test_legacy_load_data_function_mock_called(self, mock_loader_class):
        mock_loader = Mock()
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_loader.load.return_value = test_data
        mock_loader_class.return_value = mock_loader
        load_data("test_root")
        mock_loader.load.assert_called_once()

    @patch('src.data.load_data.FileDataLoader')
    def test_legacy_load_data_nonexistent_file_raises(self, mock_loader_class):
        mock_loader = Mock()
        mock_loader.load.side_effect = FileNotFoundError("File not found")
        mock_loader_class.return_value = mock_loader
        with pytest.raises(FileNotFoundError):
            load_data("test_root")

    @patch('src.data.load_data.FileDataLoader')
    def test_legacy_load_data_creates_loader_type(self, mock_loader_class):
        mock_loader = Mock()
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_loader.load.return_value = test_data
        mock_loader_class.return_value = mock_loader
        result = load_data("test_root")
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.load_data.FileDataLoader')
    def test_legacy_load_data_creates_loader_mock_called(self, mock_loader_class):
        mock_loader = Mock()
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_loader.load.return_value = test_data
        mock_loader_class.return_value = mock_loader
        load_data("test_root")
        mock_loader_class.assert_called_once()

    @patch('src.data.load_data.FileDataLoader')
    def test_legacy_load_data_creates_loader_load_called(self, mock_loader_class):
        mock_loader = Mock()
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        mock_loader.load.return_value = test_data
        mock_loader_class.return_value = mock_loader
        load_data("test_root")
        mock_loader.load.assert_called_once() 