import pytest
import pandas as pd
from unittest.mock import patch, Mock
from src.data.data_cleaning import DataCleaner, pipeline

class TestDataCleaner:
    def test_has_logger(self):
        cleaner = DataCleaner()
        assert hasattr(cleaner, 'logger')

    def test_has_root(self):
        cleaner = DataCleaner()
        assert hasattr(cleaner, 'root')

    def test_has_data_loader(self):
        cleaner = DataCleaner()
        assert hasattr(cleaner, 'data_loader')

    def test_has_data_saver(self):
        cleaner = DataCleaner()
        assert hasattr(cleaner, 'data_saver')

    def test_drop_irrelevant_columns_removes_hotel(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({'hotel': [1], 'source': ['a'], 'polarity': [0], 'keep': [123]})
        result = cleaner.drop_irrelevant_columns(df)
        assert 'hotel' not in result.columns

    def test_drop_irrelevant_columns_removes_source(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({'hotel': [1], 'source': ['a'], 'polarity': [0], 'keep': [123]})
        result = cleaner.drop_irrelevant_columns(df)
        assert 'source' not in result.columns

    def test_drop_irrelevant_columns_removes_polarity(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({'hotel': [1], 'source': ['a'], 'polarity': [0], 'keep': [123]})
        result = cleaner.drop_irrelevant_columns(df)
        assert 'polarity' not in result.columns

    def test_drop_irrelevant_columns_keeps_keep(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({'hotel': [1], 'source': ['a'], 'polarity': [0], 'keep': [123]})
        result = cleaner.drop_irrelevant_columns(df)
        assert 'keep' in result.columns

    def test_rename_columns_label(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({'deceptive': ['truthful'], 'text': ['review']})
        result = cleaner.rename_columns(df)
        assert 'label' in result.columns

    def test_rename_columns_review_text(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({'deceptive': ['truthful'], 'text': ['review']})
        result = cleaner.rename_columns(df)
        assert 'review_text' in result.columns

    def test_drop_duplicated_rows_count(self, caplog):
        caplog.set_level('INFO')
        cleaner = DataCleaner()
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [2, 2, 3]})
        result = cleaner.drop_duplicated_rows(df)
        assert len(result) == 2

    def test_drop_duplicated_rows_log(self, caplog):
        caplog.set_level('INFO')
        cleaner = DataCleaner()
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [2, 2, 3]})
        cleaner.drop_duplicated_rows(df)
        assert "Dropped 1 duplicate rows" in caplog.text

    def test_recode_truthful_deceptive_maps_values(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({'label': ['truthful', 'deceptive', 'truthful']})
        result = cleaner.recode_truthful_deceptive(df)
        assert set(result['label']) == {0, 1}

    @patch('src.data.data_cleaning.FileDataSaver')
    def test_process_with_data_parameter_type(self, mock_saver):
        cleaner = DataCleaner()
        raw = pd.DataFrame({'hotel': [1], 'source': ['x'], 'polarity': [0], 'deceptive': ['truthful'], 'text': ['hello world']})
        mock_saver_instance = Mock()
        cleaner.data_saver = mock_saver_instance
        cleaned = cleaner.process(raw)
        assert isinstance(cleaned, pd.DataFrame)

    @patch('src.data.data_cleaning.FileDataSaver')
    def test_process_with_data_parameter_save(self, mock_saver):
        cleaner = DataCleaner()
        raw = pd.DataFrame({'hotel': [1], 'source': ['x'], 'polarity': [0], 'deceptive': ['truthful'], 'text': ['hello world']})
        mock_saver_instance = Mock()
        cleaner.data_saver = mock_saver_instance
        cleaner.process(raw)
        mock_saver_instance.save.assert_called_once()

    @patch('src.data.data_cleaning.FileDataLoader')
    @patch('src.data.data_cleaning.FileDataSaver')
    def test_process_loads_data_type(self, mock_saver, mock_loader):
        cleaner = DataCleaner()
        mock_loader_instance = Mock()
        mock_saver_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_saver.return_value = mock_saver_instance
        mock_loader_instance.load.return_value = pd.DataFrame({'deceptive': ['truthful'], 'text': ['hello world']})
        cleaner.data_loader = mock_loader_instance
        cleaner.data_saver = mock_saver_instance
        cleaned = cleaner.process()
        assert isinstance(cleaned, pd.DataFrame)

    @patch('src.data.data_cleaning.FileDataLoader')
    @patch('src.data.data_cleaning.FileDataSaver')
    def test_process_loads_data_loader_called(self, mock_saver, mock_loader):
        cleaner = DataCleaner()
        mock_loader_instance = Mock()
        mock_saver_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_saver.return_value = mock_saver_instance
        mock_loader_instance.load.return_value = pd.DataFrame({'deceptive': ['truthful'], 'text': ['hello world']})
        cleaner.data_loader = mock_loader_instance
        cleaner.data_saver = mock_saver_instance
        cleaner.process()
        mock_loader_instance.load.assert_called_once()

    @patch('src.data.data_cleaning.FileDataLoader')
    @patch('src.data.data_cleaning.FileDataSaver')
    def test_process_loads_data_saver_called(self, mock_saver, mock_loader):
        cleaner = DataCleaner()
        mock_loader_instance = Mock()
        mock_saver_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_saver.return_value = mock_saver_instance
        mock_loader_instance.load.return_value = pd.DataFrame({'deceptive': ['truthful'], 'text': ['hello world']})
        cleaner.data_loader = mock_loader_instance
        cleaner.data_saver = mock_saver_instance
        cleaner.process()
        mock_saver_instance.save.assert_called_once()

    @patch('src.data.data_cleaning.write_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_file_data_saver_write_csv(self, mock_makedirs, mock_write_csv):
        from src.data.data_cleaning import FileDataSaver
        saver = FileDataSaver()
        df = pd.DataFrame({'a': [1, 2, 3]})
        saver.save(df, 'dummy_path.csv')
        mock_write_csv.assert_called_once_with(df, 'dummy_path.csv')


def test_legacy_pipeline_function(monkeypatch, tmp_path):
    import src.data.data_cleaning as dc
    import pandas as pd
    monkeypatch.setattr(dc, "get_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(dc, "get_logger", lambda name: type("L", (), {"info": lambda self, *a, **k: None})())
    cleaner = dc.DataCleaner()
    dummy_df = pd.DataFrame({
        "hotel": [1],
        "source": ["a"],
        "polarity": [0],
        "deceptive": ["truthful"],
        "text": ["sample review"]
    })
    cleaner.data_loader = type("DL", (), {"load": lambda s: dummy_df})()
    cleaner.data_saver = type("DS", (), {"save": lambda s, d, p: None})()
    cleaner.process()



