import pandas as pd
import pytest
from unittest.mock import patch, Mock
from src.data.preprocessing import DataPreprocessor, pipeline


class TestDataPreprocessor:
    def test_has_logger(self):
        preprocessor = DataPreprocessor()
        assert hasattr(preprocessor, 'logger')

    def test_has_root(self):
        preprocessor = DataPreprocessor()
        assert hasattr(preprocessor, 'root')

    def test_root_is_str(self):
        preprocessor = DataPreprocessor()
        assert isinstance(preprocessor.root, str)

    def test_has_process(self):
        preprocessor = DataPreprocessor()
        assert hasattr(preprocessor, 'process')

    def test_process_is_callable(self):
        preprocessor = DataPreprocessor()
        assert callable(preprocessor.process)

    @patch('src.data.preprocessing.load_csv_file')
    def test_load_cleaned_data_type(self, mock_load_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2', 'text3'], 'label': [0, 1, 0]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        result = preprocessor.load_cleaned_data()
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.preprocessing.load_csv_file')
    def test_load_cleaned_data_shape(self, mock_load_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2', 'text3'], 'label': [0, 1, 0]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        result = preprocessor.load_cleaned_data()
        assert result.shape == (3, 2)

    @patch('src.data.preprocessing.load_csv_file')
    def test_load_cleaned_data_columns(self, mock_load_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2', 'text3'], 'label': [0, 1, 0]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        result = preprocessor.load_cleaned_data()
        assert list(result.columns) == ['review_text', 'label']

    @patch('src.data.preprocessing.load_csv_file')
    def test_load_cleaned_data_mock_called(self, mock_load_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2', 'text3'], 'label': [0, 1, 0]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        preprocessor.load_cleaned_data()
        mock_load_csv.assert_called_once()

    @patch('src.data.preprocessing.write_csv_file')
    def test_save_preprocessed_data_mock_called(self, mock_write_csv):
        test_data = pd.DataFrame({'review_text': ['processed text1', 'processed text2'], 'label': [0, 1], 'tokens': [['processed', 'text1'], ['processed', 'text2']]})
        preprocessor = DataPreprocessor()
        preprocessor.save_preprocessed_data(test_data)
        mock_write_csv.assert_called_once()

    def test_compute_text_stats_total_words(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert 'total_words' in result.columns

    def test_compute_text_stats_total_characters(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert 'total_characters' in result.columns

    def test_compute_text_stats_total_stopwords(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert 'total_stopwords' in result.columns

    def test_compute_text_stats_total_punctuations(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert 'total_punctuations' in result.columns

    def test_compute_text_stats_total_uppercases(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert 'total_uppercases' in result.columns

    def test_compute_text_stats_word_count(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert result['total_words'].iloc[0] == 2

    def test_compute_text_stats_char_count(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert result['total_characters'].iloc[0] == 12

    def test_compute_text_stats_punct_count(self):
        test_data = pd.DataFrame({'review_text': ['Hello world!', 'This is a test.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.compute_text_stats(test_data)
        assert result['total_punctuations'].iloc[0] == 1

    def test_clean_text_column(self):
        test_data = pd.DataFrame({'review_text': ['Hello World!', 'This is a TEST.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.clean_text(test_data)
        assert 'review_text' in result.columns

    def test_clean_text_not_equal_0(self):
        test_data = pd.DataFrame({'review_text': ['Hello World!', 'This is a TEST.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.clean_text(test_data)
        assert result['review_text'].iloc[0] != 'Hello World!'

    def test_clean_text_not_equal_1(self):
        test_data = pd.DataFrame({'review_text': ['Hello World!', 'This is a TEST.'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.clean_text(test_data)
        assert result['review_text'].iloc[1] != 'This is a TEST.'

    def test_tokenize_text_column(self):
        test_data = pd.DataFrame({'review_text': ['Hello world', 'This is a test'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.tokenize_text(test_data)
        assert 'tokens' in result.columns

    def test_tokenize_text_is_list(self):
        test_data = pd.DataFrame({'review_text': ['Hello world', 'This is a test'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.tokenize_text(test_data)
        assert isinstance(result['tokens'].iloc[0], list)

    def test_tokenize_text_len(self):
        test_data = pd.DataFrame({'review_text': ['Hello world', 'This is a test'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.tokenize_text(test_data)
        assert len(result['tokens'].iloc[0]) > 0

    @patch('src.data.preprocessing.write_csv_file')
    @patch('src.data.preprocessing.load_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_no_data_type(self, mock_makedirs, mock_load_csv, mock_write_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        result = preprocessor.process()
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.preprocessing.write_csv_file')
    @patch('src.data.preprocessing.load_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_no_data_tokens(self, mock_makedirs, mock_load_csv, mock_write_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        result = preprocessor.process()
        assert 'tokens' in result.columns

    @patch('src.data.preprocessing.write_csv_file')
    @patch('src.data.preprocessing.load_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_no_data_load_called(self, mock_makedirs, mock_load_csv, mock_write_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        preprocessor.process()
        mock_load_csv.assert_called_once()

    @patch('src.data.preprocessing.write_csv_file')
    @patch('src.data.preprocessing.load_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_no_data_write_called(self, mock_makedirs, mock_load_csv, mock_write_csv):
        mock_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        mock_load_csv.return_value = mock_data
        preprocessor = DataPreprocessor()
        preprocessor.process()
        mock_write_csv.assert_called_once()

    @patch('src.data.preprocessing.write_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_data_type(self, mock_makedirs, mock_write_csv):
        test_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.process(test_data)
        assert isinstance(result, pd.DataFrame)

    @patch('src.data.preprocessing.write_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_data_tokens(self, mock_makedirs, mock_write_csv):
        test_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        result = preprocessor.process(test_data)
        assert 'tokens' in result.columns

    @patch('src.data.preprocessing.write_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_data_makedirs(self, mock_makedirs, mock_write_csv):
        test_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        preprocessor.process(test_data)
        mock_makedirs.assert_called()

    @patch('src.data.preprocessing.write_csv_file')
    @patch('pathlib.Path.mkdir')
    def test_process_with_data_write_called(self, mock_makedirs, mock_write_csv):
        test_data = pd.DataFrame({'review_text': ['text1', 'text2'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        preprocessor.process(test_data)
        mock_write_csv.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axis')
    @patch('src.data.preprocessing.WordCloud')
    def test_create_wordcloud_savefig(self, mock_wordcloud, mock_axis, mock_figure, mock_imshow, mock_close, mock_savefig):
        mock_wc_instance = Mock()
        mock_wordcloud.return_value = mock_wc_instance
        mock_wc_instance.generate.return_value = mock_wc_instance
        test_data = pd.DataFrame({'review_text': ['Hello world', 'This is a test'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        preprocessor.create_wordcloud(test_data, "test_folder", "test_wordcloud.png")
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axis')
    @patch('src.data.preprocessing.WordCloud')
    def test_create_wordcloud_close(self, mock_wordcloud, mock_axis, mock_figure, mock_imshow, mock_close, mock_savefig):
        mock_wc_instance = Mock()
        mock_wordcloud.return_value = mock_wc_instance
        mock_wc_instance.generate.return_value = mock_wc_instance
        test_data = pd.DataFrame({'review_text': ['Hello world', 'This is a test'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        preprocessor.create_wordcloud(test_data, "test_folder", "test_wordcloud.png")
        mock_close.assert_called_once()

    @patch('src.data.preprocessing.WordCloud')
    def test_create_wordcloud_handles_error(self, mock_wordcloud):
        mock_wordcloud.side_effect = Exception("Save error")
        test_data = pd.DataFrame({'review_text': ['Hello world', 'This is a test'], 'label': [0, 1]})
        preprocessor = DataPreprocessor()
        preprocessor.create_wordcloud(test_data, "test_folder", "test_wordcloud.png")

class TestPipeline:
    @patch('src.data.preprocessing.DataPreprocessor')
    def test_pipeline_class_called(self, mock_preprocessor_class):
        mock_preprocessor = Mock()
        mock_preprocessor_class.return_value = mock_preprocessor
        pipeline()
        mock_preprocessor_class.assert_called_once()

    @patch('src.data.preprocessing.DataPreprocessor')
    def test_pipeline_process_called(self, mock_preprocessor_class):
        mock_preprocessor = Mock()
        mock_preprocessor_class.return_value = mock_preprocessor
        pipeline()
        mock_preprocessor.process.assert_called_once() 