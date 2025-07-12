import pandas as pd
import pytest
import tempfile
import os
from src.data.data_io import load_csv_file, write_csv_file

class TestDataIO:
    def test_load_csv_file_type(self, tmp_path):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path, index=False)
        loaded_data = load_csv_file(str(csv_path))
        assert isinstance(loaded_data, pd.DataFrame)

    def test_load_csv_file_shape(self, tmp_path):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path, index=False)
        loaded_data = load_csv_file(str(csv_path))
        assert loaded_data.shape == (3, 2)

    def test_load_csv_file_columns(self, tmp_path):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path, index=False)
        loaded_data = load_csv_file(str(csv_path))
        assert list(loaded_data.columns) == ['col1', 'col2']

    def test_load_csv_file_col1_values(self, tmp_path):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path, index=False)
        loaded_data = load_csv_file(str(csv_path))
        assert loaded_data['col1'].tolist() == [1, 2, 3]

    def test_load_csv_file_col2_values(self, tmp_path):
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path, index=False)
        loaded_data = load_csv_file(str(csv_path))
        assert loaded_data['col2'].tolist() == ['a', 'b', 'c']

    def test_load_csv_file_latin1_type(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé']})
        csv_path = tmp_path / "test_latin1.csv"
        test_data.to_csv(csv_path, index=False, encoding='latin1')
        loaded_data = load_csv_file(str(csv_path))
        assert isinstance(loaded_data, pd.DataFrame)

    def test_load_csv_file_latin1_shape(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé']})
        csv_path = tmp_path / "test_latin1.csv"
        test_data.to_csv(csv_path, index=False, encoding='latin1')
        loaded_data = load_csv_file(str(csv_path))
        assert loaded_data.shape == (3, 1)

    def test_load_csv_file_latin1_columns(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé']})
        csv_path = tmp_path / "test_latin1.csv"
        test_data.to_csv(csv_path, index=False, encoding='latin1')
        loaded_data = load_csv_file(str(csv_path))
        assert list(loaded_data.columns) == ['text']

    def test_load_csv_file_latin1_values(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé']})
        csv_path = tmp_path / "test_latin1.csv"
        test_data.to_csv(csv_path, index=False, encoding='latin1')
        loaded_data = load_csv_file(str(csv_path))
        assert loaded_data['text'].tolist() == ['café', 'naïve', 'résumé']

    def test_write_csv_file_exists(self, tmp_path):
        test_data = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'city': ['New York', 'London', 'Paris']})
        csv_path = tmp_path / "output.csv"
        write_csv_file(test_data, str(csv_path))
        assert csv_path.exists()

    def test_write_csv_file_shape(self, tmp_path):
        test_data = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'city': ['New York', 'London', 'Paris']})
        csv_path = tmp_path / "output.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data.shape == (3, 3)

    def test_write_csv_file_columns(self, tmp_path):
        test_data = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'city': ['New York', 'London', 'Paris']})
        csv_path = tmp_path / "output.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert list(loaded_data.columns) == ['name', 'age', 'city']

    def test_write_csv_file_name_values(self, tmp_path):
        test_data = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'city': ['New York', 'London', 'Paris']})
        csv_path = tmp_path / "output.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data['name'].tolist() == ['Alice', 'Bob', 'Charlie']

    def test_write_csv_file_age_values(self, tmp_path):
        test_data = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'city': ['New York', 'London', 'Paris']})
        csv_path = tmp_path / "output.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data['age'].tolist() == [25, 30, 35]

    def test_write_csv_file_city_values(self, tmp_path):
        test_data = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'city': ['New York', 'London', 'Paris']})
        csv_path = tmp_path / "output.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data['city'].tolist() == ['New York', 'London', 'Paris']

    def test_write_csv_file_no_index_unnamed(self, tmp_path):
        test_data = pd.DataFrame({'value': [1, 2, 3]})
        csv_path = tmp_path / "no_index.csv"
        write_csv_file(test_data, str(csv_path))
        with open(csv_path, 'r') as f:
            content = f.read()
        assert 'Unnamed: 0' not in content

    def test_write_csv_file_no_index_value(self, tmp_path):
        test_data = pd.DataFrame({'value': [1, 2, 3]})
        csv_path = tmp_path / "no_index.csv"
        write_csv_file(test_data, str(csv_path))
        with open(csv_path, 'r') as f:
            content = f.read()
        assert 'value' in content

    def test_write_csv_file_empty_exists(self, tmp_path):
        empty_df = pd.DataFrame()
        csv_path = tmp_path / "empty.csv"
        write_csv_file(empty_df, str(csv_path))
        assert csv_path.exists()

    def test_write_csv_file_empty_content(self, tmp_path):
        empty_df = pd.DataFrame()
        csv_path = tmp_path / "empty.csv"
        write_csv_file(empty_df, str(csv_path))
        with open(csv_path, 'r') as f:
            content = f.read()
        assert content.strip() == ""

    def test_write_csv_file_special_exists(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé'], 'number': [1, 2, 3]})
        csv_path = tmp_path / "special_chars.csv"
        write_csv_file(test_data, str(csv_path))
        assert csv_path.exists()

    def test_write_csv_file_special_shape(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé'], 'number': [1, 2, 3]})
        csv_path = tmp_path / "special_chars.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data.shape == (3, 2)

    def test_write_csv_file_special_text_values(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé'], 'number': [1, 2, 3]})
        csv_path = tmp_path / "special_chars.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data['text'].tolist() == ['café', 'naïve', 'résumé']

    def test_write_csv_file_special_number_values(self, tmp_path):
        test_data = pd.DataFrame({'text': ['café', 'naïve', 'résumé'], 'number': [1, 2, 3]})
        csv_path = tmp_path / "special_chars.csv"
        write_csv_file(test_data, str(csv_path))
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data['number'].tolist() == [1, 2, 3]

    def test_load_csv_file_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv_file("nonexistent_file.csv")

    def test_write_csv_file_nonexistent_directory(self):
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        write_csv_file(test_data, "nonexistent/directory/file.csv")
        import os
        assert os.path.exists("nonexistent/directory/file.csv")
        os.remove("nonexistent/directory/file.csv")
        os.rmdir("nonexistent/directory")
        os.rmdir("nonexistent") 