import pandas as pd
from src.data.data_io import load_csv_file, write_csv_file

def test_loads_dataframe_from_csv(tmp_path):
    dataframe = pd.DataFrame({'number': [1, 2], 'letter': list('xy')})
    file_path = tmp_path / "input.csv"
    dataframe.to_csv(file_path, index=False)
    loaded = load_csv_file(str(file_path))
    assert loaded.equals(dataframe)

def test_writes_dataframe_to_csv(tmp_path):
    dataframe = pd.DataFrame({'alpha': [3, 4], 'beta': list('uv')})
    file_path = tmp_path / "output.csv"
    write_csv_file(dataframe, str(file_path))
    reloaded = pd.read_csv(file_path)
    assert reloaded.equals(dataframe)