import pandas as pd

def load_csv_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding="latin1")

def write_csv_file(data_frame: pd.DataFrame, file_path: str) -> None:
    data_frame.to_csv(file_path, index=False)
