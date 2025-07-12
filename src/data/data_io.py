import pandas as pd
from pathlib import Path
from typing import Optional

def load_csv_file(file_path: str, encoding: str = "latin1") -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(path, encoding=encoding)

def write_csv_file(data_frame: pd.DataFrame, file_path: str, index: bool = False) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(path, index=index)
