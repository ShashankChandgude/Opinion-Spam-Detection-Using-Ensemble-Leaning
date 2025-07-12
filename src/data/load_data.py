import pandas as pd
from pathlib import Path
from functools import lru_cache
from src.utils.helpers import get_project_root
from src.utils.interfaces import DataLoader
from src.utils.config import config
from src.data.data_io import load_csv_file

class FileDataLoader(DataLoader):
    def __init__(self):
        self.root = get_project_root()
    @lru_cache(maxsize=1)
    def load(self) -> pd.DataFrame:
        raw_csv = config.get_raw_data_path(self.root)
        return load_csv_file(raw_csv)

@lru_cache(maxsize=1)
def load_data(root) -> pd.DataFrame:
    loader = FileDataLoader()
    return loader.load()
