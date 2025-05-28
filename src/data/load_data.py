import os
import pandas as pd
from src.utils.helpers import get_project_root

def load_data(root) -> pd.DataFrame:
    raw_csv = os.path.join(root, "data", "raw", "deceptive-opinion-corpus.csv")
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Expected dataset at {raw_csv}")
    df = pd.read_csv(raw_csv)
    return df
