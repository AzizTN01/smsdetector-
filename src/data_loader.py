import os
import requests
import pandas as pd


def download_dataset(dest_path: str):
    """Download the SMSSpamCollection dataset and save to dest_path.

    Args:
        dest_path: path to write the file (including filename).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/SMSSpamCollection"
    resp = requests.get(url)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw SMSSpamCollection file into a DataFrame with columns ['label','text'].

    The dataset is tab-separated with no header.
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'text'], quoting=3)
    return df

