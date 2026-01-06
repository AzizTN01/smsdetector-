import re
import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Take raw df with columns ['label','text'] and return cleaned df with binary label column 'y'."""
    df = df.copy()
    df['text'] = df['text'].astype(str).map(clean_text)
    df['y'] = (df['label'] == 'spam').astype(int)
    return df[['text', 'y']]

