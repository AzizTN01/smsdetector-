from src.data_loader import load_raw
import os

def test_load_raw_temp(tmp_path):
    sample = "ham\tHello there\nspam\tWin money now"
    p = tmp_path / "SMSSpamCollection"
    p.write_text(sample)
    df = load_raw(str(p))
    assert 'label' in df.columns
    assert 'text' in df.columns
    assert len(df) == 2

