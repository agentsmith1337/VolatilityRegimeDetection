"""Locating and loading the raw parquet archive produced by scripts/fetch_data.py."""
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_raw_dir():
    """First existing candidate: $VRDET_DATA, <repo>/data/raw."""
    candidates = [os.environ.get('VRDET_DATA'), REPO_ROOT / 'data' / 'raw']
    for c in candidates:
        if c and Path(c).exists() and any(Path(c).rglob('*.parquet')):
            return Path(c)
    raise FileNotFoundError(f'No parquet files in any of {candidates[1:]} (or set VRDET_DATA)')


def load_raw(raw_dir=None):
    """Dict of DataFrames keyed by file stem, e.g. 'NIFTY_15m', 'USDINR_1d'.

    Intraday frames keep their Asia/Kolkata tz-aware index; daily frames get a naive date index.
    """
    raw_dir = Path(raw_dir) if raw_dir else find_raw_dir()
    data = {}
    for p in sorted(raw_dir.rglob('*.parquet')):
        df = pd.read_parquet(p)
        if p.stem.endswith('_1d'):
            idx = pd.DatetimeIndex(df.index)
            df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        data[p.stem] = df
    return data
