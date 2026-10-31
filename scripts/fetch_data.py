"""Stage 01 (local): download NIFTY 50 / India VIX candles from Angel One SmartAPI
and cross-asset daily series from yfinance, archived as parquet.

Setup (Python 3.13; hmmlearn has no 3.14 wheel yet):
    pip install uv
    uv venv .venv --python 3.13   (uv downloads Python 3.13 if it is not installed)
    .venv\\Scripts\\activate
    uv pip install -r requirements.txt
    copy .env.example .env   (then fill in your credentials)

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --intervals FIFTEEN_MINUTE ONE_DAY --skip-yf
"""
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vrdet.angel import TOKENS, clean_intraday, load_env, login, request_candles, to_frame  # noqa: E402
from vrdet.quality import session_report, yearly_summary  # noqa: E402

# Max days per historical request for each interval (SmartAPI limits)
MAX_DAYS = {'ONE_MINUTE': 30, 'FIVE_MINUTE': 100, 'FIFTEEN_MINUTE': 200,
            'ONE_HOUR': 400, 'ONE_DAY': 2000}

# interval -> (file tag, expected bars per full session; None for daily)
INTERVALS = {'ONE_MINUTE': ('1m', 375), 'FIVE_MINUTE': ('5m', 75), 'FIFTEEN_MINUTE': ('15m', 25),
             'ONE_HOUR': ('1h', 7), 'ONE_DAY': ('1d', None)}

YF_TICKERS = {'USDINR': 'USDINR=X', 'BRENT': 'BZ=F', 'NIFTY_YF': '^NSEI'}


def fetch_history(smart, name, interval, floor=datetime(2010, 1, 1), max_empty=2):
    """Walk backwards from today in max-size chunks until history runs out."""
    token, step = TOKENS[name], timedelta(days=MAX_DAYS[interval] - 1)
    end, rows, empty = datetime.now(), [], 0
    while end > floor and empty < max_empty:
        start = max(end - step, floor)
        chunk = request_candles(smart, token, interval, start.replace(hour=9, minute=0), end.replace(hour=15, minute=30))
        empty = 0 if chunk else empty + 1
        rows.extend(chunk)
        print(f'  {start:%Y-%m-%d} -> {end:%Y-%m-%d}: {len(chunk)} bars')
        end = start - timedelta(days=1)
        time.sleep(0.4)
    return to_frame(rows)


def quality_report(df, name, bars_per_day=None):
    if df.empty:
        print(f'{name}: NO DATA')
        return
    days = df.groupby(df.index.date).size()
    print(f'{name}: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}, {len(days):,} sessions')
    if bars_per_day:
        short = days[days < bars_per_day]
        print(f'  sessions with < {bars_per_day} bars: {len(short)} (Muhurat/half days, or gaps)')


def fetch_yfinance(out):
    import yfinance as yf

    for name, ticker in YF_TICKERS.items():
        df = yf.download(ticker, start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df.to_parquet(out / f'{name}_1d.parquet')
        print(f'{name}: {len(df):,} rows, {df.index[0].date()} -> {df.index[-1].date()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--intervals', nargs='+', default=['FIFTEEN_MINUTE', 'FIVE_MINUTE', 'ONE_DAY'],
                        choices=list(INTERVALS))
    parser.add_argument('--skip-yf', action='store_true', help='skip yfinance cross-asset series')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    load_env()
    smart = login()
    for name in TOKENS:
        for interval in args.intervals:
            tag, bars_per_day = INTERVALS[interval]
            print(f'Downloading {name} {tag} ...')
            df = fetch_history(smart, name, interval)
            if bars_per_day and not df.empty:
                df = clean_intraday(df)
            quality_report(df, f'{name}_{tag}', bars_per_day)
            df.to_parquet(args.out / f'{name}_{tag}.parquet')

    if not args.skip_yf:
        fetch_yfinance(args.out)
    validate(args.out)
    print('Saved to', args.out)


def validate(out):
    """Check every intraday session against the official daily candles; save per-session reports."""
    for name in TOKENS:
        daily_path = out / f'{name}_1d.parquet'
        if not daily_path.exists():
            continue
        daily = pd.read_parquet(daily_path)
        daily.index = daily.index.tz_localize(None).normalize()
        for tag, bars_per_day in (v for v in INTERVALS.values() if v[1]):
            path = out / f'{name}_{tag}.parquet'
            if not path.exists():
                continue
            rep = session_report(pd.read_parquet(path), daily, bars_per_day)
            rep.to_csv(out / f'{name}_{tag}_sessions.csv')
            print(f'\nValidation {name}_{tag} vs official daily (per year):')
            print(yearly_summary(rep).to_string())
            bad = rep.index[~rep['valid']]
            if len(bad):
                print(f'  invalid sessions: {len(bad)} ({bad.min()} .. {bad.max()}), see {name}_{tag}_sessions.csv')


if __name__ == '__main__':
    main()
