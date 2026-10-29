"""Stage 01 (local): download NIFTY 50 / India VIX candles from Angel One SmartAPI
and cross-asset daily series from yfinance, archived as parquet.

Runs on the local machine (broker APIs are unreliable from Colab's cloud IPs).
The parquet files are then uploaded to Google Drive for the Colab notebooks.

Setup:
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install smartapi-python pyotp logzero websocket-client yfinance pandas pyarrow
    copy .env.example .env   (then fill in your credentials)

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --out "G:/My Drive/vrdet/data/raw"   # Google Drive for desktop
    python scripts/fetch_data.py --intervals FIFTEEN_MINUTE ONE_DAY --skip-yf
"""
import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vrdet.quality import session_report, yearly_summary  # noqa: E402

TOKENS = {'NIFTY': '99926000', 'INDIAVIX': '99926017'}

# Max days per historical request for each interval (SmartAPI limits)
MAX_DAYS = {'ONE_MINUTE': 30, 'FIVE_MINUTE': 100, 'FIFTEEN_MINUTE': 200,
            'ONE_HOUR': 400, 'ONE_DAY': 2000}

# interval -> (file tag, expected bars per full session; None for daily)
INTERVALS = {'ONE_MINUTE': ('1m', 375), 'FIVE_MINUTE': ('5m', 75), 'FIFTEEN_MINUTE': ('15m', 25),
             'ONE_HOUR': ('1h', 7), 'ONE_DAY': ('1d', None)}

YF_TICKERS = {'USDINR': 'USDINR=X', 'BRENT': 'BZ=F', 'NIFTY_YF': '^NSEI'}


def load_env(path=ROOT / '.env'):
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def login():
    import pyotp
    from SmartApi import SmartConnect

    smart = SmartConnect(api_key=os.environ['ANGEL_API_KEY'])
    session = smart.generateSession(os.environ['ANGEL_CLIENT_CODE'], os.environ['ANGEL_PIN'],
                                    pyotp.TOTP(os.environ['ANGEL_TOTP_SECRET']).now())
    if not session.get('status'):
        raise SystemExit(f'Login failed: {session}')
    print('Logged in as', session['data'].get('name', os.environ['ANGEL_CLIENT_CODE']))
    return smart


def _request(smart, token, interval, start, end, retries=5):
    params = {'exchange': 'NSE', 'symboltoken': token, 'interval': interval,
              'fromdate': start.strftime('%Y-%m-%d %H:%M'),
              'todate': end.strftime('%Y-%m-%d %H:%M')}
    msg = ''
    for attempt in range(retries):
        try:
            resp = smart.getCandleData(params)
            if resp and resp.get('status'):
                return resp.get('data') or []
            msg = (resp or {}).get('message', '')
        except Exception as e:
            msg = str(e)
        time.sleep(1.5 * (attempt + 1))
    print(f'  giving up on {params["fromdate"]} -> {params["todate"]}: {msg}')
    return []


def fetch_history(smart, name, interval, floor=datetime(2010, 1, 1), max_empty=2):
    """Walk backwards from today in max-size chunks until history runs out."""
    token, step = TOKENS[name], timedelta(days=MAX_DAYS[interval] - 1)
    end, rows, empty = datetime.now(), [], 0
    while end > floor and empty < max_empty:
        start = max(end - step, floor)
        chunk = _request(smart, token, interval, start.replace(hour=9, minute=0), end.replace(hour=15, minute=30))
        empty = 0 if chunk else empty + 1
        rows.extend(chunk)
        print(f'  {start:%Y-%m-%d} -> {end:%Y-%m-%d}: {len(chunk)} bars')
        end = start - timedelta(days=1)
        time.sleep(0.4)
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
    return df.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')


def clean_intraday(df):
    t = df.index.time
    keep = (t >= pd.Timestamp('09:15').time()) & (t < pd.Timestamp('15:30').time()) & (df.index.dayofweek < 5)
    if (~keep).any():
        dropped = df[~keep].groupby(df[~keep].index.date).size()
        print(f'  dropping {(~keep).sum()} bars outside the regular session on {len(dropped)} dates '
              f'(special/extended sessions): {", ".join(str(d) for d in dropped.index[:10])}'
              f'{" ..." if len(dropped) > 10 else ""}')
    df = df[keep]
    bad = (df['high'] < df[['open', 'close']].max(axis=1)) | (df['low'] > df[['open', 'close']].min(axis=1))
    if bad.any():
        print(f'  dropping {bad.sum()} bars with inconsistent OHLC')
    return df[~bad]


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
