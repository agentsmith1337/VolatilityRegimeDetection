"""Angel One SmartAPI access: login from .env credentials and candle downloads.

Shared by scripts/fetch_data.py (history) and the live app (recent candles).
"""
import os
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENS = {'NIFTY': '99926000', 'INDIAVIX': '99926017'}


def load_env(path=REPO_ROOT / '.env'):
    """Copy KEY=VALUE lines from .env into os.environ (existing variables win)."""
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
        raise RuntimeError(f'Angel One login failed: {session.get("message", session)}')
    print('Logged in as', session['data'].get('name', os.environ['ANGEL_CLIENT_CODE']))
    return smart


def request_candles(smart, token, interval, start, end, retries=5):
    """Raw candle rows [timestamp, open, high, low, close, volume]; [] if every retry fails."""
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


def to_frame(rows):
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
    return df.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')


def clean_intraday(df, verbose=True):
    """Keep regular-session weekday bars (09:15-15:30 IST) with consistent OHLC."""
    t = df.index.time
    keep = (t >= pd.Timestamp('09:15').time()) & (t < pd.Timestamp('15:30').time()) & (df.index.dayofweek < 5)
    if (~keep).any() and verbose:
        dropped = df[~keep].groupby(df[~keep].index.date).size()
        print(f'  dropping {(~keep).sum()} bars outside the regular session on {len(dropped)} dates '
              f'(special/extended sessions): {", ".join(str(d) for d in dropped.index[:10])}'
              f'{" ..." if len(dropped) > 10 else ""}')
    df = df[keep]
    bad = (df['high'] < df[['open', 'close']].max(axis=1)) | (df['low'] > df[['open', 'close']].min(axis=1))
    if bad.any() and verbose:
        print(f'  dropping {bad.sum()} bars with inconsistent OHLC')
    return df[~bad]
