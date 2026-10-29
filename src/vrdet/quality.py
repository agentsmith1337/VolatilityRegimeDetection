"""Session-level data validation of intraday candles against the official daily series.

A session is *valid* when it is complete enough, not flat, and its intraday candles reproduce at
least 2 of the 3 official daily reference values (high, low, close) within `tol`. Requiring all
three is too strict: official closes are computed differently from the last traded price
(NIFTY: weighted average of the last 30 minutes; recent India VIX closes differ by ~0.35%), and
India VIX official daily lows sometimes contain a momentary opening-quote dip that no 15-minute
candle shows. Corrupt sessions (e.g. VIX scaled by 1/100) match none of the three.
"""
import numpy as np
import pandas as pd

from .features import bar_dates


def to_daily(intra):
    g = intra.groupby(bar_dates(intra.index))
    return pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                         'close': g['close'].last(), 'n_bars': g.size(),
                         'n_unique_close': g['close'].nunique(),
                         'zero_range_bars': g.apply(lambda d: int((d['high'] == d['low']).sum()))})


def session_report(intra, daily, bars_per_day=25, tol=0.01, min_bars=13):
    """One row per intraday session with completeness and agreement checks.

    daily: official daily OHLC with a naive date index (e.g. Angel One 1d or Yahoo).
    """
    s = to_daily(intra)
    d = daily[['high', 'low', 'close']].add_prefix('daily_')
    s = s.join(d, how='left')
    s['missing_bars'] = (bars_per_day - s['n_bars']).clip(lower=0)
    s['flat'] = s['n_unique_close'] <= 2
    for k in ('high', 'low', 'close'):
        s[f'err_{k}'] = (s[k] / s[f'daily_{k}'] - 1).abs()
    s['partial'] = s['n_bars'] < min_bars
    s['n_match'] = sum((s[f'err_{k}'] <= tol).astype(int) for k in ('high', 'low', 'close'))
    s['valid'] = ~s['flat'] & ~s['partial'] & ((s['n_match'] >= 2) | s['daily_high'].isna())
    return s


def yearly_summary(report):
    by = report.groupby(report.index.year)
    return pd.DataFrame({
        'sessions': by.size(),
        'valid_%': (by['valid'].mean() * 100).round(1),
        'flat': by['flat'].sum(),
        'partial': by['partial'].sum(),
        'missing_bars': by['missing_bars'].sum(),
        'zero_range_bars': by['zero_range_bars'].sum(),
        'high_match_%': (by['err_high'].apply(lambda e: (e <= 0.01).mean()) * 100).round(1),
        'low_match_%': (by['err_low'].apply(lambda e: (e <= 0.01).mean()) * 100).round(1),
        'close_match_%': (by['err_close'].apply(lambda e: (e <= 0.01).mean()) * 100).round(1),
        'median_close_err_%': (by['err_close'].median() * 100).round(3),
    })


def mask_invalid_sessions(intra, report):
    """Set all candles of invalid sessions to NaN (keeps the index)."""
    bad = report.index[~report['valid']]
    bad_rows = pd.Index(bar_dates(intra.index)).isin(bad)
    out = intra.copy()
    out.loc[bad_rows] = np.nan
    return out
