"""Causal feature engineering for intraday NIFTY bars.

Every feature at bar t uses only data available at the close of bar t. Intraday volatility
seasonality (the U-shape: busy open and close, quiet lunch) is removed with a time-of-day
profile estimated on *previous* sessions only.
"""
import numpy as np
import pandas as pd

SESSION_START_MIN = 9 * 60 + 15
SESSION_LENGTH_MIN = 375
GK_K = 2 * np.log(2) - 1


def bar_dates(index):
    """Naive session date for each bar of a tz-aware intraday index."""
    return index.tz_localize(None).normalize()


def seasonal_shape(x, min_days=20):
    """Causal time-of-day profile of a non-negative series, normalised to mean 1 across the day.

    For a bar on day d at time-of-day k, uses the mean of x at time k over days strictly before d.
    """
    df = pd.DataFrame({'x': x.to_numpy(), 'date': bar_dates(x.index), 'tod': x.index.time})
    wide = df.pivot_table(index='date', columns='tod', values='x', aggfunc='mean')
    prof = wide.expanding(min_periods=min_days).mean().shift(1)
    shape = prof.div(prof.mean(axis=1), axis=0)
    lookup = shape.stack()
    keys = pd.MultiIndex.from_arrays([df['date'], df['tod']])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=x.index)


def _prev_day_lookup(daily, dates):
    """Value of a daily series from the last trading day strictly before each date (causal)."""
    daily = daily.dropna()
    pos = np.searchsorted(daily.index.values, dates.values, side='left') - 1
    out = np.where(pos >= 0, daily.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return out


def build_features(nifty, vix=None, daily=None, nifty_daily=None, vix_daily=None, bars_per_day=25,
                   windows=(4, 8, 25, 125)):
    """Feature frame indexed by the valid sessions of `nifty`.

    nifty, vix: intraday OHLC frames (tz-aware index). nifty_daily, vix_daily: official daily OHLC
    used to validate each intraday session (see vrdet.quality); invalid NIFTY sessions are dropped,
    invalid VIX sessions are set to NaN. vix_daily is required when vix is given.
    daily: optional dict name -> daily frame with a 'close' column and naive date index
    (e.g. {'usdinr': ..., 'brent': ...}). Volatility columns are annualised percentages.
    """
    from .quality import session_report, mask_invalid_sessions

    if nifty_daily is not None:
        rep = session_report(nifty, nifty_daily, bars_per_day)
        nifty = nifty[pd.Index(bar_dates(nifty.index)).isin(rep.index[rep['valid']])]
    if vix is not None:
        if vix_daily is None:
            raise ValueError('vix_daily is required to validate intraday VIX sessions')
        vix = mask_invalid_sessions(vix, session_report(vix, vix_daily, bars_per_day))

    o, h, l, c = (nifty[k].astype(float) for k in ('open', 'high', 'low', 'close'))
    idx = nifty.index
    dates = pd.Series(bar_dates(idx), index=idx)
    new_day = dates != dates.shift(1)
    bar_len = pd.Timedelta(minutes=SESSION_LENGTH_MIN // bars_per_day)
    contiguous = pd.Series(idx, index=idx).diff() == bar_len   # False across missing candles
    ann = np.sqrt(bars_per_day * 252) * 100
    f = pd.DataFrame(index=idx)

    def roll(x, w):
        return x.rolling(w, min_periods=int(np.ceil(0.75 * w)))

    # Returns: first bar of a session uses close/open; the overnight move is a separate feature.
    # Returns spanning missing candles are NaN rather than a fake 15-minute move.
    r = np.log(c / c.shift(1)).where(contiguous & ~new_day)
    r[new_day] = np.log(c / o)[new_day]
    f['ret'] = r
    f['gap'] = np.log(o / c.shift(1)).where(new_day).ffill()

    # Deseasonalised per-bar quantities. Zero-range candles (e.g. the single-print closing candle
    # in the vendor feed since Aug 2026) carry no range information -> NaN, not "very quiet".
    shape_r = seasonal_shape(r.abs())
    hl = np.log(h / l).where(h > l)
    shape_hl = seasonal_shape(hl)
    z = r / shape_r
    f['z'] = z
    f['range_ds'] = hl / shape_hl
    pk = f['range_ds'] ** 2 / (4 * np.log(2))
    gk = (0.5 * hl ** 2 - GK_K * np.log(c / o) ** 2).clip(lower=0) / shape_hl ** 2

    for w in windows:
        f[f'rv_{w}'] = np.sqrt(roll(z ** 2, w).mean()) * ann
        f[f'pk_{w}'] = np.sqrt(roll(pk, w).mean()) * ann
        f[f'gk_{w}'] = np.sqrt(roll(gk, w).mean()) * ann
    short, long_ = windows[1], windows[-1]
    f['rv_ratio'] = f[f'rv_{short}'] / f[f'rv_{long_}']
    f['pk_ratio'] = f[f'pk_{short}'] / f[f'pk_{long_}']
    log_rv = np.log(f[f'pk_{windows[2]}'])
    f['pk_z'] = (log_rv - log_rv.rolling(100 * bars_per_day, min_periods=20 * bars_per_day).mean()) \
        / log_rv.rolling(100 * bars_per_day, min_periods=20 * bars_per_day).std()

    # Trend / path context
    for w in (windows[2], windows[-1]):
        f[f'ret_{w}'] = roll(r, w).sum()
        f[f'ema_dist_{w}'] = c / c.ewm(span=w, adjust=False).mean() - 1
    f['drawdown'] = c / c.rolling(windows[-1], min_periods=1).max() - 1
    f['day_ret'] = np.log(c / o.groupby(dates).transform('first'))

    # Calendar
    minutes = idx.hour * 60 + idx.minute - SESSION_START_MIN
    f['tod_sin'] = np.sin(2 * np.pi * minutes / SESSION_LENGTH_MIN)
    f['tod_cos'] = np.cos(2 * np.pi * minutes / SESSION_LENGTH_MIN)
    f['dow'] = idx.dayofweek
    f['first_hour'] = (minutes < 60).astype(int)
    f['last_hour'] = (minutes >= SESSION_LENGTH_MIN - 60).astype(int)

    # Implied volatility (India VIX bars share NIFTY timestamps; short ffill = last known value only,
    # invalid sessions stay NaN rather than carrying a stale value forward)
    if vix is not None:
        v = vix['close'].astype(float).reindex(idx).ffill(limit=bars_per_day)
        f['vix'] = v
        for w in (1, windows[0], windows[2]):
            f[f'vix_chg_{w}'] = np.log(v / v.shift(w))
        f['vix_rv_spread'] = v - f[f'pk_{windows[2]}']
        f['vix_z'] = (v - v.rolling(100 * bars_per_day, min_periods=20 * bars_per_day).mean()) \
            / v.rolling(100 * bars_per_day, min_periods=20 * bars_per_day).std()

    # Cross-asset daily series: previous trading day's values only
    for name, d in (daily or {}).items():
        dr = np.log(d['close'].astype(float)).diff()
        f[f'{name}_ret_1d'] = _prev_day_lookup(dr, dates)
        f[f'{name}_vol_5d'] = _prev_day_lookup(dr.rolling(5).std() * np.sqrt(252) * 100, dates)

    return f.replace([np.inf, -np.inf], np.nan)


def add_lags(f, cols, lags=(1, 2, 4)):
    """Append lagged copies of `cols` (for tree models)."""
    lagged = {f'{c}_lag{k}': f[c].shift(k) for c in cols for k in lags}
    return pd.concat([f, pd.DataFrame(lagged, index=f.index)], axis=1)
