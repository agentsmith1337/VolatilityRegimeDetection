"""Regime models for any NSE stock: data from Angel One SmartAPI, same pipeline as the NIFTY 50 study.

For a stock: download (and cache) 15-minute and daily candles -> validate sessions -> causal features (the
stock's own bars + India VIX + USD/INR + Brent) -> fit an HMM labeler on the training span only -> hindsight
labels as target -> train the selected zoo models -> test on the last `test_months` (never seen in training)
-> refit on all data -> score the latest bars (current regime).

Artefacts: data/stocks/<SYMBOL>_{15m,1d}.parquet (candle cache), models/stocks/<SYMBOL>/ (hmm.json,
<model>.joblib, meta.json with test results and the training log, current.parquet with the latest calls).
"""
import json
import re
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from .angel import REPO_ROOT, clean_intraday, load_env, login, request_candles, to_frame
from .classifiers import TunedChaoticThreshold
from .data import load_raw
from .evaluate import regime_episodes_count, regime_metrics
from .labels import STATE_NAMES, HMMLabeler
from .live import BAR, MODEL_DIR, TZ, LiveData, base_frame, classify, model_frame
from .zoo import CONTEXT, MAKERS, MODELS

DATA_DIR = REPO_ROOT / 'data' / 'stocks'
MODELS_DIR = MODEL_DIR / 'stocks'
META_DIR = REPO_ROOT / 'data' / 'meta'
SCRIP_MASTER = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'
HISTORY_START = pd.Timestamp('2017-01-01')   # Angel One 15m stock history (and valid India VIX) start here
MAX_DAYS = {'FIFTEEN_MINUTE': 199, 'ONE_DAY': 1999}
GAP_LIMIT = np.log(1.35)                      # overnight moves beyond +-35% are treated as corporate actions
SHOWN_SESSIONS = 10


def safe(symbol):
    return re.sub(r'[^A-Za-z0-9]+', '_', symbol)


def instruments(max_age_days=7):
    """NSE cash equities from Angel One's public instrument master (cached): columns token, symbol, name."""
    META_DIR.mkdir(parents=True, exist_ok=True)
    path = META_DIR / 'nse_equities.parquet'
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return pd.read_parquet(path)
    import requests
    d = pd.DataFrame(requests.get(SCRIP_MASTER, timeout=120).json())
    eq = d[(d['exch_seg'] == 'NSE') & d['symbol'].str.endswith('-EQ')][['token', 'symbol', 'name']]
    eq = eq.drop_duplicates('name').sort_values('name').reset_index(drop=True)
    eq.to_parquet(path)
    return eq


def _fetch_range(smart, token, interval, start, end):
    rows, step = [], timedelta(days=MAX_DAYS[interval])
    while start < end:
        stop = min(start + step, end)
        rows += request_candles(smart, token, interval, start.replace(hour=9, minute=0), stop.replace(hour=15, minute=30))
        start = stop + timedelta(days=1)
        time.sleep(0.4)  # SmartAPI rate limit
    return to_frame(rows) if rows else None


def _yf_daily(symbol, start):
    """Fallback daily reference candles from Yahoo Finance (<SYMBOL>.NS)."""
    import yfinance as yf
    df = yf.download(f'{symbol}.NS', start=str(start.date()), interval='1d', auto_adjust=False, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize(TZ)
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_stock(smart, symbol, token, years, log, now=None):
    """15m + daily candles since `years` ago (not before 2017), cached and topped up incrementally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = now or pd.Timestamp.now(tz=TZ)
    want = max(HISTORY_START, now.tz_localize(None).normalize() - pd.DateOffset(years=years))
    out = {}
    for interval, tag in (('FIFTEEN_MINUTE', '15m'), ('ONE_DAY', '1d')):
        path = DATA_DIR / f'{safe(symbol)}_{tag}.parquet'
        cache = pd.read_parquet(path) if path.exists() else None
        if cache is not None and cache.index[0].tz_localize(None) <= want + pd.Timedelta(days=7):
            start = cache.index[-1].tz_localize(None).to_pydatetime() - timedelta(days=3)
        else:
            cache, start = None, want.to_pydatetime()
        t0 = time.perf_counter()
        new = _fetch_range(smart, token, interval, start, now.tz_localize(None).to_pydatetime())
        if tag == '1d' and (new is None or new.empty) and cache is None:
            log('  Angel One returned no daily candles, using Yahoo Finance as the daily reference')
            new = _yf_daily(symbol, want)
        if new is not None and not new.empty:
            if tag == '15m':
                new = clean_intraday(new, verbose=False)
                new = new[new.index + BAR <= now]                        # completed bars only
            else:
                new = new[new.index.tz_convert(TZ).normalize() < now.normalize()]  # today's candle still forming
            cache = new if cache is None else pd.concat([cache[~cache.index.isin(new.index)], new]).sort_index()
        if cache is None or cache.empty:
            raise RuntimeError(f'No {tag} candles for {symbol} from Angel One')
        cache = cache[cache.index.tz_localize(None) >= want]
        cache.to_parquet(path)
        out[tag] = cache
        log(f'  {tag}: {len(cache):,} candles, {cache.index[0]:%Y-%m-%d} -> {cache.index[-1]:%Y-%m-%d %H:%M} '
            f'({time.perf_counter() - t0:.0f}s)')
    return out['15m'], out['1d']


def align_to_daily(intra, daily, log):
    """Rescale intraday sessions whose level differs from the official daily series by a steady factor.

    Angel One sometimes adjusts its intraday and daily archives differently for corporate actions (e.g.
    RELIANCE intraday before 2022-01-05 is x1.10 the daily series, which creates a fake -9.5% overnight jump).
    The factor per session is the median of daily close / intraday close over 11 neighbouring sessions, so a
    single bad session does not move it; it is a level correction only (features are scale-free within a session).
    """
    day = intra.index.tz_convert(TZ).tz_localize(None).normalize()
    last = intra['close'].groupby(day).last()
    ref = _naive_daily(daily)['close'].reindex(last.index)
    factor = (ref / last).rolling(11, center=True, min_periods=3).median().ffill().bfill().fillna(1.0)
    factor = factor.where((factor - 1).abs() > 0.005, 1.0).round(4)
    if (factor != 1).any():
        from .evaluate import runs
        for a, b in runs(factor.to_numpy() != 1):
            log(f'  intraday level x{factor.iloc[a:b].median():.4f} on {b - a} sessions ({factor.index[a]:%Y-%m-%d} -> '
                f'{factor.index[b - 1]:%Y-%m-%d}) to match the official daily series (vendor adjusted them differently)')
        intra = intra.copy()
        cols = ['open', 'high', 'low', 'close']
        intra[cols] = intra[cols].mul(factor.reindex(day).to_numpy(), axis=0)
    return intra


def adjust_corporate_actions(intra, daily, log):
    """Back-adjust prices before overnight jumps beyond +-35% (splits / bonuses missed by the vendor)."""
    first = intra.groupby(intra.index.normalize()).head(1)
    prev_close = intra['close'].groupby(intra.index.normalize()).last().shift(1)
    gap = np.log(first['open'].to_numpy() / prev_close.reindex(first.index.normalize()).to_numpy())
    events = [(t, np.exp(g)) for t, g in zip(first.index, gap) if np.isfinite(g) and abs(g) > GAP_LIMIT]
    intra, daily = intra.copy(), daily.copy()
    for t, ratio in events:
        cols = ['open', 'high', 'low', 'close']
        intra.loc[intra.index < t, cols] *= ratio
        daily.loc[daily.index.tz_convert(TZ).normalize() < t.normalize(), cols] *= ratio
        log(f'  corporate action? overnight jump x{ratio:.3f} on {t:%Y-%m-%d}: earlier prices adjusted')
    return intra, daily


def _naive_daily(df):
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return df


def market_data(smart, now, log):
    """India VIX + cross-asset series (research archive + live cache, refreshed from Angel One / yfinance)."""
    live = LiveData(load_raw())
    try:
        live.refresh_angel(smart, now)
        live.refresh_yf(now)
    except Exception as e:  # stale market context only affects the last bars' VIX / cross-asset inputs
        log(f'  market data refresh failed ({e}); using cached India VIX / cross-asset data')
    return live.frames()


def stock_raw(symbol, intra, daily, market):
    raw = {k: v for k, v in market.items() if not k.startswith('NIFTY')}
    raw[f'{symbol}_15m'], raw[f'{symbol}_1d'] = intra, _naive_daily(daily)
    return raw


def train_stock(symbol, models, years=8, test_months=12, log=print, smart=None):
    """Full pipeline for one stock; returns the result dict also saved to models/stocks/<SYMBOL>/meta.json."""
    import joblib
    from sklearn.metrics import average_precision_score
    import logging
    logging.getLogger('hmmlearn').setLevel(logging.ERROR)  # benign "not converging" notes from the sticky prior
    t_all = time.perf_counter()
    models = [k for k in MODELS if k in models]            # zoo order, unknown keys dropped
    out_dir = MODELS_DIR / safe(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=TZ)
    log(f'=== {symbol}: {len(models)} model(s), {years} years of data, test = last {test_months} months ===')

    log('[1/5] Data (Angel One SmartAPI)')
    if smart is None:
        load_env()
        smart = login()
    eq = instruments()
    match = eq[eq['name'] == symbol]
    if match.empty:
        raise ValueError(f'{symbol} is not an NSE equity in the Angel One instrument list')
    token = str(match['token'].iloc[0])
    intra, daily = fetch_stock(smart, symbol, token, years, log, now)
    intra = align_to_daily(intra, daily, log)
    intra, daily = adjust_corporate_actions(intra, daily, log)
    market = market_data(smart, now, log)
    raw = stock_raw(symbol, intra, daily, market)

    log('[2/5] Features and session validation')
    t0 = time.perf_counter()
    f = base_frame(raw, live=True, asset=symbol)
    sessions = intra.index.normalize().nunique()
    kept = f.index.normalize().nunique()
    log(f'  {len(f):,} bars, {kept:,}/{sessions:,} sessions valid ({time.perf_counter() - t0:.0f}s)')

    test_start = (now.normalize() - pd.DateOffset(months=test_months)).tz_convert(TZ)
    train_end = f.index[f.index < test_start][-1]
    log(f'[3/5] Regime labels: HMM fitted on {f.index[0]:%Y-%m-%d} -> {train_end:%Y-%m-%d} only')
    t0 = time.perf_counter()
    hmm = HMMLabeler().fit(f, end=train_end)
    X = model_frame(raw, hmm, base=f)
    y = hmm.hindsight(X)
    X = X.loc[y.index]
    test = X.index >= test_start
    shares = y.value_counts(normalize=True).reindex(range(3), fill_value=0)
    eps_train, eps_test = regime_episodes_count(y[~test]), regime_episodes_count(y[test])
    log(f'  regime shares Quiet {shares[0]:.1%} / Normal {shares[1]:.1%} / Chaotic {shares[2]:.1%}; '
        f'Chaotic episodes: {eps_train} in training, {eps_test} in test ({time.perf_counter() - t0:.0f}s)')
    summary = hmm.summary()
    log('  typical 15-min range by regime: ' + ', '.join(
        f'{k} {v:.2%}' for k, v in summary['range_ds_typical'].items()))
    if shares[2] < 0.01 or shares[2] > 0.3:
        log('  WARNING: unusual Chaotic share for this stock; treat its results with care')
    if eps_test < 20:
        log(f'  NOTE: only {eps_test} Chaotic episodes in the test period, so test scores are noisy '
            f'(a longer test period gives steadier numbers)')
    (out_dir / 'hmm.json').write_text(hmm.to_json())

    log(f'[4/5] Train and test ({(~test).sum():,} training bars, {test.sum():,} test bars)')
    rows = np.flatnonzero(test)
    ctx = X.iloc[max(0, rows[0] - CONTEXT):]
    y_test = y[test]
    results = {}
    hmm_call = pd.Series(X.loc[test, ['p_hmm_Quiet', 'p_hmm_Normal', 'p_hmm_Chaotic']].to_numpy().argmax(axis=1),
                         index=y_test.index)
    results['hmm_filter'] = {**regime_metrics(y_test, hmm_call),
                             'pr_auc_chaotic': average_precision_score(y_test == 2, X.loc[test, 'p_hmm_Chaotic'])}
    for key in models:
        t0 = time.perf_counter()
        m = TunedChaoticThreshold(MAKERS[key]).fit(X[~test], y[~test])
        p = m.predict_proba(ctx)[-len(rows):]
        r = regime_metrics(y_test, pd.Series(classify(p, m.tau_), index=y_test.index))
        r['pr_auc_chaotic'] = average_precision_score(y_test == 2, p[:, 2])
        fit_test = time.perf_counter() - t0
        t0 = time.perf_counter()
        final = TunedChaoticThreshold(MAKERS[key]).fit(X, y)
        joblib.dump({'model': final.model_, 'tau': final.tau_, 'features': list(X.columns)}, out_dir / f'{key}.joblib')
        results[key] = {**r, 'tau': final.tau_, 'seconds': round(fit_test + time.perf_counter() - t0, 1)}
        log(f'  {MODELS[key][0]:<20} macro-F1 {r["macro_f1"]:.3f} | Chaotic P/R {r["chaotic_precision"]:.2f}/'
            f'{r["chaotic_recall"]:.2f} | PR-AUC {r["pr_auc_chaotic"]:.3f} | episodes caught '
            f'{r["episode_recall"]:.0%} | false alarms/wk {r["false_alarms_per_week"]:.2f} '
            f'({results[key]["seconds"]:.0f}s)')
    table = pd.DataFrame(results).T[['macro_f1', 'chaotic_precision', 'chaotic_recall', 'pr_auc_chaotic',
                                     'episode_recall', 'median_delay_bars', 'false_alarms_per_week']]
    table.index = [MODELS[k][0] if k in MODELS else 'HMM filter (benchmark)' for k in table.index]
    table.columns = ['F1', 'Prec', 'Recall', 'PR-AUC', 'Episodes', 'Delay', 'FA/wk']
    log('\n  test results (Prec / Recall / PR-AUC for Chaotic; Delay in bars; FA/wk = false alarms per week)\n'
        + table.astype(float).round(3).to_string() + '\n')

    log('[5/5] Current regime')
    meta = {'symbol': symbol, 'token': token, 'models': models, 'years': years, 'test_months': test_months,
            'trained_at': now.isoformat(timespec='seconds'), 'test_start': str(test_start),
            'bars': len(X), 'sessions_valid': int(kept), 'sessions_total': int(sessions),
            'label_shares': {STATE_NAMES[i]: float(shares[i]) for i in range(3)},
            'chaotic_episodes': {'train': eps_train, 'test': eps_test}, 'results': results}
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1, default=float))
    current = score_latest(symbol, raw=raw, hmm=hmm, log=log, base=f)
    log(f'=== done in {time.perf_counter() - t_all:.0f}s ===')
    return meta, current


def score_latest(symbol, raw=None, hmm=None, log=print, base=None, smart=None):
    """Probabilities of every trained model for the last sessions; refetches candles when raw is None."""
    import joblib
    out_dir = MODELS_DIR / safe(symbol)
    meta = json.loads((out_dir / 'meta.json').read_text())
    if raw is None:
        if smart is None:
            load_env()
            smart = login()
        now = pd.Timestamp.now(tz=TZ)
        intra, daily = fetch_stock(smart, symbol, meta['token'], meta['years'], log, now)
        intra = align_to_daily(intra, daily, lambda m: None)
        intra, daily = adjust_corporate_actions(intra, daily, lambda m: None)
        raw = stock_raw(symbol, intra, daily, market_data(smart, now, log))
    hmm = hmm or HMMLabeler.from_json((out_dir / 'hmm.json').read_text())
    X = model_frame(raw, hmm, live=True, asset=symbol, base=base)
    X = X[X['p_hmm_Chaotic'].notna()]
    dates = X.index.normalize().unique()
    first = X.index.get_loc(X.index[X.index.normalize() >= dates[-SHOWN_SESSIONS:][0]][0])
    ctx = X.iloc[max(0, first - CONTEXT):]
    shown = X.index[first:]
    frames = {'close': raw[f'{symbol}_15m']['close'].reindex(shown)}
    frames['hmm_filter'] = pd.Series(X.loc[shown, 'p_hmm_Chaotic'])
    calls = {'hmm_filter': ctx.loc[shown, ['p_hmm_Quiet', 'p_hmm_Normal', 'p_hmm_Chaotic']].to_numpy().argmax(axis=1)}
    for key in meta['models']:
        frozen = joblib.load(out_dir / f'{key}.joblib')
        p = frozen['model'].predict_proba(ctx[frozen['features']])[-len(shown):]
        frames[key] = pd.Series(p[:, 2], index=shown)
        calls[key] = classify(p, frozen['tau'])
    current = pd.DataFrame(frames).join(pd.DataFrame({f'regime_{k}': v for k, v in calls.items()}, index=shown))
    # Hindsight labels (the training target) for comparison; the latest bars are provisional (Viterbi uses later bars)
    current['truth'] = hmm.hindsight(X).reindex(shown)
    current.to_parquet(out_dir / 'current.parquet')
    last = current.iloc[-1]
    known = current['truth'].notna()
    log(f'  as of bar close {(shown[-1] + BAR):%a %d %b %H:%M} IST, {symbol} {last["close"]:,.2f} '
        f'(match = agreement with the hindsight HMM labels over the last {len(current.index.normalize().unique())} sessions):')
    for key in ['hmm_filter', *meta['models']]:
        name = MODELS[key][0] if key in MODELS else 'HMM filter (benchmark)'
        match = (current.loc[known, f'regime_{key}'] == current.loc[known, 'truth']).mean()
        log(f'    {name:<22} {STATE_NAMES[int(last[f"regime_{key}"])]:<8} P(Chaotic) {last[key]:>4.0%}   match {match:.0%}')
    if known.iloc[-1]:
        log(f'    {"HMM smoothing (truth)":<22} {STATE_NAMES[int(last["truth"])]:<8} provisional: can change as new bars arrive')
    return current


def trained_stocks():
    """Symbols with a finished training run, newest first."""
    metas = [json.loads(p.read_text()) for p in MODELS_DIR.glob('*/meta.json')]
    return sorted(metas, key=lambda m: m['trained_at'], reverse=True)


def stock_frames(symbol):
    out_dir = MODELS_DIR / safe(symbol)
    meta = json.loads((out_dir / 'meta.json').read_text())
    current = pd.read_parquet(out_dir / 'current.parquet') if (out_dir / 'current.parquet').exists() else None
    return meta, current

