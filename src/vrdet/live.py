"""Live inference: the frozen production model and the data it needs at each bar close.

The production model is "LightGBM + HMM" from Stage 03: LightGBM on the causal features plus the HMM
forward-filter probabilities, calling Chaotic when P(Chaotic) >= tau. `model_frame` builds exactly the
inputs used in Stage 03, so training (scripts/train_live_model.py) and live scoring cannot drift apart.
"""
import json
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from .angel import REPO_ROOT, TOKENS, clean_intraday, request_candles, to_frame
from .features import add_lags, build_features
from .labels import STATE_NAMES, HMMLabeler

MODEL_DIR = REPO_ROOT / 'models'
LIVE_DIR = REPO_ROOT / 'data' / 'live'
LAGGED = ['abs_z', 'range_ds', 'pk_4', 'vix_chg_1']
BAR = pd.Timedelta(minutes=15)
TZ = 'Asia/Kolkata'

# Angel One series refreshed live: archive file stem -> (token name, interval)
ANGEL_SERIES = {'NIFTY_15m': ('NIFTY', 'FIFTEEN_MINUTE'), 'INDIAVIX_15m': ('INDIAVIX', 'FIFTEEN_MINUTE'),
                'NIFTY_1d': ('NIFTY', 'ONE_DAY'), 'INDIAVIX_1d': ('INDIAVIX', 'ONE_DAY')}
YF_SERIES = {'USDINR_1d': 'USDINR=X', 'BRENT_1d': 'BZ=F'}


def base_frame(raw, live=False, asset='NIFTY'):
    """Causal features and lags for every bar of `asset` (raw keys '<asset>_15m', '<asset>_1d'), no HMM inputs."""
    f = build_features(raw[f'{asset}_15m'], raw['INDIAVIX_15m'], {'usdinr': raw['USDINR_1d'], 'brent': raw['BRENT_1d']},
                       nifty_daily=raw[f'{asset}_1d'], vix_daily=raw['INDIAVIX_1d'], live=live)
    f['abs_z'] = f['z'].abs()
    return add_lags(f, LAGGED)


def model_frame(raw, hmm_lab, live=False, asset='NIFTY', base=None):
    """Classifier inputs for every bar: causal features, lags, and HMM filtered probabilities."""
    f = base_frame(raw, live, asset) if base is None else base
    return f.join(hmm_lab.filtered_proba(f).add_prefix('p_hmm_'))


def classify(proba, tau):
    """Chaotic if P(Chaotic) >= tau, otherwise the more likely of Quiet / Normal."""
    p = np.asarray(proba)
    return np.where(p[:, 2] >= tau, 2, p[:, :2].argmax(axis=1))


class RegimeModel:
    """Frozen production model: HMM labeler parameters + LightGBM booster + metadata (features, tau)."""

    def __init__(self, model_dir=MODEL_DIR):
        import lightgbm as lgb
        self.hmm = HMMLabeler.from_json((model_dir / 'hmm_labeler.json').read_text())
        self.meta = json.loads((model_dir / 'live_model.json').read_text())
        self.features, self.tau = self.meta['features'], self.meta['tau']
        self.booster = lgb.Booster(model_file=str(model_dir / 'live_lgbm.txt'))

    def predict_proba(self, X):
        """Regime probabilities, only for bars with an HMM observation (the population the model was trained on)."""
        X = X[X['p_hmm_Chaotic'].notna()]
        return pd.DataFrame(self.booster.predict(X[self.features]), index=X.index, columns=list(STATE_NAMES))


def _merge(old, new):
    return new if old is None or old.empty else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()


def _naive_dates(df):
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return df


class LiveData:
    """Research archive (data/raw) + candles fetched since, cached in data/live.

    Only completed bars are kept: a 15-minute bar stamped t is final once t + 15 min has passed. Today's
    daily candle is dropped (still forming). Cross-asset daily series come from yfinance once a day.
    """

    def __init__(self, raw, live_dir=LIVE_DIR):
        self.raw, self.dir = raw, live_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        for name in list(ANGEL_SERIES) + list(YF_SERIES):
            p = self.dir / f'{name}.parquet'
            self.cache[name] = pd.read_parquet(p) if p.exists() else None
        self.yf_date = None

    def refresh_angel(self, smart, now):
        for name, (sym, interval) in ANGEL_SERIES.items():
            last = self.combined(name).index[-1]
            last = last.tz_localize(None) if last.tzinfo is not None else last
            start = max(last - timedelta(days=3), now.tz_localize(None) - timedelta(days=150))
            rows = request_candles(smart, TOKENS[sym], interval, start.replace(hour=9, minute=0),
                                   now.tz_localize(None).replace(second=0, microsecond=0))
            time.sleep(0.4)  # SmartAPI rate limit
            if not rows:
                continue
            df = to_frame(rows)
            if name.endswith('_15m'):
                df = clean_intraday(df, verbose=False)
                df = df[df.index + BAR <= now]
            else:
                df = df[df.index.tz_convert(TZ).normalize() < now.normalize()]
            self.cache[name] = _merge(self.cache[name], df)
            self.cache[name].to_parquet(self.dir / f'{name}.parquet')

    def refresh_yf(self, now):
        """USD/INR and Brent daily closes, at most once per day (the model only uses previous days)."""
        if self.yf_date == now.date():
            return
        import warnings
        import yfinance as yf
        warnings.filterwarnings('ignore', module='yfinance')  # pandas deprecation noise inside yfinance
        for name, ticker in YF_SERIES.items():
            try:
                df = yf.download(ticker, period='3mo', interval='1d', auto_adjust=False, progress=False)
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
                df = _naive_dates(df)
                df = df[df.index < now.tz_localize(None).normalize()]
                if not df.empty:
                    self.cache[name] = _merge(self.cache[name], df)
                    self.cache[name].to_parquet(self.dir / f'{name}.parquet')
            except Exception as e:  # stale cross-asset data is tolerable; it only enters as previous-day values
                print(f'yfinance {ticker} failed: {e}')
        self.yf_date = now.date()

    def combined(self, name):
        base, extra = self.raw[name], self.cache.get(name)
        if extra is not None and name.endswith('_1d'):
            extra = _naive_dates(extra.copy())
        return _merge(base, extra) if extra is not None else base

    def frames(self):
        return {name: self.combined(name) for name in list(ANGEL_SERIES) + list(YF_SERIES)}
