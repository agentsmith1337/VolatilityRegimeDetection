"""Train and freeze the production regime model for the live app.

Model: "LightGBM + HMM" from Stage 03 (best walk-forward macro-F1), now trained on all labelled bars
from 2017. Target: hindsight labels of the HMM in models/hmm_labeler.json (fitted on 2015-2021 in Stage 02).
The Chaotic threshold tau is tuned on the last year of the training data, as in Stage 03.

Writes models/live_lgbm.txt (LightGBM booster) and models/live_model.json (features, tau, metadata).

Usage:
    python scripts/train_live_model.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vrdet.classifiers import TunedChaoticThreshold, lightgbm  # noqa: E402
from vrdet.data import load_raw  # noqa: E402
from vrdet.labels import HMMLabeler  # noqa: E402
from vrdet.live import MODEL_DIR, model_frame  # noqa: E402

TRAIN_START = '2017-01-01'  # valid India VIX intraday data from Sep 2016


def main():
    hmm = HMMLabeler.from_json((MODEL_DIR / 'hmm_labeler.json').read_text())
    X = model_frame(load_raw(), hmm)
    y = hmm.hindsight(X)
    y = y[y.index >= pd.Timestamp(TRAIN_START, tz=y.index.tz)]
    X = X.loc[y.index]
    features = list(X.columns)
    print(f'{len(X):,} bars ({X.index[0]} -> {X.index[-1]}), {len(features)} features, '
          f'Chaotic share {(y == 2).mean():.3f}')

    m = TunedChaoticThreshold(lightgbm).fit(X, y)
    m.model_.booster_.save_model(str(MODEL_DIR / 'live_lgbm.txt'))
    meta = {'model': 'LightGBM + HMM', 'features': features, 'tau': m.tau_,
            'train_start': str(X.index[0]), 'train_end': str(X.index[-1]), 'n_bars': len(X),
            'chaotic_share': round(float((y == 2).mean()), 4),
            'trained_at': pd.Timestamp.now(tz='Asia/Kolkata').isoformat(timespec='seconds')}
    results = ROOT / 'data' / 'processed' / 'results_overall.csv'
    if results.exists():  # out-of-sample scores from the Stage 03 walk-forward test, for the dashboard
        r = pd.read_csv(results, index_col='model')
        meta['walk_forward'] = {k: r.loc[k].to_dict() for k in ('HMM filter', 'LightGBM + HMM') if k in r.index}
    (MODEL_DIR / 'live_model.json').write_text(json.dumps(meta, indent=1))
    print(f'tau = {m.tau_}; saved {MODEL_DIR / "live_lgbm.txt"} and live_model.json')


if __name__ == '__main__':
    main()
