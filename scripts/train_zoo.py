"""Stage 04: walk-forward test and freeze every model in the zoo (one model at a time, resumable).

Same protocol as Stage 03: HMM hindsight labels as target, training from 2017, test years 2022-2026, each
test year predicted by a model trained only on earlier years, Chaotic threshold tau tuned on the last
training year. Then each model is refitted on all labelled bars and frozen for the dashboard's Model Lab.

Outputs:
    models/zoo/<key>.joblib            frozen model + tau + feature list
    models/zoo/<key>.json              description, walk-forward metrics, timings
    data/processed/zoo/<key>_wf.parquet out-of-sample probabilities and calls, 2022-2026
    data/processed/zoo_results.csv     one row per model (+ the HMM filter benchmark)

Usage:
    python scripts/train_zoo.py                       # all models not trained yet
    python scripts/train_zoo.py --models lstm gru --force
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vrdet.classifiers import TunedChaoticThreshold  # noqa: E402
from vrdet.data import load_raw  # noqa: E402
from vrdet.evaluate import regime_metrics  # noqa: E402
from vrdet.labels import STATE_NAMES, HMMLabeler  # noqa: E402
from vrdet.live import MODEL_DIR, classify, model_frame  # noqa: E402
from vrdet.zoo import CONTEXT, MAKERS, MODELS  # noqa: E402

ZOO_DIR = MODEL_DIR / 'zoo'
WF_DIR = ROOT / 'data' / 'processed' / 'zoo'
TRAIN_START = '2017-01-01'
TEST_YEARS = range(2022, 2027)
ORDER = ['lightgbm', 'logistic', 'xgboost', 'mlp', 'extra_trees', 'random_forest', 'adaboost',
         'gru', 'lstm', 'transformer', 'soft_vote', 'stacking',
         'catboost', 'hist_gb', 'decision_tree', 'knn', 'naive_bayes', 'lda', 'qda', 'svm', 'tcn', 'ordinal']


def log(msg):
    print(f'{pd.Timestamp.now():%H:%M:%S} {msg}', flush=True)


def walk_forward(key, X, y):
    parts, taus = [], {}
    for year in TEST_YEARS:
        t0 = time.perf_counter()
        train = X.index < pd.Timestamp(f'{year}-01-01', tz=X.index.tz)
        rows = np.flatnonzero(X.index.year == year)
        m = TunedChaoticThreshold(MAKERS[key]).fit(X[train], y[train])
        ctx = X.iloc[max(0, rows[0] - CONTEXT):rows[-1] + 1]          # history for sequence models
        p = pd.DataFrame(m.predict_proba(ctx)[-len(rows):], index=X.index[rows], columns=list(STATE_NAMES))
        p['pred'] = classify(p[list(STATE_NAMES)], m.tau_)
        parts.append(p)
        taus[year] = m.tau_
        log(f'  {key} {year}: tau {m.tau_:.2f}, {time.perf_counter() - t0:.0f}s')
    return pd.concat(parts), taus


def summarise():
    rows = [json.loads(p.read_text()) for p in sorted(ZOO_DIR.glob('*.json'))]
    table = pd.DataFrame([{'key': r['key'], 'model': r['name'], 'family': r['family'], **r['walk_forward'],
                           'train_seconds': r.get('final_fit_seconds')} for r in rows]).set_index('key')
    table = table.sort_values('macro_f1', ascending=False)
    table.to_csv(ROOT / 'data' / 'processed' / 'zoo_results.csv')
    print(table.drop(columns=['family']).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=ORDER, choices=ORDER)
    parser.add_argument('--force', action='store_true', help='retrain models that are already frozen')
    args = parser.parse_args()
    import joblib
    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    WF_DIR.mkdir(parents=True, exist_ok=True)

    hmm = HMMLabeler.from_json((MODEL_DIR / 'hmm_labeler.json').read_text())
    X = model_frame(load_raw(), hmm)
    y = hmm.hindsight(X)
    y = y[y.index >= pd.Timestamp(TRAIN_START, tz=y.index.tz)]
    X = X.loc[y.index]
    test = X.index.year >= min(TEST_YEARS)
    log(f'{len(X):,} bars, {X.shape[1]} inputs, test bars {test.sum():,}')

    # Benchmark: the labelling HMM's own real-time (forward-filtered) call
    hmm_live = pd.Series(X[['p_hmm_Quiet', 'p_hmm_Normal', 'p_hmm_Chaotic']].to_numpy().argmax(axis=1), index=X.index)
    (ZOO_DIR / 'hmm_filter.json').write_text(json.dumps({
        'key': 'hmm_filter', 'name': 'HMM filter (benchmark)', 'family': 'Benchmark',
        'description': 'The labelling HMM decoded forward in real time: what the HMM alone can do live.',
        'walk_forward': regime_metrics(y[test], hmm_live[test])}, indent=1))

    for key in args.models:
        if (ZOO_DIR / f'{key}.joblib').exists() and not args.force:
            log(f'{key}: already trained, skipping (use --force to retrain)')
            continue
        name, family, desc = MODELS[key]
        log(f'{name}: walk-forward {min(TEST_YEARS)}-{max(TEST_YEARS)}')
        t0 = time.perf_counter()
        wf, taus = walk_forward(key, X, y)
        wf_seconds = time.perf_counter() - t0
        wf.to_parquet(WF_DIR / f'{key}_wf.parquet')
        metrics = regime_metrics(y, wf['pred'])

        t0 = time.perf_counter()
        m = TunedChaoticThreshold(MAKERS[key]).fit(X, y)
        fit_seconds = time.perf_counter() - t0
        joblib.dump({'model': m.model_, 'tau': m.tau_, 'features': list(X.columns)}, ZOO_DIR / f'{key}.joblib')
        (ZOO_DIR / f'{key}.json').write_text(json.dumps({
            'key': key, 'name': name, 'family': family, 'description': desc, 'tau': m.tau_,
            'walk_forward': metrics, 'walk_forward_taus': taus, 'walk_forward_seconds': round(wf_seconds),
            'final_fit_seconds': round(fit_seconds), 'train_start': str(X.index[0]), 'train_end': str(X.index[-1]),
            'trained_at': pd.Timestamp.now(tz='Asia/Kolkata').isoformat(timespec='seconds')}, indent=1))
        log(f'{name}: macro-F1 {metrics["macro_f1"]:.3f}, Chaotic recall {metrics["chaotic_recall"]:.3f}, '
            f'false alarms/wk {metrics["false_alarms_per_week"]:.2f} | walk-forward {wf_seconds:.0f}s, '
            f'final fit {fit_seconds:.0f}s')
    summarise()


if __name__ == '__main__':
    main()
