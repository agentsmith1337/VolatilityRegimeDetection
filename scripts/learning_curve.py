"""Learning curves: do the sequence models (GRU, Transformer) gain more from extra training data than small models?

If deep models were held back by too little data, their test scores would still be rising with training size
while the small models flatten out. Fixed test period: 2025-01-01 onwards, never used for training or tuning.
Each model is trained on the last N years before 2025; tau is tuned on the last 25% of that window (the
calendar-year holdout of Stage 03/04 does not exist for half-year windows). Neural networks run with several
seeds, so their run-to-run spread is visible.

Scores: PR-AUC for Chaotic (threshold-free, the cleanest comparison) and macro-F1 / episode metrics at the
tuned tau.

Outputs: data/processed/learning_curve.csv, data/processed/figures/learning_curve.png

Usage:
    python scripts/learning_curve.py
    python scripts/learning_curve.py --years 0.5 1 --models logistic gru --seeds 1    # quick check
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vrdet.data import load_raw  # noqa: E402
from vrdet.evaluate import regime_metrics  # noqa: E402
from vrdet.labels import HMMLabeler  # noqa: E402
from vrdet.live import MODEL_DIR, classify, model_frame  # noqa: E402
from vrdet.zoo import CONTEXT, MAKERS, MODELS  # noqa: E402

TEST_START = '2025-01-01'
NEURAL = {'gru', 'lstm', 'transformer', 'tcn', 'mlp'}
TAU_GRID = np.round(np.arange(0.10, 0.65, 0.05), 2)


def log(msg):
    print(f'{pd.Timestamp.now():%H:%M:%S} {msg}', flush=True)


def make(key, seed):
    m = MAKERS[key]()
    if key in NEURAL:
        if hasattr(m, 'seed'):
            m.seed = seed
        else:  # sklearn pipeline ending in MLPClassifier
            m.steps[-1][1].set_params(random_state=seed)
    return m


def size(m):
    """Trainable parameters (neural nets, logistic regression); None for tree ensembles."""
    if hasattr(m, 'net_'):
        from vrdet.seqnet import n_params
        return n_params(m.net_)
    last = m.steps[-1][1] if hasattr(m, 'steps') else m
    if hasattr(last, 'coefs_'):
        return int(sum(w.size for w in last.coefs_) + sum(b.size for b in last.intercepts_))
    if hasattr(last, 'coef_'):
        return int(last.coef_.size + last.intercept_.size)
    return None


def fit_tuned(key, seed, X, y):
    """Fit on the first 75%, pick the tau maximising Chaotic F1 on the last 25%, refit on everything."""
    from sklearn.metrics import f1_score
    cut = int(len(X) * 0.75)
    p = make(key, seed).fit(X.iloc[:cut], y[:cut]).predict_proba(X.iloc[max(0, cut - CONTEXT):])[-(len(X) - cut):, 2]
    tau = float(TAU_GRID[int(np.argmax([f1_score(y[cut:] == 2, p >= t) for t in TAU_GRID]))])
    return make(key, seed).fit(X, y), tau


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=['logistic', 'lightgbm', 'xgboost', 'gru', 'transformer'])
    parser.add_argument('--years', nargs='+', type=float, default=[0.5, 1, 2, 4, 8])
    parser.add_argument('--seeds', type=int, default=3, help='seeds per neural network')
    args = parser.parse_args()
    from sklearn.metrics import average_precision_score

    hmm = HMMLabeler.from_json((MODEL_DIR / 'hmm_labeler.json').read_text())
    X = model_frame(load_raw(), hmm)
    y = hmm.hindsight(X)
    X = X.loc[y.index]
    start = pd.Timestamp(TEST_START, tz=X.index.tz)
    test_rows = np.flatnonzero(X.index >= start)
    y_test = y.iloc[test_rows]
    ctx = X.iloc[test_rows[0] - CONTEXT:]
    log(f'test {X.index[test_rows[0]].date()} -> {X.index[-1].date()}: {len(test_rows):,} bars, '
        f'{(y_test == 2).mean():.1%} Chaotic')

    rows = []
    out = ROOT / 'data' / 'processed' / 'learning_curve.csv'
    for years in args.years:
        train = (X.index < start) & (X.index >= start - pd.DateOffset(months=int(years * 12)))
        Xtr, ytr = X[train], y[train].to_numpy()
        episodes = int((np.diff(np.r_[0, (ytr == 2).astype(int)]) == 1).sum())
        for key in args.models:
            for seed in range(args.seeds if key in NEURAL else 1):
                t0 = time.perf_counter()
                m, tau = fit_tuned(key, seed, Xtr, ytr)
                p = m.predict_proba(ctx)[-len(test_rows):]
                pred = pd.Series(classify(p, tau), index=y_test.index)
                r = regime_metrics(y_test, pred)
                rows.append({'model': MODELS[key][0], 'key': key, 'train_years': years, 'train_bars': len(Xtr),
                             'train_chaotic_episodes': episodes, 'seed': seed, 'params': size(m), 'tau': tau,
                             'pr_auc_chaotic': average_precision_score(y_test == 2, p[:, 2]), **r,
                             'seconds': time.perf_counter() - t0})
                log(f'{years:>4} y | {MODELS[key][0]:<20} seed {seed}: PR-AUC {rows[-1]["pr_auc_chaotic"]:.3f}, '
                    f'macro-F1 {r["macro_f1"]:.3f} ({rows[-1]["seconds"]:.0f}s)')
                pd.DataFrame(rows).to_csv(out, index=False)  # keep partial results if interrupted

    df = pd.DataFrame(rows)
    summary = df.groupby(['model', 'train_years'])[['pr_auc_chaotic', 'macro_f1', 'chaotic_recall',
                                                    'false_alarms_per_week']].mean().round(3)
    print(summary.unstack('model')['pr_auc_chaotic'].to_string())
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = ROOT / 'data' / 'processed' / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [(axes[0], 'pr_auc_chaotic', 'PR-AUC, Chaotic (threshold-free)'),
                           (axes[1], 'macro_f1', 'Macro-F1 (tuned τ)')]:
        for name, g in df.groupby('model', sort=False):
            s = g.groupby('train_years')[col].agg(['mean', 'min', 'max'])
            ax.plot(s.index, s['mean'], 'o-', label=name)
            if (s['max'] > s['min']).any():
                ax.fill_between(s.index, s['min'], s['max'], alpha=0.15)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(df['train_years'].unique()), [f'{v:g}' for v in sorted(df['train_years'].unique())])
        ax.set_xlabel('training data (years before the 2025–26 test period)')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle('Learning curves: test on 2025–2026, train on the preceding N years (bands: spread over seeds)')
    fig.tight_layout()
    fig.savefig(figdir / 'learning_curve.png', dpi=130)
    print('saved', figdir / 'learning_curve.png')


if __name__ == '__main__':
    main()
