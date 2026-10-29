"""Evaluation of real-time regime classifiers against hindsight labels.

Bar-level scores (macro-F1, Chaotic precision / recall) say how often each 15-minute bar is right.
Episode-level scores say what a trader experiences from an alarm system:
  - episode recall: share of true Chaotic episodes in which the model raises Chaotic at least once
  - detection delay: bars from the start of a true episode to the first Chaotic call inside it
  - false alarms / week: predicted Chaotic runs that touch no true Chaotic bar
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

CHAOTIC = 2


def runs(mask):
    """(start, end) positions of contiguous True runs; end is exclusive."""
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def regime_metrics(y_true, y_pred, state=CHAOTIC):
    y_true, y_pred = y_true.align(y_pred, join='inner')
    t, p = y_true.to_numpy().astype(int), y_pred.to_numpy().astype(int)
    true_eps, pred_eps = runs(t == state), runs(p == state)

    delays = []
    for s, e in true_eps:
        hits = np.flatnonzero(p[s:e] == state)
        if len(hits):
            delays.append(hits[0])
    false_alarms = sum(1 for s, e in pred_eps if not (t[s:e] == state).any())
    weeks = len(np.unique(y_true.index.date)) / 5

    return {
        'macro_f1': f1_score(t, p, average='macro'),
        'chaotic_precision': precision_score(t, p, labels=[state], average='macro', zero_division=0),
        'chaotic_recall': recall_score(t, p, labels=[state], average='macro', zero_division=0),
        'episode_recall': len(delays) / max(len(true_eps), 1),
        'median_delay_bars': float(np.median(delays)) if delays else np.nan,
        'false_alarms_per_week': false_alarms / weeks,
    }


def compare(y_true, preds, by_year=False):
    """One row of `regime_metrics` per model (optionally per model and test year)."""
    rows = []
    for name, p in preds.items():
        if by_year:
            for year, py in p.groupby(p.index.year):
                rows.append({'model': name, 'year': year, **regime_metrics(y_true, py)})
        else:
            rows.append({'model': name, **regime_metrics(y_true, p)})
    keys = ['model', 'year'] if by_year else ['model']
    return pd.DataFrame(rows).set_index(keys).round(3)
