"""Shared helpers for the dashboard pages: paths, file reading, regime colours and charts."""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from vrdet.labels import STATE_NAMES  # noqa: E402
from vrdet.live import LIVE_DIR  # noqa: E402

REPLAY = '--replay' in sys.argv[1:]
DATA_DIR = ROOT / 'data' / 'replay' if REPLAY else LIVE_DIR
COLORS = {'Quiet': '#2e9e5b', 'Normal': '#d9a400', 'Chaotic': '#d93636'}
LINE = '#7d8a99'    # price line: readable on both the light and the dark theme
Y_WIDTH = 64        # fixed y-axis width, so charts stacked as separate elements line up bar for bar
TRUTH = 'HMM smoothing (truth)'   # hindsight labels: Viterbi over all bars + minimum duration (the training target)
TRUTH_NOTE = ('Top row: the hindsight labels the models were trained to predict (HMM smoothing, which also uses '
              'later bars). They are not available in real time, and the latest bars can still change as new '
              'bars arrive.')


def worker():
    """The one background worker shared by every page: jobs (model scoring, stock training) run strictly one at
    a time, however many buttons are pressed or browser tabs are open."""
    import streamlit as st

    @st.cache_resource
    def _pool():
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix='vrdet-job')
    return _pool()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def read_parquet(path):
    """None if missing or being rewritten by the runner at this instant."""
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


def ago(ts, now):
    s = int((now - pd.Timestamp(ts)).total_seconds())
    if s < 90:
        return f'{s}s ago'
    if s < 5400:
        return f'{s // 60} min ago'
    if s < 172800:
        return f'{s // 3600} h ago'
    return f'{s // 86400} days ago'


def last_sessions(df, n):
    dates = df.index.normalize().unique()
    return df[df.index.normalize() >= dates[-n:][0]]


def _bar_axis(h):
    """Bars on an evenly spaced axis (no overnight gaps), labelled at each session start."""
    starts = h.groupby(h['ts'].dt.date)['i'].min()
    label = ' : '.join(f"datum.value == {i} ? '{d:%a %d %b}'" for d, i in starts.items()) + " : ''"

    def x(field='i'):
        return alt.X(f'{field}:Q', title=None, scale=alt.Scale(domain=[-0.5, len(h) - 0.5], nice=False),
                     axis=alt.Axis(values=list(starts.values), labelExpr=label, grid=False, labelAngle=0))
    return x, starts


def _frame(h):
    h = h.reset_index(names='ts')
    h['i'] = range(len(h))
    h['x0'], h['x1'] = h['i'] - 0.5, h['i'] + 0.5
    h['time'] = h['ts'].dt.strftime('%a %d %b %H:%M')
    return h


def _y(field, title, **scale):
    return alt.Y(field, title=title, scale=alt.Scale(**scale),
                 axis=alt.Axis(minExtent=Y_WIDTH, maxExtent=Y_WIDTH, labelLimit=Y_WIDTH - 20))


def show(charts):
    """Render charts as separate elements: Altair concatenations do not shrink to the container width."""
    import streamlit as st
    for c in charts:
        st.altair_chart(c, width='stretch')


def price_chart(h, tau, probs, height=380, prob_height=200, label='NIFTY 50'):
    """[NIFTY close shaded by h['regime'], P(Chaotic) lines]. probs: {legend label: column}, first = main."""
    h = _frame(h)
    x, starts = _bar_axis(h)
    h['Regime'] = [STATE_NAMES[r] for r in h['regime']]
    main = next(iter(probs.values()))
    h['P(Chaotic)'] = h[main].round(3)
    scale = alt.Scale(domain=list(STATE_NAMES), range=[COLORS[k] for k in STATE_NAMES])
    tips = ['time:N', 'close:Q', 'Regime:N', 'P(Chaotic):Q']
    bands = alt.Chart(h).mark_rect(opacity=0.28).encode(
        x=x('x0'), x2='x1:Q', color=alt.Color('Regime:N', scale=scale, legend=alt.Legend(orient='top', title=None)),
        tooltip=tips)
    line = alt.Chart(h).mark_line(color=LINE, strokeWidth=1.8).encode(
        x=x(), y=_y('close:Q', label, zero=False), tooltip=tips)
    rules = alt.Chart(pd.DataFrame({'i': starts.values - 0.5})).mark_rule(color='#888', strokeDash=[3, 3]).encode(x=x())
    price = (bands + rules + line).properties(height=height)

    labels = list(probs)
    hp = h.melt(id_vars=['i', 'time'], value_vars=list(probs.values()), var_name='series', value_name='p')
    hp['series'] = hp['series'].map({v: k for k, v in probs.items()})
    prob = alt.Chart(hp).mark_line(strokeWidth=1.4).encode(
        x=x(), y=_y('p:Q', 'P(Chaotic)', domain=[0, 1]),
        color=alt.Color('series:N', scale=alt.Scale(domain=labels, range=['#d93636', '#8a8a8a'][:len(labels)]),
                        legend=alt.Legend(orient='top', title=None)),
        strokeDash=alt.StrokeDash('series:N', scale=alt.Scale(domain=labels, range=[[1, 0], [4, 3]][:len(labels)]),
                                  legend=None),
        tooltip=['time:N', 'series:N', alt.Tooltip('p:Q', format='.2f')])
    thr = alt.Chart(pd.DataFrame({'tau': [tau]})).mark_rule(color='#d93636', strokeDash=[6, 4], opacity=0.6).encode(y='tau:Q')
    return [price, (prob + thr).properties(height=prob_height)]


def strip_chart(regimes, close, label='NIFTY 50'):
    """One row of regime colours per model (regimes: DataFrame bars x models), NIFTY close on top."""
    h = _frame(close.to_frame('close'))
    x, starts = _bar_axis(h)
    long = regimes.reset_index(drop=True).assign(i=range(len(regimes))).melt('i', var_name='model', value_name='r')
    long = long.dropna()
    long['Regime'] = [STATE_NAMES[int(r)] for r in long['r']]
    long['time'] = h['time'].to_numpy()[long['i'].to_numpy()]
    long['x0'], long['x1'] = long['i'] - 0.5, long['i'] + 0.5
    scale = alt.Scale(domain=list(STATE_NAMES), range=[COLORS[k] for k in STATE_NAMES])
    names = list(regimes.columns)
    width = max(Y_WIDTH, 7 * max(len(n) for n in names) + 12)
    line = alt.Chart(h).mark_line(color=LINE, strokeWidth=1.6).encode(
        x=x(), y=alt.Y('close:Q', title=label, scale=alt.Scale(zero=False),
                       axis=alt.Axis(minExtent=width, maxExtent=width)),
        tooltip=['time:N', 'close:Q']).properties(height=200)
    strips = alt.Chart(long).mark_rect().encode(
        x=x('x0'), x2='x1:Q', y=alt.Y('model:N', title=None, sort=names,
                                      axis=alt.Axis(minExtent=width, maxExtent=width, labelLimit=width,
                                                    labelOverlap=False)),
        color=alt.Color('Regime:N', scale=scale, legend=alt.Legend(orient='top', title=None)),
        tooltip=['model:N', 'time:N', 'Regime:N']).properties(height=alt.Step(24))  # fixed row height
    return [line, strips]
