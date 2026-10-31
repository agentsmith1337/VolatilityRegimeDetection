"""Model Lab page: every model in the zoo, each scored on demand on the latest live (or replay) data.

Pressing Play queues a job; one background worker runs jobs strictly one at a time, so several Play
presses (or several browser tabs) never load the CPU with simultaneous model runs. Results are shared
by all viewers and kept until the runner publishes a newer bar.
"""
import threading
import time

import pandas as pd
import streamlit as st

from ui import COLORS, DATA_DIR, TRUTH, TRUTH_NOTE, last_sessions, price_chart, read_json, read_parquet, show, strip_chart, worker
from vrdet.labels import STATE_NAMES
from vrdet.live import BAR, MODEL_DIR, classify

ZOO_DIR = MODEL_DIR / 'zoo'
PRODUCTION = 'lightgbm'
BENCHMARK = 'hmm_filter'
HMM_COLS = ['p_hmm_Quiet', 'p_hmm_Normal', 'p_hmm_Chaotic']


class JobQueue:
    """Model-scoring jobs on the shared single worker (ui.worker): they run one after another, never in parallel."""

    def __init__(self):
        self.pool = worker()
        self.jobs, self.lock, self.models = {}, threading.Lock(), {}

    def submit(self, key, version, features):
        with self.lock:
            job = self.jobs.get(key)
            if job and job['version'] == version and job['status'] in ('queued', 'running', 'done'):
                return
            job = {'version': version, 'status': 'queued', 'queued_at': time.time()}
            self.jobs[key] = job
        self.pool.submit(self._run, key, job, features)

    def _load(self, key):
        import joblib
        mtime = (ZOO_DIR / f'{key}.joblib').stat().st_mtime
        if self.models.get(key, (None, None))[0] != mtime:
            self.models = {key: (mtime, joblib.load(ZOO_DIR / f'{key}.joblib'))}  # keep one model in memory
        return self.models[key][1]

    def _run(self, key, job, features):
        job['status'], t0 = 'running', time.perf_counter()
        try:
            if key == BENCHMARK:
                p = features[HMM_COLS].to_numpy()
            else:
                frozen = self._load(key)
                p = frozen['model'].predict_proba(features[frozen['features']])
            job['proba'] = pd.DataFrame(p, index=features.index, columns=list(STATE_NAMES))
            job['status'] = 'done'
        except Exception as e:  # shown on the model's card
            job['status'], job['error'] = 'error', f'{type(e).__name__}: {e}'
        job['seconds'] = time.perf_counter() - t0

    def pending(self):
        return [k for k, j in self.jobs.items() if j['status'] in ('queued', 'running')]


@st.cache_resource
def job_queue():
    return JobQueue()


def catalogue():
    """Model metadata from models/zoo/*.json: production first, benchmark last, the rest by walk-forward macro-F1."""
    metas = {p.stem: read_json(p) for p in ZOO_DIR.glob('*.json')}
    metas = {k: m for k, m in metas.items() if m and (k == BENCHMARK or (ZOO_DIR / f'{k}.joblib').exists())}
    rest = sorted((k for k in metas if k not in (PRODUCTION, BENCHMARK)),
                  key=lambda k: -metas[k]['walk_forward']['macro_f1'])
    order = [k for k in [PRODUCTION, *rest, BENCHMARK] if k in metas]
    return {k: metas[k] for k in order}


def leaderboard(metas):
    rows = []
    for k, m in metas.items():
        w = m['walk_forward']
        rows.append({'Model': m['name'] + (' ★' if k == PRODUCTION else ''), 'Family': m['family'],
                     'Macro-F1': w['macro_f1'], 'Chaotic precision': w['chaotic_precision'],
                     'Chaotic recall': w['chaotic_recall'], 'Episodes caught': w['episode_recall'],
                     'Delay (min)': w['median_delay_bars'] * 15, 'False alarms / wk': w['false_alarms_per_week']})
    df = pd.DataFrame(rows)
    best = {c: (df[c].min() if c in ('Delay (min)', 'False alarms / wk') else df[c].max())
            for c in df.columns[2:]}
    st.dataframe(
        df.style.format({c: '{:.3f}' for c in df.columns[2:]} | {'Delay (min)': '{:.0f}', 'False alarms / wk': '{:.2f}'})
          .apply(lambda col: ['font-weight:700' if col.name in best and v == best[col.name] else '' for v in col]),
        hide_index=True, width='stretch', height=35 * (len(df) + 1) + 3)
    st.caption('Walk-forward test 2022–2026: each year predicted by models trained only on earlier years; τ tuned on '
               'the last training year. Bold = best in column. ★ = production model behind the Risk Radar.')


def badge(regime, p_chaotic):
    c = COLORS[regime]
    return (f'<span style="display:inline-block;padding:3px 12px;border-radius:999px;background:{c};color:white;'
            f'font-weight:700;">{regime}</span> <span style="opacity:0.8;">P(Chaotic) {p_chaotic:.0%}</span>')


def model_card(key, meta, job, version, features, hist, n_sessions, prod_regime):
    q = job_queue()
    with st.container(border=True):
        head, btn = st.columns([5, 1], vertical_alignment='center')
        w = meta['walk_forward']
        head.markdown(f"**{meta['name']}**{' ★' if key == PRODUCTION else ''} · *{meta['family']}*  \n"
                      f"{meta['description']}  \n"
                      f"<small>Test macro-F1 **{w['macro_f1']:.3f}** · Chaotic recall {w['chaotic_recall']:.2f} · "
                      f"false alarms/wk {w['false_alarms_per_week']:.2f}"
                      + (f" · τ {meta['tau']:.2f}" if 'tau' in meta else '') + '</small>', unsafe_allow_html=True)
        status = job['status'] if job and job['version'] == version else None
        label = {'queued': 'Queued…', 'running': 'Running…'}.get(status, 'Run again' if status == 'done' else 'Play')
        if btn.button(label, key=f'play_{key}', icon=':material/play_arrow:', width='stretch',
                      disabled=status in ('queued', 'running'), type='primary' if status is None else 'secondary'):
            q.submit(key, version, features)
            st.rerun()

        if job and job['status'] == 'error':
            st.error(job['error'])
        elif job and job['status'] == 'done':
            tau = meta.get('tau', 0.5) if key != BENCHMARK else None
            p = job['proba'].reindex(hist.index).dropna()
            regimes = p.to_numpy().argmax(axis=1) if tau is None else classify(p, tau)
            shown = hist.loc[p.index, ['close']].assign(regime=regimes, Chaotic=p['Chaotic'])
            last = shown.iloc[-1]
            agree = (regimes == prod_regime.reindex(p.index).to_numpy()).mean()
            truth = hist['truth'].reindex(p.index) if 'truth' in hist else None
            match = '' if truth is None else                 f' · matches the hindsight truth on {(regimes[truth.notna()] == truth.dropna().to_numpy()).mean():.0%}'
            stale = job['version'] != version
            st.markdown(badge(STATE_NAMES[int(last['regime'])], last['Chaotic'])
                        + f" &nbsp; <small>as of bar close {(p.index[-1] + BAR):%a %d %b %H:%M} · "
                          f"{len(p)} bars scored in {job['seconds']:.1f}s · agrees with production on {agree:.0%} of bars"
                          f"{match}"
                        + (' · <b>newer data available, press Run again</b>' if stale else '') + '</small>',
                        unsafe_allow_html=True)
            show(price_chart(last_sessions(shown, n_sessions), tau if tau is not None else 0.5,
                             {meta['name']: 'Chaotic'}, height=260, prob_height=150))
            return pd.Series(regimes, index=p.index)
        elif status == 'queued':
            ahead = [k for k in q.pending() if k != key and q.jobs[k]['queued_at'] < job['queued_at']]
            st.caption(f'Waiting for {len(ahead)} model(s) ahead in the queue.')
        elif status == 'running':
            st.caption('Scoring the latest bars…')
    return None


def render():
    st.title('Model Lab')
    st.caption('Every trained model, scored on demand on the same live inputs as the Risk Radar. Press **Play** on a '
               'model to run it; runs are queued and execute one at a time.')
    metas = catalogue()
    if len(metas) <= 1:
        st.info('No trained models yet. Train them with `python scripts/train_zoo.py` (takes about an hour).')
        return
    features = read_parquet(DATA_DIR / 'features.parquet')
    hist = read_parquet(DATA_DIR / 'history.parquet')
    if features is None or hist is None:
        st.info('No live data yet. Start the runner: `python app/runner.py` (or `--replay 2025-04-07`).')
        return
    version = str(features.index[-1])
    n_sessions = st.session_state.get('n_sessions', 3)
    prod_tau = read_json(MODEL_DIR / 'live_model.json', {}).get('tau', 0.3)
    prod_regime = pd.Series(classify(hist[list(STATE_NAMES)], prod_tau), index=hist.index)
    q = job_queue()

    with st.expander('Leaderboard (walk-forward test 2022–2026)', expanded=True):
        leaderboard(metas)

    a, b, c = st.columns([1.3, 1, 3], vertical_alignment='center')
    if a.button('Play all, one at a time', icon=':material/playlist_play:', type='primary', width='stretch'):
        for k in metas:
            q.submit(k, version, features)
        st.rerun()
    if b.button('Clear results', icon=':material/delete:', width='stretch', disabled=bool(q.pending())):
        q.jobs.clear()
        st.rerun()
    pending = q.pending()
    running = [k for k in pending if q.jobs[k]['status'] == 'running']
    c.caption(f"Data: {len(hist.index.normalize().unique())} sessions up to bar close {(features.index[-1] + BAR):%a %d %b %H:%M}"
              + (f" · running **{metas[running[0]]['name']}**, {len(pending) - 1} queued" if running else ''))

    compare = st.container()
    results = {}
    for k, m in metas.items():
        r = model_card(k, m, q.jobs.get(k), version, features, hist, n_sessions, prod_regime)
        if r is not None:
            results[m['name']] = r
    has_truth = 'truth' in hist
    if len(results) >= (1 if has_truth else 2):
        with compare:
            st.subheader('Side by side')
            regimes = pd.DataFrame(results)
            if has_truth:
                regimes.insert(0, TRUTH, hist['truth'].reindex(regimes.index))
            regimes = last_sessions(regimes, n_sessions)
            show(strip_chart(regimes, hist.loc[regimes.index, 'close']))
            st.caption('Each row is one model’s call for every bar (green Quiet, yellow Normal, red Chaotic). '
                       + (TRUTH_NOTE if has_truth else ''))

    if pending:  # poll until the queue drains
        time.sleep(1.0)
        st.rerun()

