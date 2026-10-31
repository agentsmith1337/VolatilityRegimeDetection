"""Stock Lab page: train the regime models on any NSE stock (Angel One data) and see their current calls.

Train runs the whole pipeline (data -> labels -> train/test -> current regime) as one job on the shared
single worker, streaming its log into a console panel (and the terminal running Streamlit).
"""
import json
import time

import pandas as pd
import streamlit as st

from ui import COLORS, TRUTH, TRUTH_NOTE, show, strip_chart, price_chart, worker
from vrdet.labels import STATE_NAMES
from vrdet.live import BAR
from vrdet.stocks import MODELS_DIR, instruments, safe, score_latest, stock_frames, train_stock, trained_stocks
from vrdet.zoo import MODELS

FAST = ['lightgbm', 'xgboost', 'logistic', 'soft_vote']
# rough seconds per model (test fit + final fit) on 8 years of 15-minute bars, for the time estimate
COST = {'lightgbm': 15, 'logistic': 8, 'xgboost': 20, 'mlp': 60, 'extra_trees': 30, 'random_forest': 60,
        'adaboost': 330, 'gru': 150, 'lstm': 150, 'transformer': 600, 'soft_vote': 75, 'stacking': 120,
        'catboost': 60, 'hist_gb': 20, 'decision_tree': 4, 'knn': 15, 'naive_bayes': 4, 'lda': 3, 'qda': 3,
        'svm': 25, 'tcn': 100, 'ordinal': 10}
NAMES = {k: v[0] for k, v in MODELS.items()} | {'hmm_filter': 'HMM filter (benchmark)'}


@st.cache_resource
def jobs():
    return {}   # symbol -> job; shared by all viewers


@st.cache_data(ttl=86400, show_spinner='Loading the NSE instrument list from Angel One…')
def stock_list():
    return instruments()['name'].tolist()


def submit(kind, symbol, **kw):
    job = {'kind': kind, 'status': 'queued', 'log': [], 'queued_at': time.time()}
    jobs()[symbol] = job
    log_file = MODELS_DIR / safe(symbol) / 'console.log'

    def log(msg):
        job['log'].append(msg)
        print(msg, flush=True)   # results also in the terminal running Streamlit

    def run():
        job['status'] = 'running'
        try:
            if kind == 'train':
                train_stock(symbol, log=log, **kw)
            else:
                log(f'=== {symbol}: update regime with the trained models ===')
                score_latest(symbol, log=log)
            job['status'] = 'done'
        except Exception as e:
            log(f'ERROR: {type(e).__name__}: {e}')
            job['status'] = 'error'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as fh:
            fh.write('\n'.join(job['log']) + '\n\n')
    worker().submit(run)


@st.fragment(run_every='1.5s')
def console(symbol):
    """Live log of the symbol's job. Only this fragment re-runs while polling (charts are not redrawn); the whole
    page refreshes once when the job finishes."""
    job = jobs()[symbol]
    state = {'queued': 'waiting for the worker (another job is running)', 'running': 'running',
             'done': 'finished', 'error': 'failed'}[job['status']]
    st.markdown(f"**Console** · {job['kind']} · {state}")
    st.code('\n'.join(job['log'][-60:]) or 'Starting…', language=None, wrap_lines=True)
    key = f'console_status_{symbol}'
    if st.session_state.get(key) in ('queued', 'running') and job['status'] in ('done', 'error'):
        st.session_state[key] = job['status']
        st.rerun(scope='app')
    st.session_state[key] = job['status']


def badge(name, regime, detail, dashed=False):
    c = COLORS[regime]
    return (f'<div style="border:1px {"dashed" if dashed else "solid"} {c};border-radius:10px;padding:8px 10px;'
            f'background:{c}1f;">'
            f'<div style="font-size:0.8rem;opacity:0.8;">{name}</div>'
            f'<div style="font-weight:700;color:{c};font-size:1.15rem;">{regime}</div>'
            f'<div style="font-size:0.8rem;opacity:0.8;">{detail}</div></div>')


def results_view(symbol, n_sessions):
    try:
        meta, current = stock_frames(symbol)
    except FileNotFoundError:
        return
    st.subheader(f'{symbol}: test results')
    res = pd.DataFrame(meta['results']).T
    res.index = [NAMES[k] for k in res.index]
    cols = {'macro_f1': 'Macro-F1', 'chaotic_precision': 'Chaotic precision', 'chaotic_recall': 'Chaotic recall',
            'pr_auc_chaotic': 'PR-AUC', 'episode_recall': 'Episodes caught', 'median_delay_bars': 'Delay (bars)',
            'false_alarms_per_week': 'False alarms / wk'}
    st.dataframe(res[list(cols)].rename(columns=cols).astype(float).style.format('{:.3f}')
                 .format('{:.2f}', subset=['False alarms / wk']).format('{:.1f}', subset=['Delay (bars)']),
                 width='stretch')
    eps = meta['chaotic_episodes']
    sh = meta['label_shares']
    st.caption(f"Test period: {pd.Timestamp(meta['test_start']):%d %b %Y} onwards (never used for training or tuning), "
               f"{eps['test']} Chaotic episodes ({eps['train']} in training). Labels: Quiet {sh['Quiet']:.0%} · "
               f"Normal {sh['Normal']:.0%} · Chaotic {sh['Chaotic']:.0%}. {meta['sessions_valid']:,}/"
               f"{meta['sessions_total']:,} sessions valid. Trained {pd.Timestamp(meta['trained_at']):%d %b %H:%M}."
               + (' Few test episodes: scores are noisy.' if eps['test'] < 20 else ''))

    if current is None:
        return
    keys = ['hmm_filter', *meta['models']]
    last, ts = current.iloc[-1], current.index[-1]
    st.subheader(f'{symbol}: current regime')
    st.caption(f"As of bar close {(ts + BAR):%a %d %b %H:%M} IST · {symbol} {last['close']:,.2f}. "
               'Press Update regime to fetch new candles and rescore without retraining.')
    truth = current['truth'] if 'truth' in current else None   # runs before the truth column existed
    known = truth.notna() if truth is not None else None
    cards = [badge(NAMES[k], STATE_NAMES[int(last[f'regime_{k}'])], f'P(Chaotic) {last[k]:.0%}'
                   + (f" · match {(current.loc[known, f'regime_{k}'] == truth[known]).mean():.0%}" if known is not None
                      else '')) for k in keys]
    if truth is not None and known.iloc[-1]:
        cards.insert(0, badge(TRUTH, STATE_NAMES[int(truth.iloc[-1])], 'hindsight, provisional', dashed=True))
    for row in range(0, len(cards), 6):
        for cell, card in zip(st.columns(6), cards[row:row + 6]):
            cell.markdown(card, unsafe_allow_html=True)
    if truth is not None:
        st.caption(f'match = share of bars over the last {current.index.normalize().nunique()} sessions where the model '
                   'agrees with the hindsight labels (HMM smoothing). Those labels use later bars, so they are not '
                   'available live, and the latest bars are provisional.')

    shown = current[current.index.normalize() >= current.index.normalize().unique()[-n_sessions:][0]]
    regimes = pd.DataFrame(({TRUTH: shown['truth']} if truth is not None else {})
                           | {NAMES[k]: shown[f'regime_{k}'] for k in keys})
    show(strip_chart(regimes, shown['close'], label=symbol))
    if truth is not None:
        st.caption(TRUTH_NOTE)
    pick = st.selectbox('Chart one model', meta['models'] + ['hmm_filter'], format_func=NAMES.get, key=f'pick_{symbol}')
    tau = json.loads((MODELS_DIR / safe(symbol) / 'meta.json').read_text())['results'].get(pick, {}).get('tau', 0.5)
    h = shown[['close']].assign(regime=shown[f'regime_{pick}'], Chaotic=shown[pick])
    show(price_chart(h, tau, {NAMES[pick]: 'Chaotic'}, height=300, prob_height=160, label=symbol))


def render():
    st.title('Stock Lab')
    st.caption('Train the regime models on any NSE stock: candles from Angel One SmartAPI, labels from an HMM fitted on '
               'the training years only, test on the most recent months, then the current regime from each model. '
               'Jobs share one background worker with the Model Lab, so they never run in parallel.')
    try:
        names = stock_list()
    except Exception as e:
        st.error(f'Could not load the Angel One instrument list: {e}')
        return
    done = [m['symbol'] for m in trained_stocks()]

    with st.container(border=True):
        a, b = st.columns([1, 2])
        default = done[0] if done else 'RELIANCE'
        symbol = a.selectbox('Stock (NSE)', names, index=names.index(default) if default in names else 0,
                             help='Type to search ~2,700 NSE equities.')
        models = b.multiselect('Models', list(MODELS), default=FAST, format_func=NAMES.get,
                               help='The HMM filter benchmark is always evaluated as well.')
        c, d, e = st.columns([1, 1, 2])
        years = c.slider('Years of history', 3, 9, 8, help='Angel One 15-minute stock history starts in 2017.')
        test_months = d.select_slider('Test period (months)', [6, 12, 18, 24], 12)
        est = 30 + sum(COST[k] for k in models) * years / 8
        e.caption(f'Estimated time: about {est / 60:.0f} min' if est > 90 else f'Estimated time: about {est:.0f} s')
        e.caption('Sequence models (GRU, LSTM, Transformer, temporal CNN) and AdaBoost are slow; the defaults take about 2 minutes.')
        job = jobs().get(symbol)
        busy = job is not None and job['status'] in ('queued', 'running')
        trained = (MODELS_DIR / safe(symbol) / 'meta.json').exists()
        x, y_, _ = st.columns([1, 1, 3])
        if x.button('Train & test', icon=':material/model_training:', type='primary', width='stretch',
                    disabled=busy or not models):
            submit('train', symbol, models=models, years=years, test_months=test_months)
            st.rerun()
        if y_.button('Update regime', icon=':material/refresh:', width='stretch', disabled=busy or not trained):
            submit('update', symbol)
            st.rerun()

    if job:
        console(symbol)
    elif trained and (MODELS_DIR / safe(symbol) / 'console.log').exists():
        with st.expander('Console log of the last run'):
            st.code((MODELS_DIR / safe(symbol) / 'console.log').read_text(encoding='utf-8')[-20000:], language=None)

    results_view(symbol, st.session_state.get('n_sessions', 3))

    if done:
        with st.expander(f'Trained stocks ({len(done)})'):
            st.dataframe(pd.DataFrame([{'Stock': m['symbol'], 'Trained': pd.Timestamp(m['trained_at']).strftime('%d %b %H:%M'),
                                        'Models': ', '.join(NAMES[k] for k in m['models']),
                                        'Best macro-F1': max(r['macro_f1'] for r in m['results'].values())}
                                       for m in trained_stocks()]), hide_index=True, width='stretch')

