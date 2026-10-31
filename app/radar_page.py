"""Risk Radar page: the production model's live regime call, chart and alerts."""
import json

import pandas as pd
import streamlit as st

from ui import (COLORS, DATA_DIR, REPLAY, TRUTH, TRUTH_NOTE, ago, last_sessions, price_chart, read_json,
                read_parquet, show, strip_chart)
from vrdet.labels import STATE_NAMES
from vrdet.live import BAR, LIVE_DIR, MODEL_DIR, TZ, classify

LIGHT = {'Quiet': 'GREEN', 'Normal': 'YELLOW', 'Chaotic': 'RED'}
ADVICE = {'Quiet': 'Calm market: small, orderly moves.',
          'Normal': 'Typical intraday volatility.',
          'Chaotic': 'Large, fast swings: elevated risk of sudden losses.'}


def sidebar(meta):
    """Alarm sensitivity (shared with the runner through data/live/settings.json) and the model card."""
    settings = read_json(LIVE_DIR / 'settings.json', {})
    tuned = float(meta.get('tau', 0.3))
    presets = {f'Balanced (tuned, τ = {tuned:.2f})': tuned,
               'High: catch more, more false alarms (τ = 0.20)': 0.20,
               'Low: fewer alarms, may miss some (τ = 0.50)': 0.50}
    current = float(settings.get('tau', tuned))
    names = list(presets)
    idx = next((i for i, n in enumerate(names) if abs(presets[n] - current) < 1e-9), 0)
    with st.sidebar:
        choice = st.radio('Alarm sensitivity', names, index=idx,
                          help='Chaotic is called when the model gives P(Chaotic) ≥ τ. Also used by the alert runner.')
        if abs(presets[choice] - current) > 1e-9:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            (LIVE_DIR / 'settings.json').write_text(json.dumps({'tau': presets[choice]}))

        st.divider()
        st.subheader('Model')
        wf = meta.get('walk_forward', {})
        if meta:
            st.markdown(f"**{meta.get('model')}**, trained on {meta.get('n_bars', 0):,} bars "
                        f"({meta.get('train_start', '')[:10]} to {meta.get('train_end', '')[:10]}).")
        if 'LightGBM + HMM' in wf:
            m, h = wf['LightGBM + HMM'], wf.get('HMM filter', {})
            st.dataframe(pd.DataFrame({
                'Model': [m['macro_f1'], m['chaotic_recall'], m['episode_recall'], m['false_alarms_per_week']],
                'HMM': [h.get('macro_f1'), h.get('chaotic_recall'), h.get('episode_recall'),
                        h.get('false_alarms_per_week')],
            }, index=['Macro-F1', 'Chaotic recall', 'Episodes caught', 'False alarms/wk']), width='stretch')
            st.caption('Walk-forward test 2022–2026, never seen in training (tuned τ). HMM = the HMM filter alone.')
    return presets[choice]


def status_card(regime, sub):
    c = COLORS[regime]
    st.markdown(
        f"""<div style="border-radius:14px;padding:22px 26px;background:{c}1f;border:2px solid {c};">
        <div style="display:flex;align-items:center;gap:18px;">
          <div style="width:58px;height:58px;border-radius:50%;background:{c};box-shadow:0 0 22px {c};flex:none;"></div>
          <div><div style="font-size:0.85rem;letter-spacing:0.08em;opacity:0.75;">CURRENT REGIME · {LIGHT[regime]}</div>
          <div style="font-size:2.3rem;font-weight:700;line-height:1.1;color:{c};">{regime}</div>
          <div style="opacity:0.85;">{ADVICE[regime]}</div></div></div>
        <div style="margin-top:12px;font-size:0.85rem;opacity:0.75;">{sub}</div></div>""",
        unsafe_allow_html=True)


def render():
    tau = sidebar(read_json(MODEL_DIR / 'live_model.json', {}))
    live_view(tau, st.session_state.get('n_sessions', 3))


@st.fragment(run_every='20s')
def live_view(tau, n_sessions):
    now = pd.Timestamp.now(tz=TZ)
    state = read_json(DATA_DIR / 'state.json')
    beat = read_json(DATA_DIR / 'heartbeat.json', {})
    hist = read_parquet(DATA_DIR / 'history.parquet')
    if not state or hist is None:
        st.info('No replay running. Start one: `python app/runner.py --replay 2025-04-07`.' if REPLAY else
                'No live data yet. Start the runner: `python app/runner.py` (or `python app/run.py` for both).')
        return
    hist['regime'] = classify(hist[list(STATE_NAMES)], tau)

    alive = beat.get('alive_at') and (now - pd.Timestamp(beat['alive_at'])) < pd.Timedelta(minutes=2)
    if state.get('replay'):
        st.info(f"REPLAY of the {state['replay']} session through the live pipeline (historical data, no Telegram).",
                icon=':material/replay:')
    elif not alive:
        st.warning('The runner is not running, so this view will not update. Start it with `python app/runner.py`.')
    if state.get('error'):
        st.error(f"Last update could not reach Angel One ({state['error']}); showing the latest data available.")

    last = hist.iloc[-1]
    ts = hist.index[-1]
    regime = STATE_NAMES[int(last['regime'])]
    p = last[list(STATE_NAMES)]
    market_open = now.day_of_week < 5 and (now.normalize() + pd.Timedelta(hours=9, minutes=15)) <= now \
        <= (now.normalize() + pd.Timedelta(hours=15, minutes=31))
    sub = (f"Bar closed {(ts + BAR):%a %d %b %H:%M} IST · updated {ago(state['updated_at'], now)}"
           + (f" · next update {pd.Timestamp(beat['next_update']):%a %H:%M}" if alive and beat.get('next_update') else '')
           + ('' if market_open or state.get('replay') else ' · market closed, showing the last session'))

    left, right = st.columns([1.15, 1])
    with left:
        status_card(regime, sub)
    with right:
        a, b = st.columns(2)
        a.metric('P(Chaotic)', f"{p['Chaotic']:.0%}", help=f'Alarm when ≥ {tau:.0%}')
        b.metric('Confidence', f'{p.max():.0%}', help='Probability of the regime shown')
        prev = hist['close'].iloc[-2] if len(hist) > 1 else None
        a.metric('NIFTY 50', f"{last['close']:,.2f}", None if prev is None else f"{last['close'] - prev:+,.2f}")
        b.metric('India VIX', '–' if pd.isna(last['vix']) else f"{last['vix']:.2f}")
        hmm = STATE_NAMES[int(last['hmm_regime'])]
        st.caption(f"Probabilities: Quiet {p['Quiet']:.0%} · Normal {p['Normal']:.0%} · Chaotic {p['Chaotic']:.0%}"
                   f" · HMM filter alone says **{hmm}**")

    recent = last_sessions(hist, n_sessions)
    show(price_chart(recent, tau, {'Model (LightGBM + HMM)': 'Chaotic', 'HMM filter': 'hmm_p_chaotic'}))
    if 'truth' in recent:
        known = recent['truth'].notna()
        match = {name: (recent.loc[known, col] == recent.loc[known, 'truth']).mean()
                 for name, col in [('model', 'regime'), ('HMM filter', 'hmm_regime')]}
        show(strip_chart(pd.DataFrame({TRUTH: recent['truth'], 'Model (LightGBM + HMM)': recent['regime'],
                                       'HMM filter': recent['hmm_regime']}), recent['close'])[1:])  # strips only
        st.caption(f"{TRUTH_NOTE} Over these bars the model matches it on {match['model']:.0%}, the HMM filter alone "
                   f"on {match['HMM filter']:.0%}.")

    st.subheader('Alerts')
    alerts = [json.loads(line) for line in (DATA_DIR / 'alerts.jsonl').read_text().splitlines()] \
        if (DATA_DIR / 'alerts.jsonl').exists() else []
    if alerts:
        newest = alerts[-1]['time'] + alerts[-1]['bar_start']
        if st.session_state.get('seen_alert') not in (None, newest):
            st.toast(('Market turned CHAOTIC' if alerts[-1]['event'] == 'chaotic' else 'All clear: Chaotic regime over'),
                     icon=':material/warning:' if alerts[-1]['event'] == 'chaotic' else ':material/check_circle:')
        st.session_state['seen_alert'] = newest
        df = pd.DataFrame(alerts[::-1][:50])
        df['bar'] = pd.to_datetime(df['bar_start']).dt.strftime('%a %d %b %H:%M')
        df['event'] = df['event'].map({'chaotic': 'Entered CHAOTIC', 'clear': 'All clear'})
        st.dataframe(df[['bar', 'event', 'p_chaotic', 'close', 'telegram_sent']].rename(
            columns={'bar': 'Bar', 'event': 'Event', 'p_chaotic': 'P(Chaotic)', 'close': 'NIFTY', 'telegram_sent': 'Telegram'}),
            hide_index=True, width='stretch')
    else:
        st.session_state.setdefault('seen_alert', '')
        st.caption('No alerts yet. You are alerted when the market turns Chaotic, and again when it clears '
                   '(after 4 calm bars).' + ('' if state.get('telegram') is not None else
                                            ' Telegram is off: set TELEGRAM_BOT_TOKEN in .env to enable it.'))
