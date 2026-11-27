"""Risk Radar page: live NIFTY 50 ticker, the production model's live regime call, chart and alerts."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from ui import (COLORS, DATA_DIR, LINE, REPLAY, TRUTH, TRUTH_NOTE, ago, last_sessions, price_chart, read_json,
                read_parquet, show, strip_chart)
from vrdet.labels import STATE_NAMES
from vrdet.live import BAR, LIVE_DIR, MODEL_DIR, TZ, classify
from vrdet.stream import TICKS_FILE, market_window

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
    if not REPLAY:
        ticker()
    live_view(tau, st.session_state.get('n_sessions', 3))


def _pill(text, color, pulse=False):
    anim = 'animation:vrdet-pulse 1.6s ease-in-out infinite;' if pulse else ''
    return (f'<style>@keyframes vrdet-pulse {{0%,100% {{opacity:1}} 50% {{opacity:0.25}}}}</style>'
            f'<span style="display:inline-flex;align-items:center;gap:8px;font-size:0.85rem;letter-spacing:0.06em;">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{color};{anim}"></span>{text}</span>')


@st.fragment(run_every='2s')
def ticker():
    """Live NIFTY 50 / India VIX quotes streamed by the runner (Angel One WebSocket). Display only: the regime
    below is still computed from completed 15-minute bars."""
    now = pd.Timestamp.now(tz=TZ)
    t = read_json(TICKS_FILE)
    nifty = (t or {}).get('quotes', {}).get('NIFTY')
    fresh_file = t and now - pd.Timestamp(t['written_at']) < pd.Timedelta(minutes=2)
    if not nifty:
        st.markdown(_pill('NIFTY 50 LIVE · waiting for the stream' if fresh_file else
                          'NIFTY 50 LIVE · stream offline (it runs inside `python app/runner.py`)', '#9aa3ad'),
                    unsafe_allow_html=True)
        return
    tick_age = now - pd.Timestamp(nifty['time'])
    if not market_window(now) or (t or {}).get('status') == 'closed':
        pill = _pill(f"MARKET CLOSED · last tick {pd.Timestamp(nifty['time']):%a %d %b %H:%M:%S}", '#9aa3ad')
    elif fresh_file and t['status'] == 'live' and tick_age < pd.Timedelta(seconds=15):
        pill = _pill(f"LIVE · {pd.Timestamp(nifty['time']):%H:%M:%S} IST · Angel One stream", '#2e9e5b', pulse=True)
    elif fresh_file:
        pill = _pill(f"{t['status'].upper()} · last tick {ago(nifty['time'], now)}", '#d9a400')
    else:
        pill = _pill(f"STREAM OFFLINE · last tick {ago(nifty['time'], now)}; start `python app/runner.py`", '#d93636')
    st.markdown(pill, unsafe_allow_html=True)

    vix = t['quotes'].get('INDIAVIX')
    path = pd.DataFrame(t.get('path', []), columns=['t', 'ltp'])
    path['time'] = pd.to_datetime(path['t'], unit='s', utc=True).dt.tz_convert(TZ)
    today = path[path['time'].dt.normalize() == pd.Timestamp(nifty['time']).normalize()]

    def change(q):
        if not q.get('prev_close'):
            return None
        d = q['ltp'] - q['prev_close']
        return f"{d:+,.2f} ({d / q['prev_close']:+.2%})"

    cols = st.columns([1.3, 1, 1.2, 1.4])
    cols[0].metric('NIFTY 50 (live)', f"{nifty['ltp']:,.2f}", change(nifty), help='Change vs the previous close')
    if vix:
        cols[1].metric('India VIX (live)', f"{vix['ltp']:.2f}", change(vix), delta_color='inverse')
    if nifty.get('high'):
        cols[2].metric('Day range', f"{nifty['high'] - nifty['low']:,.1f} pts",
                       f"{nifty['low']:,.0f} – {nifty['high']:,.0f} ({(nifty['high'] - nifty['low']) / nifty['low']:.2%})",
                       delta_color='off', delta_arrow='off', help='Day high minus day low')
    # the 15-minute bar now forming: what the model will score at its close
    bar_start = (now.normalize() + pd.Timedelta(hours=9, minutes=15)
                 + BAR * ((now - now.normalize() - pd.Timedelta(hours=9, minutes=15)) // BAR))
    bar = today[today['time'] >= bar_start]
    if len(bar) and market_window(now):
        o, last = bar['ltp'].iloc[0], bar['ltp'].iloc[-1]
        cols[3].metric(f"Bar forming {bar_start:%H:%M}–{bar_start + BAR:%H:%M}", f'{last - o:+,.2f}',
                       f"range {bar['ltp'].max() - bar['ltp'].min():,.2f} so far", delta_color='off', delta_arrow='off',
                       help='Move since the bar opened. The regime is scored when this bar closes.')

    if len(today) > 1:
        base = alt.Chart(today).encode(x=alt.X('time:T', title=None,
                                               axis=alt.Axis(format='%H:%M', grid=False, tickCount=8)))
        line = base.mark_line(color=LINE, strokeWidth=1.6).encode(
            y=alt.Y('ltp:Q', title='NIFTY 50', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('time:T', format='%H:%M:%S'), alt.Tooltip('ltp:Q', format=',.2f', title='NIFTY')])
        dot = alt.Chart(today.tail(1)).mark_point(filled=True, size=60, color=LINE).encode(x='time:T', y='ltp:Q')
        layers = line + dot
        if nifty.get('prev_close'):
            layers += alt.Chart(pd.DataFrame({'y': [nifty['prev_close']]})).mark_rule(
                color='#888', strokeDash=[4, 4]).encode(y='y:Q', tooltip=[alt.Tooltip('y:Q', title='Previous close')])
        st.altair_chart(layers.properties(height=150), width='stretch')
        st.caption("Today's NIFTY 50: 1-minute closes before the stream started, then live ticks (5-second points; "
                   'dashed: previous close). The regime below is scored on completed 15-minute bars.')


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
        a.metric('NIFTY 50 (bar close)', f"{last['close']:,.2f}", None if prev is None else f"{last['close'] - prev:+,.2f}")
        b.metric('India VIX (bar close)', '–' if pd.isna(last['vix']) else f"{last['vix']:.2f}")
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
