"""VRDet live inference loop (runs locally, nothing is hosted).

At each 15-minute bar close (09:30 ... 15:30 IST, plus a short delay for the candle to be published):
fetch new NIFTY / India VIX candles from Angel One -> rebuild the causal features -> score the frozen
"LightGBM + HMM" model -> write data/live/state.json + history.parquet for the dashboard -> alert on
a transition into / out of Chaotic.

Replay mode re-runs a past session bar by bar through the same pipeline (into data/replay, no Telegram),
for demos when the market is closed or calm. Live features are causal, so scoring bar t with all
history gives exactly what the app computed at bar t.

Usage:
    python app/runner.py                          # loop forever
    python app/runner.py --once                   # one update now, then exit
    python app/runner.py --replay 2025-04-07      # replay a session, one bar every 4 s
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vrdet.alerts import AlertState, Telegram, message  # noqa: E402
from vrdet.angel import load_env, login  # noqa: E402
from vrdet.data import load_raw  # noqa: E402
from vrdet.labels import STATE_NAMES  # noqa: E402
from vrdet.live import BAR, LIVE_DIR, TZ, LiveData, RegimeModel, classify, model_frame  # noqa: E402
from vrdet.zoo import CONTEXT  # noqa: E402

REPLAY_DIR = ROOT / 'data' / 'replay'
PUBLISH_DELAY = pd.Timedelta(seconds=45)   # Angel One publishes a candle shortly after it closes
HISTORY_SESSIONS = 10                       # sessions of per-bar output kept for the dashboard
ALERT_MAX_AGE = 2 * BAR                     # older events (e.g. after downtime) update state silently

log = logging.getLogger('vrdet')


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    for _ in range(5):  # Windows: the dashboard may be reading the target at this instant
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2)


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def bar_closes(day):
    """Close times of the 25 regular-session 15-minute bars on `day` (09:30 ... 15:30 IST)."""
    start = pd.Timestamp(day).tz_localize(TZ) + pd.Timedelta(hours=9, minutes=30)
    return [start + i * BAR for i in range(25)]


def next_run(now):
    """Next bar close (+ publish delay) after `now`, skipping weekends. Holidays just yield no new bar."""
    day = now.normalize().tz_localize(None)
    while True:
        if day.day_of_week < 5:
            for c in bar_closes(day):
                if c + PUBLISH_DELAY > now:
                    return c + PUBLISH_DELAY
        day += pd.Timedelta(days=1)


class Runner:
    def __init__(self, out_dir=LIVE_DIR, telegram=True):
        load_env()
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = RegimeModel()
        self.data = LiveData(load_raw())
        self.smart = None
        s = read_json(self.dir / 'alert_state.json', None)
        self.alerts = AlertState(**s) if s else AlertState()
        token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip() if telegram else ''
        self.telegram = Telegram(token, os.environ.get('TELEGRAM_CHAT_IDS', '').split(','),
                                 os.environ.get('TELEGRAM_SUBSCRIBE_CODE', ''),
                                 LIVE_DIR / 'telegram.json') if token else None
        self.last_status = ''
        log.info('model %s, tau %.2f, trained to %s; Telegram %s', self.model.meta['model'], self.model.tau,
                 self.model.meta['train_end'], 'on' if self.telegram else 'off')

    def tau(self):
        """Alert threshold: the dashboard's sensitivity setting, else the tuned value."""
        return float(read_json(LIVE_DIR / 'settings.json', {}).get('tau', self.model.tau))

    def fetch(self, now):
        for attempt in range(2):
            try:
                if self.smart is None:
                    self.smart = login()
                self.data.refresh_angel(self.smart, now)
                return
            except Exception as e:  # expired session or network error: log in again once
                log.warning('Angel One fetch failed (%s), re-login', e)
                self.smart = None
        raise RuntimeError('Angel One unavailable')

    def cycle(self):
        t0 = time.perf_counter()
        now = pd.Timestamp.now(tz=TZ)
        error = None
        try:
            self.fetch(now)
        except RuntimeError as e:
            error = str(e)
        self.data.refresh_yf(now)
        raw = self.data.frames()
        X = model_frame(raw, self.model.hmm, live=True)
        state = self.publish(X, self.model.predict_proba(X), raw, now, error=error)
        state['cycle_seconds'] = round(time.perf_counter() - t0, 1)
        write_json(self.dir / 'state.json', state)
        log.info('%s | %.1fs%s', self.last_status, state['cycle_seconds'], f' | {error}' if error else '')

    def publish(self, X, proba, raw, now, error=None, replay=None, truth=None):
        """Write the dashboard files for the latest bar in `proba` and process alerts.

        truth: hindsight HMM labels (Viterbi + minimum duration, i.e. the training target) for comparison only;
        they use later bars, so the most recent ones are provisional and can change as new bars arrive."""
        tau = self.tau()
        if truth is None:
            truth = self.model.hmm.hindsight(X)
        dates = proba.index.normalize()
        hist = proba[dates >= dates.unique()[-HISTORY_SESSIONS:][0]].copy()
        hist['regime'] = classify(hist[list(STATE_NAMES)], tau)
        hmm_p = X.loc[hist.index, ['p_hmm_Quiet', 'p_hmm_Normal', 'p_hmm_Chaotic']].to_numpy()
        hist['hmm_regime'] = hmm_p.argmax(axis=1)
        hist['hmm_p_chaotic'] = hmm_p[:, 2]
        hist['truth'] = truth.reindex(hist.index)
        hist = hist.join(raw['NIFTY_15m'][['open', 'high', 'low', 'close']])
        hist = hist.join(raw['INDIAVIX_15m']['close'].rename('vix'))
        hist.to_parquet(self.dir / 'history.parquet')
        # Inputs of the same bars (+ history for sequence models), so the Model Lab can score any model
        feats = X[(X.index <= hist.index[-1]) & X['p_hmm_Chaotic'].notna()]
        first = feats.index.get_loc(hist.index[0])
        feats.iloc[max(0, first - CONTEXT):].to_parquet(self.dir / 'features.parquet')

        self.process_alerts(hist, now)

        last, ts = hist.iloc[-1], hist.index[-1]
        state = {
            'updated_at': pd.Timestamp.now(tz=TZ).isoformat(timespec='seconds'),
            'bar_start': ts.isoformat(), 'bar_close': (ts + BAR).isoformat(),
            'regime': int(last['regime']), 'regime_name': STATE_NAMES[int(last['regime'])],
            'proba': {k: float(last[k]) for k in STATE_NAMES},
            'hmm_regime': STATE_NAMES[int(last['hmm_regime'])],
            'close': float(last['close']), 'vix': None if pd.isna(last['vix']) else float(last['vix']),
            'tau': tau, 'in_alarm': self.alerts.in_alarm, 'error': error, 'replay': replay,
            'telegram': len(self.telegram.recipients) if self.telegram else None,
        }
        write_json(self.dir / 'state.json', state)
        self.last_status = (f'NIFTY {state["close"]:,.2f}: {state["regime_name"]} '
                            f'(P(Chaotic) {state["proba"]["Chaotic"]:.0%}) as of {(ts + BAR):%d %b %H:%M} IST')
        return state

    def process_alerts(self, hist, now):
        """Feed bars not yet seen to the hysteresis; deliver events for recent bars only."""
        new = hist if self.alerts.last_ts is None else hist[hist.index > pd.Timestamp(self.alerts.last_ts)]
        if self.alerts.last_ts is None:
            new = new.iloc[-1:]  # first run: judge the current bar only, do not replay history
        for ts, row in new.iterrows():
            event = self.alerts.update(ts, int(row['regime']))
            if not event:
                continue
            fresh = now - (ts + BAR) <= ALERT_MAX_AGE
            record = {'time': now.isoformat(timespec='seconds'), 'bar_start': ts.isoformat(), 'event': event,
                      'p_chaotic': round(float(row['Chaotic']), 3), 'close': float(row['close']),
                      'delivered': fresh, 'telegram_sent': 0}
            if fresh and self.telegram:
                record['telegram_sent'] = self.telegram.send(message(event, ts, row, row['close']))
            log.warning('ALERT %s at bar %s%s', event, ts, '' if fresh else ' (stale, not sent)')
            with open(self.dir / 'alerts.jsonl', 'a') as fh:
                fh.write(json.dumps(record) + '\n')
        write_json(self.dir / 'alert_state.json', self.alerts.__dict__)

    def idle(self, until):
        """Sleep until `until`, keeping a heartbeat for the dashboard and answering Telegram commands."""
        while (now := pd.Timestamp.now(tz=TZ)) < until:
            write_json(self.dir / 'heartbeat.json', {'alive_at': now.isoformat(timespec='seconds'),
                                                     'next_update': until.isoformat(timespec='seconds')})
            if self.telegram:
                try:
                    self.telegram.poll(lambda: self.last_status)
                except Exception as e:
                    log.debug('Telegram poll failed: %s', e)
            time.sleep(min(10, max(0.5, (until - now).total_seconds())))


def replay(day, pause, lead_sessions=2):
    """Replay session `day` bar by bar into data/replay, starting `lead_sessions` sessions earlier for context."""
    shutil.rmtree(REPLAY_DIR, ignore_errors=True)
    runner = Runner(REPLAY_DIR, telegram=False)
    raw = runner.data.frames()
    day = pd.Timestamp(day)
    end = day.tz_localize(TZ) + pd.Timedelta(days=1)
    cut = {k: df[df.index < end] if k.endswith('_15m') else df[df.index < day] for k, df in raw.items()}
    X = model_frame(cut, runner.model.hmm, live=True)
    proba = runner.model.predict_proba(X)
    truth = runner.model.hmm.hindsight(X)   # labels knowing the whole replayed session
    sessions = proba.index.normalize().unique()
    if day.tz_localize(TZ) not in sessions:
        raise SystemExit(f'No session on {day.date()} in the data')
    bars = proba.index[proba.index.normalize() >= sessions[sessions <= day.tz_localize(TZ)][-1 - lead_sessions]]
    first_live = proba.index[proba.index.normalize() == day.tz_localize(TZ)][0]
    log.info('replaying %s: %d bars of context, then %d bars every %.1fs', day.date(),
             (bars < first_live).sum(), (bars >= first_live).sum(), pause)
    runner.alerts.last_ts = str(bars[0] - BAR)  # alerts judged on the replayed bars only
    for ts in bars:
        now = ts + BAR + PUBLISH_DELAY
        runner.publish(X, proba[proba.index <= ts], cut, now, replay=str(day.date()), truth=truth)
        if ts >= first_live:
            log.info(runner.last_status)
            runner.idle(pd.Timestamp.now(tz=TZ) + pd.Timedelta(seconds=pause))
    log.info('replay finished')
    runner.idle(pd.Timestamp.now(tz=TZ) + pd.Timedelta(minutes=30))  # keep the dashboard "alive" for a while


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--once', action='store_true', help='run one update and exit')
    parser.add_argument('--replay', metavar='YYYY-MM-DD', help='replay a past session into data/replay')
    parser.add_argument('--pause', type=float, default=4.0, help='seconds per bar in replay mode')
    args = parser.parse_args()
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / 'logs' / 'runner.log')])

    if args.replay:
        replay(args.replay, args.pause)
        return
    runner = Runner()
    runner.cycle()
    while not args.once:
        runner.idle(next_run(pd.Timestamp.now(tz=TZ)))
        try:
            runner.cycle()
        except Exception:
            log.exception('update failed; retrying at the next bar')


if __name__ == '__main__':
    main()
