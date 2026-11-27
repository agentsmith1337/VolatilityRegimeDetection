"""Live NIFTY 50 / India VIX ticks from the Angel One SmartAPI WebSocket (smart-stream v2).

The runner owns the Angel One session, so it also owns the stream: `TickStream` runs in a background thread,
keeps the latest quote of each instrument plus a 5-second price path of today's session, and writes them to
data/live/ticks.json (at most once a second) for the dashboard's live ticker. The stream is only connected
during market hours; it reconnects with a fresh session when the connection drops.

The ticker is display only: the regime model still scores completed 15-minute candles, so live ticks never
change what was evaluated.
"""
import json
import logging
import os
import threading
import time

import pandas as pd

from .angel import TOKENS, request_candles
from .live import LIVE_DIR, TZ

log = logging.getLogger('vrdet.stream')

NAMES = {token: name for name, token in TOKENS.items()}
OPEN, CLOSE = pd.Timedelta(hours=9, minutes=0), pd.Timedelta(hours=15, minutes=40)   # connect window (IST)
SAMPLE_SECONDS = 5      # resolution of the stored price path
WRITE_EVERY = 1.0       # seconds between file writes
TICKS_FILE = LIVE_DIR / 'ticks.json'


def market_window(now):
    """True on weekdays between 09:00 and 15:40 IST (holidays simply produce no ticks)."""
    day = now.normalize()
    return now.day_of_week < 5 and day + OPEN <= now <= day + CLOSE


def parse_tick(data):
    """Angel One quote packet (prices in paise) -> plain dict in rupees with an IST timestamp."""
    name = NAMES.get(str(data.get('token')))
    if name is None or not data.get('last_traded_price'):
        return None, None
    q = {'ltp': data['last_traded_price'] / 100,
         'time': pd.Timestamp(data['exchange_timestamp'], unit='ms', tz='UTC').tz_convert(TZ).isoformat()}
    for key, field in (('open', 'open_price_of_the_day'), ('high', 'high_price_of_the_day'),
                       ('low', 'low_price_of_the_day'), ('prev_close', 'closed_price')):
        if data.get(field):
            q[key] = data[field] / 100
    return name, q


class TickStream:
    def __init__(self, session, out_path=TICKS_FILE):
        """session: callable(fresh) returning a logged-in SmartConnect (the runner's; fresh=True logs in again)."""
        self.session, self.path = session, out_path
        self.quotes, self.path_nifty = {}, []          # path: [[epoch seconds, ltp], ...] for today
        self.status, self.error, self.connected_at = 'starting', None, None
        self.ws, self.lock, self.stop_flag = None, threading.Lock(), threading.Event()
        self._last_write = 0.0
        self._load_today()

    def _load_today(self):
        """Keep today's path across runner restarts."""
        try:
            old = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        today = pd.Timestamp.now(tz=TZ).normalize().timestamp()
        self.path_nifty = [p for p in old.get('path', []) if p[0] >= today]
        self.quotes = {k: v for k, v in old.get('quotes', {}).items() if pd.Timestamp(v['time']).timestamp() >= today}

    # ------------------------------------------------------------------ thread
    def start(self):
        threading.Thread(target=self._run, name='vrdet-ticks', daemon=True).start()
        return self

    def stop(self):
        self.stop_flag.set()
        self._close()

    def _run(self):
        delay, fresh = 5, False
        while not self.stop_flag.is_set():
            if not market_window(pd.Timestamp.now(tz=TZ)):
                self._set_status('closed')
                self.stop_flag.wait(30)
                continue
            self.connected_at = None
            try:
                self._set_status('connecting')
                self._connect(fresh)                  # blocks until the socket closes
            except Exception as e:
                self.error = str(e)
                log.warning('tick stream failed: %s', e)
            if self.stop_flag.is_set():
                break
            # never got live (e.g. an expired session): log in again next time; otherwise reconnect quickly
            fresh = self.connected_at is None
            delay = min(delay * 2, 120) if fresh else 5
            self._set_status('reconnecting')
            self.stop_flag.wait(delay)

    def _connect(self, fresh):
        import logzero
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2

        smart = self.session(fresh)
        threading.Thread(target=self._backfill, args=(smart,), name='vrdet-ticks-backfill', daemon=True).start()
        ws = SmartWebSocketV2(smart.access_token, os.environ['ANGEL_API_KEY'], os.environ['ANGEL_CLIENT_CODE'],
                              smart.getfeedToken(), max_retry_attempt=0)
        logzero.logfile(None)                         # the library logs every packet to logs/<date>/app.log
        logzero.loglevel(logging.ERROR)
        ws.current_retry_attempt = 0                  # not initialised by the library, read in its error handler
        tokens = [{'exchangeType': SmartWebSocketV2.NSE_CM, 'tokens': list(NAMES)}]
        ws.on_open = lambda wsapp: (ws.subscribe('vrdet', SmartWebSocketV2.QUOTE, tokens),
                                    self._set_status('live', connected=True))
        ws.on_data = lambda wsapp, data: self._on_tick(data)
        ws.on_error = lambda *args: setattr(self, 'error', ' '.join(map(str, args)))
        # the library's own close handler has the wrong signature for websocket-client (it raises on close)
        ws._on_close = lambda wsapp, *args: self._set_status('disconnected')
        self.ws = ws
        ws.connect()

    def _close(self):
        try:
            if self.ws is not None:
                self.ws.close_connection()
        except Exception:
            pass

    # ------------------------------------------------------------------ data
    def _backfill(self, smart):
        """Seed today's path with 1-minute closes up to the first tick, so the chart covers the whole session
        even when the runner starts mid-day. Runs in its own thread: Angel One rejects a candle request made
        right after login ("exceeding access rate"), so it waits a few seconds and retries."""
        for wait in (5, 15, 30, 60, 120):
            if self.stop_flag.wait(wait):
                return
            now = pd.Timestamp.now(tz=TZ)
            start = now.normalize() + pd.Timedelta(hours=9, minutes=15)
            with self.lock:
                first = self.path_nifty[0][0] if self.path_nifty else None
            if now < start or (first is not None and first <= start.timestamp() + 120):
                return
            try:
                rows = request_candles(smart, TOKENS['NIFTY'], 'ONE_MINUTE', start, now, retries=1)
            except Exception as e:
                log.warning('tick backfill failed: %s', e)
                rows = []
            if rows:
                # a 1-minute candle closes 60 s after its timestamp; keep only those before the first tick
                seed = [[int(pd.Timestamp(r[0]).timestamp()) + 60, float(r[4])] for r in rows]
                with self.lock:
                    first = self.path_nifty[0][0] if self.path_nifty else None
                    self.path_nifty = [p for p in seed if first is None or p[0] < first] + self.path_nifty
                log.info('tick backfill: %d one-minute closes', len(seed))
                return

    def _on_tick(self, data):
        name, q = parse_tick(data)
        if name is None:
            return
        with self.lock:
            self.quotes[name] = {**self.quotes.get(name, {}), **q}
            if name == 'NIFTY':
                t = int(pd.Timestamp(q['time']).timestamp())
                t -= t % SAMPLE_SECONDS
                if self.path_nifty and self.path_nifty[-1][0] == t:
                    self.path_nifty[-1][1] = q['ltp']
                elif not self.path_nifty or t > self.path_nifty[-1][0]:
                    if self.path_nifty and pd.Timestamp(self.path_nifty[-1][0], unit='s').date() \
                            != pd.Timestamp(t, unit='s').date():
                        self.path_nifty = []          # new session
                    self.path_nifty.append([t, q['ltp']])
        if time.monotonic() - self._last_write >= WRITE_EVERY:
            self._write()

    def _set_status(self, status, connected=False):
        if connected:
            self.connected_at, self.error = pd.Timestamp.now(tz=TZ).isoformat(timespec='seconds'), None
        if status != self.status:
            log.info('tick stream: %s', status)
        self.status = status
        self._write()

    def _write(self):
        with self.lock:
            out = {'status': self.status, 'error': self.error, 'connected_at': self.connected_at,
                   'written_at': pd.Timestamp.now(tz=TZ).isoformat(timespec='seconds'),
                   'quotes': self.quotes, 'path': self.path_nifty}
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(out))
        for _ in range(5):                             # Windows: the dashboard may be reading the file right now
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                time.sleep(0.05)
        self._last_write = time.monotonic()
