"""Chaotic-regime alerts: hysteresis on per-bar regime calls, and an optional Telegram channel.

One alert when the market turns Chaotic, one "all clear" after `clear_bars` consecutive non-Chaotic
bars. A single calm bar inside a Chaotic phase therefore does not produce clear/chaotic spam.
"""
import json

import pandas as pd
import requests

from .labels import STATE_NAMES


class AlertState:
    def __init__(self, clear_bars=4, in_alarm=False, calm=0, last_ts=None, alarm_start=None):
        self.clear_bars, self.in_alarm, self.calm = clear_bars, in_alarm, calm
        self.last_ts, self.alarm_start = last_ts, alarm_start

    def update(self, ts, regime):
        """Feed one completed bar; returns 'chaotic', 'clear' or None."""
        self.last_ts = str(ts)
        if regime == 2:
            self.calm = 0
            if not self.in_alarm:
                self.in_alarm, self.alarm_start = True, str(ts)
                return 'chaotic'
        elif self.in_alarm:
            self.calm += 1
            if self.calm >= self.clear_bars:
                self.in_alarm, self.calm = False, 0
                return 'clear'
        return None

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, s):
        return cls(**json.loads(s))


def message(event, ts, proba, close):
    """Plain-text alert for a bar stamped ts (bar closes 15 minutes later)."""
    end = (ts + pd.Timedelta(minutes=15)).strftime('%d %b %H:%M')
    probs = ', '.join(f'{k} {proba[k]:.0%}' for k in STATE_NAMES)
    if event == 'chaotic':
        head = 'RED: NIFTY 50 has entered a CHAOTIC volatility regime.'
        tail = 'Expect large, fast swings. Consider reducing position size / widening stops.'
    else:
        head = 'ALL CLEAR: NIFTY 50 has left the Chaotic regime.'
        tail = 'Volatility is back to Normal/Quiet levels.'
    return f'{head}\nBar closed {end} IST, NIFTY {close:,.2f}\nModel: {probs}\n{tail}\n(Risk indicator, not investment advice.)'


class Telegram:
    """Telegram Bot API over HTTPS. Private by default: only chat ids in TELEGRAM_CHAT_IDS, plus anyone
    who sends `/start <TELEGRAM_SUBSCRIBE_CODE>` when a code is set. Subscribers are stored locally."""

    def __init__(self, token, chat_ids=(), subscribe_code='', store=None):
        self.url = f'https://api.telegram.org/bot{token}/'
        self.fixed = {str(c) for c in chat_ids if str(c).strip()}
        self.code, self.store = subscribe_code.strip(), store
        saved = json.loads(store.read_text()) if store and store.exists() else {}
        self.subscribers, self.offset = set(saved.get('subscribers', [])), saved.get('offset', 0)

    def _save(self):
        if self.store:
            self.store.write_text(json.dumps({'subscribers': sorted(self.subscribers), 'offset': self.offset}))

    def _call(self, method, **params):
        r = requests.post(self.url + method, json=params, timeout=20)
        r.raise_for_status()
        return r.json()['result']

    @property
    def recipients(self):
        return self.fixed | self.subscribers

    def send(self, text):
        sent = 0
        for chat in self.recipients:
            try:
                self._call('sendMessage', chat_id=chat, text=text)
                sent += 1
            except Exception as e:
                print(f'Telegram send to {chat} failed: {e}')
        return sent

    def poll(self, status_text=lambda: ''):
        """Handle /start <code>, /stop and /status messages sent to the bot since the last poll."""
        for upd in self._call('getUpdates', offset=self.offset, timeout=0):
            self.offset = upd['update_id'] + 1
            msg = upd.get('message') or {}
            chat, text = str(msg.get('chat', {}).get('id', '')), (msg.get('text') or '').strip()
            if not chat:
                continue
            cmd, _, arg = text.partition(' ')
            if cmd == '/start':
                if chat in self.recipients:
                    reply = 'Already subscribed to VRDet Chaotic-regime alerts.'
                elif self.code and arg.strip() == self.code:
                    self.subscribers.add(chat)
                    reply = 'Subscribed to VRDet Chaotic-regime alerts. Send /stop to unsubscribe.'
                else:
                    reply = (f'This bot is private. Your chat id is {chat}; the owner can add it to '
                             f'TELEGRAM_CHAT_IDS, or send /start <code>.')
            elif cmd == '/stop':
                self.subscribers.discard(chat)
                reply = 'Unsubscribed.' + (' (This chat is in TELEGRAM_CHAT_IDS, remove it there.)'
                                           if chat in self.fixed else '')
            elif cmd == '/status' and chat in self.recipients:
                reply = status_text() or 'No reading yet.'
            else:
                continue
            self._call('sendMessage', chat_id=chat, text=reply)
        self._save()
