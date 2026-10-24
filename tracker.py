"""
Signal tracker (NO TRADING).

The bot never places orders. After a signal is sent, it follows the price and
reports in Telegram when TP1, TP2, the stop loss or breakeven is reached, and
the final result of the signal in R (R = the distance from entry to stop).

* One open signal per pair at a time, any number of pairs at once.
* TP1 counts TP1_CLOSE_PCT % of the idea, TP2 the rest.
* After TP1 the stop moves to entry if MOVE_SL_TO_BE_AFTER_TP1=true.
* If the stop and a target are touched in the same moment, the stop is counted
  (the more careful result).
* Open signals are saved in state/signals.json, so a restart does not lose them.
"""
import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

from .notifier import Telegram, closed_message, exit_message, not_filled_message, signal_message
from .state import SymbolState
from .utils import Journal, now_ms

log = logging.getLogger("tracker")
STATE_FILE = "state/signals.json"
DATASET = "logs/dataset.csv"
HOURLY_FILE = "state/hourly.json"   # results grouped by the UTC hour the signal was sent     # one row per finished signal: its numbers + result (for learning)


@dataclass
class TrackedSignal:
    symbol: str
    side: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk: float               # entry-to-stop distance at the start
    opened_ms: int
    quality: str = "A"
    tp1_done: bool = False
    remaining: float = 1.0    # share of the idea still open (1.0 = 100%)
    result_r: float = 0.0     # result so far, in R
    pending: bool = False     # limit entry not filled yet
    valid_until_ms: int = 0   # limit entry expires at this time
    features: dict = field(default_factory=dict)
    shadow: bool = False      # skipped by the model: followed silently, no Telegram messages
    model_prob: float | None = None


class SignalTracker:
    def __init__(self, cfg, notifier: Telegram, journal: Journal):
        self.cfg, self.tg, self.journal = cfg, notifier, journal
        self.open: dict[str, TrackedSignal] = {}
        # {"all": {hour: [signals, total_R, wins]}, "day": {...}, "day_date": "YYYY-MM-DD"}
        self.hourly = {"all": {}, "day": {}, "day_date": ""}
        self._load()
        self._load_hourly()

    # ------------------------------------------------------------ persistence
    def _save(self) -> None:
        os.makedirs("state", exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in self.open.values()], f, indent=2)
        os.replace(tmp, STATE_FILE)

    def _load(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                for row in json.load(f):
                    s = TrackedSignal(**row)
                    self.open[s.symbol] = s
            log.info("Restored %d open signals", len(self.open))
        except Exception as e:
            log.error("Could not read %s: %s", STATE_FILE, e)

    # --------------------------------------------------------------- signals
    def has_open(self, symbol: str) -> bool:
        return symbol in self.open

    def real_open(self) -> int:
        return sum(1 for x in self.open.values() if not x.shadow)

    async def _send(self, s: "TrackedSignal", text: str) -> None:
        if not s.shadow:
            await self.tg.send(text)

    async def publish(self, sig, volume_24h: float = 0.0, shadow: bool = False,
                      model_prob: float | None = None) -> None:
        """Send the signal to Telegram and start following it."""
        if self.has_open(sig.symbol):
            return
        self.open[sig.symbol] = TrackedSignal(
            sig.symbol, sig.side, sig.entry, sig.sl, sig.tp1, sig.tp2, sig.risk, now_ms(), sig.quality,
            pending=sig.limit, valid_until_ms=now_ms() + int(self.cfg.limit_valid_min * 60_000),
            features=dict(sig.features), shadow=shadow, model_prob=model_prob)
        self._save()
        if shadow:
            self.journal.write("shadow_signal", symbol=sig.symbol, side=sig.side, entry=sig.entry,
                               model_prob=model_prob)
            return
        self.journal.write("signal", symbol=sig.symbol, side=sig.side, quality=sig.quality, entry=sig.entry,
                           sl=sig.sl, tp1=sig.tp1, tp2=sig.tp2, volume_24h=round(volume_24h), reasons=sig.reasons)
        await self.tg.send(signal_message(sig, self.cfg, volume_24h))

    # ------------------------------------------------------- price following
    async def on_tick(self, st: SymbolState) -> None:
        s = self.open.get(st.symbol)
        if not s:
            return
        hi = st.tick_high if st.tick_high is not None else st.last_price
        lo = st.tick_low if st.tick_low is not None else st.last_price
        if not hi or not lo:
            return
        long_ = s.side == "LONG"
        if s.pending:
            filled = lo <= s.entry if long_ else hi >= s.entry
            if not filled:
                if now_ms() > s.valid_until_ms:
                    self.open.pop(s.symbol, None)
                    self.journal.write("not_filled", symbol=s.symbol, side=s.side, entry=s.entry)
                    self._record(s, "not_filled")
                    self._save()
                    await self._send(s, not_filled_message(s, self.cfg.limit_valid_min))
                return
            s.pending = False
            self.journal.write("filled", symbol=s.symbol, side=s.side, entry=s.entry)
            self._save()
        sl_hit = lo <= s.sl if long_ else hi >= s.sl
        tp1_hit = hi >= s.tp1 if long_ else lo <= s.tp1
        tp2_hit = hi >= s.tp2 if long_ else lo <= s.tp2

        if sl_hit:
            reason = "BE" if s.tp1_done and s.sl == s.entry else "SL"
            await self._close_part(s, s.remaining, s.sl, reason)
            return
        if not s.tp1_done and tp1_hit:
            s.tp1_done = True
            share = min(self.cfg.tp1_close_pct / 100, s.remaining)
            moved_be = self.cfg.move_sl_to_be and s.remaining - share > 1e-9
            await self._close_part(s, share, s.tp1, "TP1", moved_be)
            if moved_be and s.symbol in self.open:
                s.sl = s.entry
                self._save()
        if s.symbol in self.open and s.tp1_done and tp2_hit:
            await self._close_part(s, s.remaining, s.tp2, "TP2")

    async def _close_part(self, s: TrackedSignal, share: float, price: float, reason: str,
                          moved_be: bool = False) -> None:
        sign = 1 if s.side == "LONG" else -1
        r = (price - s.entry) * sign / s.risk if s.risk else 0.0
        s.result_r += r * share
        s.remaining -= share
        self.journal.write("result", symbol=s.symbol, side=s.side, reason=reason, price=price,
                           share=round(share, 3), r=round(r, 2))
        await self._send(s, exit_message(s, reason, share, price, r, moved_be))
        if s.remaining <= 1e-9:
            self.open.pop(s.symbol, None)
            self.journal.write("signal_closed", symbol=s.symbol, side=s.side, total_r=round(s.result_r, 2))
            self._record(s, reason)
            if not s.shadow:
                self._add_hourly(s)
            await self._send(s, closed_message(s))
        self._save()

    # ------------------------------------------------------- learning data
    def _record(self, s: TrackedSignal, outcome: str) -> None:
        """Append one finished signal (its numbers + result) to logs/dataset.csv."""
        if not s.features:          # signals from before this version have no recorded numbers
            return
        row = {"time_utc": s.opened_ms, "symbol": s.symbol, **s.features,
               "outcome": outcome, "result_r": round(s.result_r, 3) if outcome != "not_filled" else "",
               "shadow": int(s.shadow), "model_prob": "" if s.model_prob is None else round(s.model_prob, 3)}
        try:
            os.makedirs(os.path.dirname(DATASET), exist_ok=True)
            header, old_rows = [], []
            if os.path.exists(DATASET):
                with open(DATASET, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    header = list(reader.fieldnames or [])
                    if any(k not in header for k in row):
                        old_rows = list(reader)          # new columns appeared: rewrite with a wider header
            if not header or old_rows or any(k not in header for k in row):
                header = header + [k for k in row if k not in header]
                with open(DATASET, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(old_rows)
                    w.writerow(row)
            else:
                with open(DATASET, "a", encoding="utf-8", newline="") as f:
                    csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)
        except Exception as e:
            log.warning("dataset write failed: %s", e)

    # ------------------------------------------------------ results by hour
    def _load_hourly(self) -> None:
        try:
            if os.path.exists(HOURLY_FILE):
                with open(HOURLY_FILE, encoding="utf-8") as f:
                    self.hourly.update(json.load(f))
        except Exception as e:
            log.warning("could not read %s: %s", HOURLY_FILE, e)

    def _save_hourly(self) -> None:
        os.makedirs("state", exist_ok=True)
        with open(HOURLY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.hourly, f)

    def _add_hourly(self, s: TrackedSignal) -> None:
        hour = str(time.gmtime(s.opened_ms / 1000).tm_hour)
        for key in ("all", "day"):
            row = self.hourly[key].setdefault(hour, [0, 0.0, 0])
            row[0] += 1
            row[1] = round(row[1] + s.result_r, 3)
            row[2] += 1 if s.result_r > 0 else 0
        self._save_hourly()

    def best_worst(self, key: str = "all", n: int = 2) -> tuple[list, list]:
        rows = [(int(h), v) for h, v in self.hourly.get(key, {}).items() if v[0] >= 3]
        rows.sort(key=lambda x: x[1][1], reverse=True)
        return rows[:n], rows[::-1][:n]

    def take_day(self) -> dict:
        """Returns the finished day's hourly results and starts a new day."""
        day = self.hourly.get("day", {})
        self.hourly["day"] = {}
        self._save_hourly()
        return day
