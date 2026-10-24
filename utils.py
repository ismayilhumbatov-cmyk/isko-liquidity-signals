"""Small shared helpers: candle field positions, formatting, logging, trade journal."""
import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# A candle is a list: [open_time_ms, open, high, low, close, volume, taker_buy_volume]
# (taker_buy_volume = volume bought with market orders; may be missing on old data)
T, O, H, L, C, V, TB = range(7)
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}

log = logging.getLogger("bot")


def now_ms() -> int:
    return int(time.time() * 1000)


def utc_str(ms: int | None = None) -> str:
    ts = (ms if ms is not None else now_ms()) / 1000
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fmt_price(p: float | None) -> str:
    """Readable price with sensible decimals for any coin size."""
    if p is None:
        return "-"
    a = abs(p)
    d = 2 if a >= 1000 else 3 if a >= 10 else 4 if a >= 1 else 5 if a >= 0.01 else 8
    return f"{p:,.{d}f}"


def fmt_usd(v: float) -> str:
    """Short money format: 1.25B, 154.2M, 850K."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= size:
            return f"{v / size:,.1f}{unit}"
    return f"{v:,.0f}"


def pct(a: float, b: float) -> float:
    """Percent change from a to b."""
    return (b - a) / a * 100 if a else 0.0


def setup_logging(level: str = "INFO") -> None:
    os.makedirs("logs", exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_h = RotatingFileHandler("logs/bot.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_h.setFormatter(fmt)
    root.handlers = [console, file_h]
    for noisy in ("websockets", "asyncio", "ccxt", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Journal:
    """Writes every signal / entry / exit with its reason to logs/journal.jsonl (one JSON per line)."""

    def __init__(self, path: str = "logs/journal.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.log = logging.getLogger("journal")

    def write(self, event: str, **data) -> None:
        record = {"time": utc_str(), "event": event, **data}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            self.log.error("journal write failed: %s", e)
        self.log.info("%s | %s", event, json.dumps(data, default=str))
