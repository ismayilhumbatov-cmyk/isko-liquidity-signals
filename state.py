"""Live data the bot keeps for each coin."""
from collections import deque
from dataclasses import dataclass, field

TIMEFRAMES = ("1m", "15m", "1h", "4h", "1d")
HISTORY = {"1m": 300, "15m": 200, "1h": 200, "4h": 300, "1d": 100}     # candles loaded at start


def _stores() -> dict:
    return {tf: deque(maxlen=HISTORY[tf] + 50) for tf in TIMEFRAMES}


@dataclass
class SymbolState:
    symbol: str                     # Binance id, e.g. "BTCUSDT"
    ccxt_symbol: str                # ccxt id, e.g. "BTC/USDT:USDT"
    volume_24h: float = 0.0         # 24h futures volume in USDT
    candles: dict = field(default_factory=_stores)                        # closed candles per timeframe
    live: dict = field(default_factory=lambda: {tf: None for tf in TIMEFRAMES})  # candle still forming
    setup: dict | None = None       # cross waiting for retest / confirmation
    last_closed_checked: int = 0    # open time of the last closed 1m candle the strategy used
    htf_checked: int = 0
    orb_day: int = -1               # New York opening range strategy: current UTC day
    orb_signals: int = 0            # signals sent on this coin today
    orb_levels: tuple | None = None # (high, low, range start, range end)            # same, for the 15-minute entry chart of the "htf" strategy
    cooldown_until: float = 0.0     # unix seconds; no new signal on this coin before this time
    last_price: float = 0.0
    tick_high: float | None = None  # highest price since the last check (for TP / SL tracking)
    tick_low: float | None = None
    updated: bool = False
