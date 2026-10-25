"""
Configuration.

Values come from (last one wins):
  1. the defaults written below
  2. .env on the server        - Telegram token and chat ID (secret, never on GitHub)
  3. settings.env from GitHub  - strategy numbers (edit it on GitHub, then re-run the installer)

This is a SIGNAL-ONLY bot: it never places orders and needs no Binance keys.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()
_SETTINGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.env")
load_dotenv(_SETTINGS, override=True)


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(float(_str(name, str(default))))


def _bool(name: str, default: bool) -> bool:
    return _str(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str = "") -> list:
    return [x.strip().upper() for x in _str(name, default).split(",") if x.strip()]


@dataclass
class Config:
    # ---------------------------------------------------------- market data
    ws_url: str = field(default_factory=lambda: _str("WS_URL", ""))
    data_source: str = field(default_factory=lambda: _str("DATA_SOURCE", "ws").lower())   # ws | rest
    rest_poll_sec: float = field(default_factory=lambda: _float("REST_POLL_SEC", 10))

    # -------------------------------------------------------------- telegram
    telegram_token: str = field(default_factory=lambda: _str("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str = field(default_factory=lambda: _str("TELEGRAM_CHAT_ID"))

    # ----------------------------------------------------------------- coins
    scan_all: bool = field(default_factory=lambda: _bool("SCAN_ALL", True))
    min_24h_volume: float = field(default_factory=lambda: _float("MIN_24H_VOLUME_USDT", 3_000_000))
    pair_refresh_min: float = field(default_factory=lambda: _float("PAIR_REFRESH_MIN", 60))
    max_pairs: int = field(default_factory=lambda: _int("MAX_PAIRS", 0))
    pairs: list = field(default_factory=lambda: _list("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT"))
    exclude_pairs: list = field(default_factory=lambda: _list("EXCLUDE_PAIRS", ""))

    # which strategy the live bot runs: ema (1m EMA 50 retest) | range (5m high/low break)
    strategy: str = field(default_factory=lambda: _str("STRATEGY", "ema").lower())
    range_5m_bars: int = field(default_factory=lambda: _int("RANGE_5M_BARS", 1))
    range_vol_mult: float = field(default_factory=lambda: _float("RANGE_VOL_MULT", 1.5))
    # "htf" strategy: the same idea one level higher (Daily+4H trend, 1H lines, 15m entry)
    htf_level_bars: int = field(default_factory=lambda: _int("HTF_LEVEL_BARS", 1))
    htf_retest_window_min: float = field(default_factory=lambda: _float("HTF_RETEST_WINDOW_MIN", 180))
    htf_min_sl_pct: float = field(default_factory=lambda: _float("HTF_MIN_SL_PCT", 0.5))
    htf_max_sl_pct: float = field(default_factory=lambda: _float("HTF_MAX_SL_PCT", 5.0))
    htf_cooldown_min: float = field(default_factory=lambda: _float("HTF_COOLDOWN_MIN", 120))

    # -------------------------------------------------- step 1: 4H trend
    trend_ema: int = field(default_factory=lambda: _int("TREND_EMA", 50))
    trend_slope_bars: int = field(default_factory=lambda: _int("TREND_SLOPE_BARS", 3))

    # ------------------------------------------------ step 2: 15m check
    use_15m_filter: bool = field(default_factory=lambda: _bool("USE_15M_FILTER", True))
    mid_ema: int = field(default_factory=lambda: _int("MID_EMA", 50))

    # --------------------------------------- market structure shift filters
    use_shift_filter: bool = field(default_factory=lambda: _bool("USE_SHIFT_FILTER", True))
    shift_pivot: int = field(default_factory=lambda: _int("SHIFT_PIVOT", 3))
    use_btc_filter: bool = field(default_factory=lambda: _bool("USE_BTC_FILTER", True))
    btc_pause_min: float = field(default_factory=lambda: _float("BTC_PAUSE_MIN", 60))

    long_needs_btc_up: bool = field(default_factory=lambda: _bool("LONG_NEEDS_BTC_UP", True))
    long_mode: str = field(default_factory=lambda: _str("LONG_MODE", "on").lower())          # on | off | strong
    long_strong_breadth: float = field(default_factory=lambda: _float("LONG_STRONG_BREADTH", 0.5))

    # --------------------------------------- order flow / positioning filters
    taker_min_share: float = field(default_factory=lambda: _float("TAKER_MIN_SHARE", 0.55))
    use_oi_filter: bool = field(default_factory=lambda: _bool("USE_OI_FILTER", True))
    oi_window_min: int = field(default_factory=lambda: _int("OI_WINDOW_MIN", 30))
    oi_min_change_pct: float = field(default_factory=lambda: _float("OI_MIN_CHANGE_PCT", 0.0))
    funding_extreme_pct: float = field(default_factory=lambda: _float("FUNDING_EXTREME_PCT", 0.05))
    funding_strong_pct: float = field(default_factory=lambda: _float("FUNDING_STRONG_PCT", 0.03))
    ls_crowded_short: float = field(default_factory=lambda: _float("LS_CROWDED_SHORT", 0.8))
    ls_crowded_long: float = field(default_factory=lambda: _float("LS_CROWDED_LONG", 2.5))

    # ------------------------------------------------------- limit entry
    use_limit_entry: bool = field(default_factory=lambda: _bool("USE_LIMIT_ENTRY", True))
    limit_offset_atr: float = field(default_factory=lambda: _float("LIMIT_OFFSET_ATR", 0.1))
    limit_valid_min: float = field(default_factory=lambda: _float("LIMIT_VALID_MIN", 3))

    # ------------------------------------------- step 3: 1m cross/retest
    entry_ema: int = field(default_factory=lambda: _int("ENTRY_EMA", 50))
    retest_window_min: float = field(default_factory=lambda: _float("RETEST_WINDOW_MIN", 15))
    touch_tolerance_atr: float = field(default_factory=lambda: _float("TOUCH_TOLERANCE_ATR", 0.1))
    close_tolerance_atr: float = field(default_factory=lambda: _float("CLOSE_TOLERANCE_ATR", 0.25))
    # extra confirmation on the big candle: off | align (EMA3>EMA9>EMA21) | fresh (align + EMA3 crossed in last 3 candles)
    confirm_ema_mode: str = field(default_factory=lambda: _str("CONFIRM_EMA_MODE", "off").lower())
    confirm_ema_fast: int = field(default_factory=lambda: _int("CONFIRM_EMA_FAST", 3))
    confirm_ema_mid: int = field(default_factory=lambda: _int("CONFIRM_EMA_MID", 9))
    confirm_ema_slow: int = field(default_factory=lambda: _int("CONFIRM_EMA_SLOW", 21))
    big_candle_mult: float = field(default_factory=lambda: _float("BIG_CANDLE_MULT", 1.5))
    confirm_vol_mult: float = field(default_factory=lambda: _float("CONFIRM_VOL_MULT", 1.0))

    # ---------------------------------------------- self-learning model
    use_learning: bool = field(default_factory=lambda: _bool("USE_LEARNING", True))
    learn_min_signals: int = field(default_factory=lambda: _int("LEARN_MIN_SIGNALS", 500))
    learn_every_days: float = field(default_factory=lambda: _float("LEARN_EVERY_DAYS", 7))
    learn_test_days: float = field(default_factory=lambda: _float("LEARN_TEST_DAYS", 7))
    learn_min_test: int = field(default_factory=lambda: _int("LEARN_MIN_TEST", 100))
    learn_min_gain_r: float = field(default_factory=lambda: _float("LEARN_MIN_GAIN_R", 0.1))
    learn_min_keep: float = field(default_factory=lambda: _float("LEARN_MIN_KEEP", 0.25))

    # ------------------------------------------------ stop / targets
    sl_buffer_atr: float = field(default_factory=lambda: _float("SL_BUFFER_ATR", 0.1))
    min_sl_pct: float = field(default_factory=lambda: _float("MIN_SL_PCT", 0.3))
    max_sl_pct: float = field(default_factory=lambda: _float("MAX_SL_PCT", 3.0))
    tp1_r: float = field(default_factory=lambda: _float("TP1_R", 1.5))
    tp2_r: float = field(default_factory=lambda: _float("TP2_R", 3.0))
    tp1_close_pct: float = field(default_factory=lambda: _float("TP1_CLOSE_PCT", 50))
    move_sl_to_be: bool = field(default_factory=lambda: _bool("MOVE_SL_TO_BE_AFTER_TP1", True))

    # ---------------------------------------------- shown in the message
    risk_pct: float = field(default_factory=lambda: _float("RISK_PCT", 1.0))
    leverage: int = field(default_factory=lambda: _int("LEVERAGE", 10))

    # --------------------------------------------------------------- runtime
    cooldown_min: float = field(default_factory=lambda: _float("COOLDOWN_MINUTES", 30))
    eval_interval_sec: float = field(default_factory=lambda: _float("EVAL_INTERVAL_SEC", 2))
    status_hours: float = field(default_factory=lambda: _float("STATUS_HOURS", 6))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO").upper())

    def __post_init__(self):
        if self.tp2_r <= self.tp1_r:
            raise ValueError("TP2_R must be larger than TP1_R")
        if not 0 < self.tp1_close_pct <= 100:
            raise ValueError("TP1_CLOSE_PCT must be between 1 and 100")
