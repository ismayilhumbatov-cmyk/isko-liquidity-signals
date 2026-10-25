"""
Strategy: 1H + 4H trend -> 5-minute high/low lines -> 1-minute break, retest and big candle.

STEP 1 - trend: 4H and 1H must agree.
  Uptrend on a timeframe = last closed candle above its EMA 50, and the EMA 50 higher than 3 candles ago.
  Both up -> LONG only.  Both down -> SHORT only.  Otherwise no signal.

STEP 2 - two lines from the 5-minute chart (built from the 1-minute candles):
  upper line = highest high of the last RANGE_5M_BARS closed 5-minute candles
  lower line = lowest low of those candles
  Price between the lines -> no signal.

STEP 3 - 1-minute entry, within RETEST_WINDOW_MIN minutes of the break:
  LONG:  a 1m candle closes above the upper line (the line is frozen at this moment)
         -> a later candle touches the line again and does not close back inside (retest)
         -> a separate big green candle: body >= BIG_CANDLE_MULT x average body,
            volume >= RANGE_VOL_MULT x average volume, closing above the line and the retest high
         -> LONG at the close.
  SHORT: the mirror image through the lower line.
  The setup is cancelled if a candle closes back inside the lines, the window passes,
  or the 1H/4H trend no longer agrees.

STEP 4 - stop just beyond the retest wick (at least MIN_SL_PCT), TP1 = TP1_R x risk (50%), TP2 = TP2_R x risk.
"""
import logging

from .indicators import atr, ema_series
from .state import SymbolState
from .strategy import MIN_MS, Signal, Strategy
from .utils import C, H, L, O, T, TB, V, fmt_price, utc_str

log = logging.getLogger("strategy")
FIVE = 5 * MIN_MS


def five_minute_levels(closed: list, i: int, bars: int) -> tuple[float | None, float | None]:
    """High/low of the last `bars` COMPLETE 5-minute candles before the one containing closed[i]."""
    current_bucket = closed[i][T] // FIVE
    buckets: dict = {}
    for k in range(i - 1, max(-1, i - 5 * (bars + 2)), -1):
        b = closed[k][T] // FIVE
        if b >= current_bucket:
            continue
        hi_lo = buckets.setdefault(b, [closed[k][H], closed[k][L], 0])
        hi_lo[0] = max(hi_lo[0], closed[k][H])
        hi_lo[1] = min(hi_lo[1], closed[k][L])
        hi_lo[2] += 1
    complete = [buckets[b] for b in sorted(buckets, reverse=True) if buckets[b][2] == 5][:bars]
    if len(complete) < bars:
        return None, None
    return max(x[0] for x in complete), min(x[1] for x in complete)


class RangeStrategy(Strategy):
    def trend_tf(self, st: SymbolState, tf: str) -> str | None:
        cs = list(st.candles[tf])
        n = self.cfg.trend_slope_bars
        if len(cs) < self.cfg.trend_ema + n + 1:
            return None
        ema = ema_series([c[C] for c in cs], self.cfg.trend_ema)
        if ema[-1] is None or ema[-1 - n] is None:
            return None
        close = cs[-1][C]
        if close > ema[-1] > ema[-1 - n]:
            return "UP"
        if close < ema[-1] < ema[-1 - n]:
            return "DOWN"
        return None

    def trend_1h(self, st: SymbolState) -> str | None:
        return self.trend_tf(st, "1h")

    def direction(self, st: SymbolState, now_ms: int | None = None):
        t4, t1 = self.trend_4h(st), self.trend_1h(st)
        if t4 is None or t1 is None:
            return None, "no clear 1H/4H trend", None
        if t4 != t1:
            return None, "1H and 4H disagree", None
        side = "LONG" if t4 == "UP" else "SHORT"
        return side, "", side

    def _step(self, st: SymbolState, closed: list, ema: list, i: int, now_ms: int) -> Signal | None:
        cfg = self.cfg
        c, prev = closed[i], closed[i - 1]
        window = closed[max(0, i - 20):i]
        a1 = atr([x[H] for x in closed[max(0, i - 60):i + 1]],
                 [x[L] for x in closed[max(0, i - 60):i + 1]],
                 [x[C] for x in closed[max(0, i - 60):i + 1]], 14)
        avg_body = sum(abs(x[C] - x[O]) for x in window) / len(window)
        avg_vol = sum(x[V] for x in window) / len(window)
        if a1 <= 0 or avg_body <= 0:
            return None
        touch_tol, close_tol = cfg.touch_tolerance_atr * a1, cfg.close_tolerance_atr * a1
        side, why_not, _ = self.direction(st, now_ms)
        s = st.setup

        if s:
            if c[T] - s["cross_ts"] > cfg.retest_window_min * MIN_MS:
                self.stats["expired"] += 1
                st.setup = s = None
            elif side != s["side"]:
                self.stats["cancelled_trend"] += 1
                st.setup = s = None

        if s:
            long_ = s["side"] == "LONG"
            line = s["level"]
            if (c[C] < line - close_tol) if long_ else (c[C] > line + close_tol):
                self.stats["failed"] += 1                 # closed back inside the lines
                st.setup = None
                return None
            touched = c[L] <= line + touch_tol if long_ else c[H] >= line - touch_tol
            body = (c[C] - c[O]) if long_ else (c[O] - c[C])
            if s["phase"] == "confirm":
                big = body >= cfg.big_candle_mult * avg_body
                vol_ok = c[V] >= cfg.range_vol_mult * avg_vol
                beyond = c[C] > max(line, s["retest_high"]) if long_ else c[C] < min(line, s["retest_low"])
                if body > 0 and big and vol_ok and beyond:
                    return self._range_signal(st, c, s, body / avg_body, c[V] / avg_vol, a1, now_ms)
                if body > 0 and big and beyond and not vol_ok:
                    self.stats["low_volume"] += 1
            if touched:
                if s["phase"] == "retest":
                    self.stats["retests"] += 1
                s["phase"] = "confirm"
                s["retest_ts"] = c[T]
                s["retest_high"], s["retest_low"] = c[H], c[L]
                s["stop_ref"] = (min(s.get("stop_ref", c[L]), c[L]) if long_
                                 else max(s.get("stop_ref", c[H]), c[H]))
            return None

        if side is None:
            return None
        hi, lo = self._levels(st, closed, i)
        if hi is None:
            return None
        if side == "LONG" and prev[C] <= hi < c[C]:
            st.setup = {"side": side, "cross_ts": c[T], "phase": "retest", "level": hi, "hi": hi, "lo": lo}
            self.stats["crosses"] += 1
        elif side == "SHORT" and prev[C] >= lo > c[C]:
            st.setup = {"side": side, "cross_ts": c[T], "phase": "retest", "level": lo, "hi": hi, "lo": lo}
            self.stats["crosses"] += 1
        return None

    def _levels(self, st, closed, i):
        return five_minute_levels(closed, i, self.cfg.range_5m_bars)

    def level_label(self, s) -> str:
        n = self.cfg.range_5m_bars
        return (f"5m lines: high {fmt_price(s['hi'])} • low {fmt_price(s['lo'])}"
                + (f" (last {n} candles)" if n > 1 else ""))

    def _range_signal(self, st, c, s, body_x, vol_x, a1, now_ms) -> Signal | None:
        cfg = self.cfg
        long_ = s["side"] == "LONG"
        sign = 1 if long_ else -1
        entry = c[C]
        sl = s["stop_ref"] - sign * cfg.sl_buffer_atr * a1
        min_dist = entry * cfg.min_sl_pct / 100
        if (entry - sl) * sign < min_dist:
            sl = entry - sign * min_dist
        st.setup = None
        risk_pct = abs(entry - sl) / entry * 100
        if risk_pct > cfg.max_sl_pct:
            self.stats["stop_too_wide"] += 1
            return None
        risk = abs(entry - sl)
        trend = "uptrend" if long_ else "downtrend"
        word, colour = (("above the upper line", "green") if long_ else ("below the lower line", "red"))
        reasons = [
            f"4H and 1H {trend}",
            self.level_label(s),
            f"1m: broke {word} at {utc_str(s['cross_ts'])[11:16]} UTC, retest at {utc_str(s['retest_ts'])[11:16]} UTC",
            f"Big {colour} candle: body {body_x:.1f}x average, volume {vol_x:.1f}x average",
        ]
        self.stats["signals"] += 1
        quality = "A+" if body_x >= 2.5 and vol_x >= 2.0 else "A"
        return Signal(st.symbol, st.ccxt_symbol, s["side"], entry, sl,
                      entry + sign * cfg.tp1_r * risk, entry + sign * cfg.tp2_r * risk,
                      quality, reasons, f"TP1 {cfg.tp1_r:g}R, TP2 {cfg.tp2_r:g}R", now_ms,
                      False, entry, {"side": s["side"], "body_x": round(body_x, 2), "vol_x": round(vol_x, 2),
                                     "sl_pct": round(risk_pct, 3)})


class HtfRangeStrategy(RangeStrategy):
    """Same idea one level higher:
    Daily + 4H trend -> high/low of the last HTF_LEVEL_BARS closed 1-hour candles ->
    15-minute close through the line, retest, separate big 15m candle with volume (within 3 hours)."""
    ENTRY_TF, LEVEL_TF, TREND_TFS = "15m", "1h", ("1d", "4h")

    def direction(self, st: SymbolState, now_ms: int | None = None):
        ta, tb = (self.trend_tf(st, tf) for tf in self.TREND_TFS)
        if ta is None or tb is None:
            return None, "no clear Daily/4H trend", None
        if ta != tb:
            return None, "Daily and 4H disagree", None
        side = "LONG" if ta == "UP" else "SHORT"
        return side, "", side

    def levels(self, st: SymbolState, entry_candle: list) -> tuple[float | None, float | None]:
        """High/low of the last N 1-hour candles that closed before this 15m candle started."""
        done = [c for c in st.candles[self.LEVEL_TF] if c[T] + 3_600_000 <= entry_candle[T]]
        n = self.cfg.htf_level_bars
        if len(done) < n:
            return None, None
        last = done[-n:]
        return max(c[H] for c in last), min(c[L] for c in last)

    def evaluate(self, st: SymbolState, now_ms: int) -> Signal | None:
        closed = list(st.candles[self.ENTRY_TF])
        if len(closed) < 40:
            return None
        new = [c for c in closed if c[T] > st.htf_checked]
        if not new:
            return None
        st.htf_checked = closed[-1][T]
        new = new[-3:]
        self.stats["candles"] += len(new)
        for c in new:
            i = next(k for k in range(len(closed) - 1, -1, -1) if closed[k][T] == c[T])
            sig = self._htf_step(st, closed, i, now_ms)
            if sig:
                return sig
        return None

    def _htf_step(self, st, closed, i, now_ms):
        cfg = self.cfg
        c, prev = closed[i], closed[i - 1]
        window = closed[max(0, i - 20):i]
        a = atr([x[H] for x in closed[max(0, i - 60):i + 1]],
                [x[L] for x in closed[max(0, i - 60):i + 1]],
                [x[C] for x in closed[max(0, i - 60):i + 1]], 14)
        avg_body = sum(abs(x[C] - x[O]) for x in window) / len(window)
        avg_vol = sum(x[V] for x in window) / len(window)
        if a <= 0 or avg_body <= 0:
            return None
        touch_tol, close_tol = cfg.touch_tolerance_atr * a, cfg.close_tolerance_atr * a
        side, _, _ = self.direction(st, now_ms)
        s = st.setup
        if s:
            if c[T] - s["cross_ts"] > cfg.htf_retest_window_min * MIN_MS:
                self.stats["expired"] += 1
                st.setup = s = None
            elif side != s["side"]:
                self.stats["cancelled_trend"] += 1
                st.setup = s = None
        if s:
            long_ = s["side"] == "LONG"
            line = s["level"]
            if (c[C] < line - close_tol) if long_ else (c[C] > line + close_tol):
                self.stats["failed"] += 1
                st.setup = None
                return None
            touched = c[L] <= line + touch_tol if long_ else c[H] >= line - touch_tol
            body = (c[C] - c[O]) if long_ else (c[O] - c[C])
            if s["phase"] == "confirm":
                big = body >= cfg.big_candle_mult * avg_body
                vol_ok = c[V] >= cfg.range_vol_mult * avg_vol
                beyond = c[C] > max(line, s["retest_high"]) if long_ else c[C] < min(line, s["retest_low"])
                if body > 0 and big and vol_ok and beyond:
                    return self._htf_signal(st, c, s, body / avg_body, c[V] / avg_vol, a, now_ms)
                if body > 0 and big and beyond and not vol_ok:
                    self.stats["low_volume"] += 1
            if touched:
                if s["phase"] == "retest":
                    self.stats["retests"] += 1
                s["phase"] = "confirm"
                s["retest_ts"] = c[T]
                s["retest_high"], s["retest_low"] = c[H], c[L]
                s["stop_ref"] = (min(s.get("stop_ref", c[L]), c[L]) if long_
                                 else max(s.get("stop_ref", c[H]), c[H]))
            return None
        if side is None:
            return None
        hi, lo = self.levels(st, c)
        if hi is None:
            return None
        if side == "LONG" and prev[C] <= hi < c[C]:
            st.setup = {"side": side, "cross_ts": c[T], "phase": "retest", "level": hi, "hi": hi, "lo": lo}
            self.stats["crosses"] += 1
        elif side == "SHORT" and prev[C] >= lo > c[C]:
            st.setup = {"side": side, "cross_ts": c[T], "phase": "retest", "level": lo, "hi": hi, "lo": lo}
            self.stats["crosses"] += 1
        return None

    def _htf_signal(self, st, c, s, body_x, vol_x, a, now_ms):
        cfg = self.cfg
        long_ = s["side"] == "LONG"
        sign = 1 if long_ else -1
        entry = c[C]
        sl = s["stop_ref"] - sign * cfg.sl_buffer_atr * a
        min_dist = entry * cfg.htf_min_sl_pct / 100
        if (entry - sl) * sign < min_dist:
            sl = entry - sign * min_dist
        st.setup = None
        risk_pct = abs(entry - sl) / entry * 100
        if risk_pct > cfg.htf_max_sl_pct:
            self.stats["stop_too_wide"] += 1
            return None
        risk = abs(entry - sl)
        trend = "uptrend" if long_ else "downtrend"
        word, colour = ("above the 1H high", "green") if long_ else ("below the 1H low", "red")
        n = cfg.htf_level_bars
        reasons = [
            f"Daily and 4H {trend}",
            f"1H lines: high {fmt_price(s['hi'])} • low {fmt_price(s['lo'])}" + (f" (last {n} hours)" if n > 1 else ""),
            f"15m: broke {word} at {utc_str(s['cross_ts'])[11:16]} UTC, retest at {utc_str(s['retest_ts'])[11:16]} UTC",
            f"Big {colour} 15m candle: body {body_x:.1f}x average, volume {vol_x:.1f}x average",
        ]
        self.stats["signals"] += 1
        quality = "A+" if body_x >= 2.5 and vol_x >= 2.0 else "A"
        return Signal(st.symbol, st.ccxt_symbol, s["side"], entry, sl,
                      entry + sign * cfg.tp1_r * risk, entry + sign * cfg.tp2_r * risk,
                      quality, reasons, f"TP1 {cfg.tp1_r:g}R, TP2 {cfg.tp2_r:g}R", now_ms,
                      False, entry, {"side": s["side"], "body_x": round(body_x, 2), "vol_x": round(vol_x, 2),
                                     "sl_pct": round(risk_pct, 3)})



# ---------------------------------------------------------------- New York open
def ny_open_utc_ms(day_ms: int, hour: int = 9, minute: int = 30) -> int:
    """UTC time of hh:mm New York time on the UTC date of day_ms (US summer time handled:
    from the 2nd Sunday of March to the 1st Sunday of November New York is UTC-4, otherwise UTC-5)."""
    import datetime as dt
    d = dt.datetime.utcfromtimestamp(day_ms / 1000).date()

    def nth_sunday(year, month, n):
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))

    summer = nth_sunday(d.year, 3, 2) <= d < nth_sunday(d.year, 11, 1)
    offset = 4 if summer else 5
    t = dt.datetime(d.year, d.month, d.day, hour, minute) + dt.timedelta(hours=offset)
    return int(t.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


class OrbStrategy(RangeStrategy):
    """New York opening range: the lines are the high and low of the first ORB_RANGE_MIN minutes
    after 9:30 New York time. 1H + 4H trend, then 1m break, retest, big candle with volume.
    Signals only during ORB_WINDOW_MIN minutes after the range, at most ORB_MAX_SIGNALS per coin per day,
    weekdays only (ORB_WEEKDAYS_ONLY)."""

    def _orb(self, st, t_ms):
        cfg = self.cfg
        day = t_ms // 86_400_000
        start = ny_open_utc_ms(day * 86_400_000, cfg.orb_open_hour, cfg.orb_open_minute)
        end = start + int(cfg.orb_range_min * MIN_MS)
        return day, start, end

    def _levels(self, st, closed, i):
        cfg = self.cfg
        c = closed[i]
        day, start, end = self._orb(st, c[T])
        if cfg.orb_weekdays_only and ((day + 3) % 7) >= 5:       # Saturday / Sunday
            return None, None
        if not (end <= c[T] < end + cfg.orb_window_min * MIN_MS):
            return None, None
        if st.orb_day != day:
            st.orb_day, st.orb_signals, st.orb_levels = day, 0, None
        if st.orb_signals >= cfg.orb_max_signals:
            return None, None
        if st.orb_levels is None:
            rng = [x for x in closed[max(0, i - 400):i] if start <= x[T] < end]
            if len(rng) < int(cfg.orb_range_min):
                return None, None                                  # range candles missing
            st.orb_levels = (max(x[H] for x in rng), min(x[L] for x in rng), start, end)
        return st.orb_levels[0], st.orb_levels[1]

    def level_label(self, s) -> str:
        lv = s.get("orb") or ()
        when = (f" ({utc_str(lv[2])[11:16]}–{utc_str(lv[3])[11:16]} UTC)" if len(lv) == 4 else "")
        return f"New York open range{when}: high {fmt_price(s['hi'])} • low {fmt_price(s['lo'])}"

    def _step(self, st, closed, ema, i, now_ms):
        had = st.setup
        sig = super()._step(st, closed, ema, i, now_ms)
        if st.setup is not None and had is None and st.orb_levels:
            st.setup["orb"] = st.orb_levels
        if sig:
            st.orb_signals += 1
        return sig
