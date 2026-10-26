"""
Strategy: Smart Money Liquidity Sweep + Order Block / FVG Reversal (whale-aligned).

Bias:   4H market structure (last break of a confirmed swing = BOS/CHoCH). Never trade against it.
        Daily structure is shown as extra information.
Setup (LONG; SHORT is the exact mirror image), on 1H candles:
  1. Sweep: a 1H candle wicks below a fresh swing low (equal lows / 1H swing low / 4H swing low)
     and closes back above it (liquidity taken, then rejected).
  2. Displacement: within SMC_DISP_BARS 1H candles, a strong bullish candle
     (body >= SMC_DISP_ATR x 1H ATR, close in the top 30% of its range).
  3. Zone: Order Block = last bearish 1H candle before the displacement; Fair Value Gap = gap between
     the candle before and the candle after the displacement. With an FVG the entry zone is its
     50-70% level (consequent encroachment), otherwise the Order Block.
  4. Retest: price taps the zone within SMC_RETEST_HOURS (checked on 15m candles).
  5. Confirmation: a bullish CHoCH/BOS on the 15m chart after the tap.
Confluence score (out of 5): 4H structure + clean sweep + fresh OB/FVG (these three form the setup),
  + 15m confirmation, + order flow (aggressive buyers >= TAKER_MIN_SHARE of the displacement volume).
  A signal is sent when the score reaches SMC_MIN_SCORE: at the tap if already enough, otherwise
  at the 15m confirmation.
Risk: stop just beyond the sweep extreme / Order Block, TP1 = 2R (1:2), TP2 = 3R (1:3).
Invalidation: a close below the sweep low (LONG) / above the sweep high (SHORT).
"""
import logging

from .indicators import atr, market_structure
from .state import SymbolState
from .strategy import MIN_MS, Signal, Strategy
from .utils import C, H, L, O, T, TB, V, fmt_price, utc_str

log = logging.getLogger("strategy")
HOUR = 60 * MIN_MS


def pivots(candles: list, n: int, highs: bool) -> list:
    """Indexes of confirmed swing highs (highs=True) or swing lows within `candles`."""
    out = []
    for i in range(n, len(candles) - n):
        win = candles[i - n:i + n + 1]
        if highs and candles[i][H] == max(x[H] for x in win):
            out.append(i)
        if not highs and candles[i][L] == min(x[L] for x in win):
            out.append(i)
    return out


class SmcStrategy(Strategy):
    def structure(self, st: SymbolState, tf: str, pivot: int = 3) -> str | None:
        cs = list(st.candles[tf])[-200:]
        if len(cs) < 4 * pivot + 5:
            return None
        return market_structure(cs, pivot)[0]

    def trend_4h(self, st: SymbolState) -> str | None:        # used by the status report
        s = self.structure(st, "4h")
        return "UP" if s == "bull" else "DOWN" if s == "bear" else None

    def direction(self, st: SymbolState, now_ms: int | None = None):
        s = self.structure(st, "4h")
        if s is None:
            return None, "no clear 4H structure", None
        side = "LONG" if s == "bull" else "SHORT"
        return side, "", side

    # ---------------------------------------------------------------- main
    def evaluate(self, st: SymbolState, now_ms: int) -> Signal | None:
        h1 = list(st.candles["1h"])
        m15 = list(st.candles["15m"])
        if len(h1) < 80 or len(m15) < 40:
            return None
        for i in [k for k in range(len(h1)) if h1[k][T] > st.smc_1h_checked][-3:]:
            self._on_1h(st, h1, i)
        st.smc_1h_checked = h1[-1][T]
        new15 = [k for k in range(len(m15)) if m15[k][T] > st.htf_checked][-3:]
        st.htf_checked = m15[-1][T]
        for k in new15:
            self.stats["candles"] += 1
            sig = self._on_15m(st, m15, k, h1, now_ms)
            if sig:
                return sig
        return None

    # ----------------------------------------------------------- 1H events
    def _on_1h(self, st: SymbolState, h1: list, i: int) -> None:
        cfg = self.cfg
        c = h1[i]
        a = atr([x[H] for x in h1[max(0, i - 60):i + 1]], [x[L] for x in h1[max(0, i - 60):i + 1]],
                [x[C] for x in h1[max(0, i - 60):i + 1]], 14)
        if a <= 0:
            return
        side, _, _ = self.direction(st)
        s = st.setup
        if s and side != s["side"]:
            self.stats["cancelled_trend"] += 1
            st.setup = s = None
        if side is None:
            return
        long_ = side == "LONG"
        if s:
            # invalidation: close back beyond the sweep extreme
            if (c[C] < s["sweep_ext"]) if long_ else (c[C] > s["sweep_ext"]):
                self.stats["failed"] += 1
                st.setup = None
                return
            if s["phase"] in ("retest", "confirm") and c[T] > s["deadline"]:
                self.stats["expired"] += 1
                st.setup = None
                return
            if s["phase"] == "disp":
                body = (c[C] - c[O]) if long_ else (c[O] - c[C])
                rng = c[H] - c[L]
                pos = ((c[C] - c[L]) / rng if long_ else (c[H] - c[C]) / rng) if rng > 0 else 0
                if body >= cfg.smc_disp_atr * a and pos >= 0.7:
                    ob = None
                    for k in range(i - 1, max(-1, i - 6), -1):
                        x = h1[k]
                        if (x[C] < x[O]) if long_ else (x[C] > x[O]):
                            ob = (x[L], x[H])
                            break
                    buy = c[TB] / c[V] if len(c) > TB and c[V] > 0 else None
                    s.update(phase="fvg", disp_i=i, disp_ts=c[T], ob=ob,
                             flow=None if buy is None else (buy if long_ else 1 - buy), atr=a)
                    self.stats["displacements"] += 1
                elif i - s["sweep_i"] >= cfg.smc_disp_bars:
                    self.stats["no_displacement"] += 1
                    st.setup = None
                return
            if s["phase"] == "fvg":
                d = s["disp_i"]
                if i > d + 1:                                   # candle missed: drop the setup
                    st.setup = None
                    return
                if i == d + 1:
                    before = h1[d - 1]
                    fvg = None
                    if long_ and before[H] < c[L]:
                        fvg = (before[H], c[L])                    # (bottom, top)
                    if not long_ and before[L] > c[H]:
                        fvg = (c[H], before[L])
                    if fvg:
                        size = fvg[1] - fvg[0]
                        zone = ((fvg[1] - 0.7 * size, fvg[1] - 0.5 * size) if long_
                                else (fvg[0] + 0.5 * size, fvg[0] + 0.7 * size))
                        label = (f"{'Bullish' if long_ else 'Bearish'} FVG {fmt_price(fvg[0])} – {fmt_price(fvg[1])}, "
                                 f"entry at its 50–70% level")
                    elif s["ob"]:
                        zone = s["ob"]
                        label = f"{'Bullish' if long_ else 'Bearish'} Order Block {fmt_price(zone[0])} – {fmt_price(zone[1])}"
                    else:
                        st.setup = None
                        return
                    if s["ob"] and fvg:
                        label += f" (Order Block {fmt_price(s['ob'][0])} – {fmt_price(s['ob'][1])} below)" if long_ \
                            else f" (Order Block {fmt_price(s['ob'][0])} – {fmt_price(s['ob'][1])} above)"
                    s.update(phase="retest", zone=zone, zone_label=label, fvg=fvg,
                             deadline=c[T] + int(cfg.smc_retest_hours * HOUR))
                return
            return

        # ---- no setup: look for a sweep of fresh liquidity on this candle
        look = h1[max(0, i - 72):i]
        base = max(0, i - 72)
        idx = pivots(look, 3, highs=not long_)
        best = None
        for p in idx:
            lvl = look[p][L] if long_ else look[p][H]
            later = look[p + 1:]
            fresh = all((x[L] >= lvl) if long_ else (x[H] <= lvl) for x in later)
            if not fresh:
                continue
            swept = (c[L] < lvl < c[C]) if long_ else (c[H] > lvl > c[C])
            if not swept:
                continue
            equal = sum(1 for q in idx if q != p and abs((look[q][L] if long_ else look[q][H]) - lvl) <= 0.15 * a)
            score = 1 + equal
            if best is None or score > best[0]:
                best = (score, lvl, base + p, equal)
        if not best:
            return
        _, lvl, p_i, equal = best
        h4 = list(st.candles["4h"])[-60:]
        four = any(abs((h4[k][L] if long_ else h4[k][H]) - lvl) <= 0.3 * a for k in pivots(h4, 2, highs=not long_))
        kind = "lows" if long_ else "highs"
        label = (f"4H swing {kind[:-1]}" if four else f"equal {kind}" if equal else f"1H swing {kind[:-1]}")
        st.setup = {"side": side, "phase": "disp", "sweep_i": i, "sweep_ts": c[T], "level": lvl,
                    "sweep_ext": c[L] if long_ else c[H], "sweep_label": label}
        self.stats["crosses"] += 1                                  # = sweeps found
        # the sweep candle itself can be the displacement (strong rejection candle)
        self._on_1h_disp_same(st, h1, i, a)

    def _on_1h_disp_same(self, st, h1, i, a):
        s = st.setup
        c = h1[i]
        long_ = s["side"] == "LONG"
        body = (c[C] - c[O]) if long_ else (c[O] - c[C])
        rng = c[H] - c[L]
        pos = ((c[C] - c[L]) / rng if long_ else (c[H] - c[C]) / rng) if rng > 0 else 0
        if body >= self.cfg.smc_disp_atr * a and pos >= 0.7:
            ob = None
            for k in range(i - 1, max(-1, i - 6), -1):
                x = h1[k]
                if (x[C] < x[O]) if long_ else (x[C] > x[O]):
                    ob = (x[L], x[H])
                    break
            buy = c[TB] / c[V] if len(c) > TB and c[V] > 0 else None
            s.update(phase="fvg", disp_i=i, disp_ts=c[T], ob=ob,
                     flow=None if buy is None else (buy if long_ else 1 - buy), atr=a)
            self.stats["displacements"] += 1

    # ---------------------------------------------------------- 15m events
    def _on_15m(self, st, m15, k, h1, now_ms):
        cfg = self.cfg
        s = st.setup
        if not s or s["phase"] not in ("retest", "confirm"):
            return None
        c = m15[k]
        long_ = s["side"] == "LONG"
        if (c[C] < s["sweep_ext"]) if long_ else (c[C] > s["sweep_ext"]):
            self.stats["failed"] += 1
            st.setup = None
            return None
        base_score = 3 + (1 if s.get("flow") is not None and s["flow"] >= cfg.taker_min_share else 0)
        if s["phase"] == "retest":
            if c[T] < s["disp_ts"] + 2 * HOUR:          # the zone is complete only after the next 1H candle
                return None
            tapped = c[L] <= s["zone"][1] if long_ else c[H] >= s["zone"][0]
            if not tapped:
                return None
            self.stats["retests"] += 1
            s["tap_ts"] = c[T]
            if base_score >= cfg.smc_min_score:
                return self._signal_smc(st, c, s, base_score, "entered on the zone tap (no 15m confirmation yet)", now_ms)
            s["phase"] = "confirm"
            s["confirm_deadline"] = c[T] + int(cfg.smc_confirm_hours * HOUR)
            return None
        if c[T] > s["confirm_deadline"]:
            self.stats["expired"] += 1
            st.setup = None
            return None
        # 15m CHoCH / BOS: a close through the most recent confirmed 15m swing high (LONG) / low (SHORT)
        window = [x for x in m15[:k + 1] if x[T] >= s["tap_ts"] - 6 * HOUR]
        if len(window) < 8:
            return None
        piv = pivots(window[:-1], 2, highs=long_)
        if not piv:
            return None
        level = window[piv[-1]][H] if long_ else window[piv[-1]][L]
        prev = window[-2]
        broke = (prev[C] <= level < c[C]) if long_ else (prev[C] >= level > c[C])
        if broke:
            word = "bullish" if long_ else "bearish"
            return self._signal_smc(st, c, s, base_score + 1,
                                    f"15m {word} break of structure at {fmt_price(level)} "
                                    f"({utc_str(c[T] + 15 * MIN_MS)[11:16]} UTC) after the tap", now_ms)
        return None

    # -------------------------------------------------------------- signal
    def _signal_smc(self, st, c, s, score, confirm_text, now_ms):
        cfg = self.cfg
        long_ = s["side"] == "LONG"
        sign = 1 if long_ else -1
        entry = c[C]
        buf = 0.05 * s.get("atr", 0)
        ext = s["sweep_ext"]
        if s.get("ob"):
            ext = min(ext, s["ob"][0]) if long_ else max(ext, s["ob"][1])
        sl = ext - sign * buf
        min_dist = entry * cfg.min_sl_pct / 100
        if (entry - sl) * sign < min_dist:
            sl = entry - sign * min_dist
        st.setup = None
        risk_pct = abs(entry - sl) / entry * 100
        if risk_pct > cfg.htf_max_sl_pct:
            self.stats["stop_too_wide"] += 1
            return None
        risk = abs(entry - sl)
        tp1, tp2 = entry + sign * cfg.smc_tp1_r * risk, entry + sign * cfg.smc_tp2_r * risk
        word = "bullish" if long_ else "bearish"
        d1 = self.structure(st, "1d")
        daily = "bullish" if d1 == "bull" else "bearish" if d1 == "bear" else "unclear"
        flow = s.get("flow")
        who = "buyers" if long_ else "sellers"
        extra = [f"aggressive {who} {flow * 100:.0f}% of the displacement volume" if flow is not None
                 else "order flow data not available"]
        card = {
            "zone": (min(s["zone"]), max(s["zone"])),
            "score": score,
            "reasons": [
                f"4H structure: {word} (Daily: {daily})",
                f"Sweep level: {s['sweep_label']} at {fmt_price(s['level'])}, swept to {fmt_price(s['sweep_ext'])} "
                f"at {utc_str(s['sweep_ts'])[11:16]} UTC and rejected",
                f"Order Block / FVG: {s['zone_label']}",
                f"Confirmation: {confirm_text}",
                "Extra filters: " + "; ".join(extra),
            ],
            "invalidation": f"Close {'below' if long_ else 'above'} {fmt_price(s['sweep_ext'])}",
        }
        self.stats["signals"] += 1
        sig = Signal(st.symbol, st.ccxt_symbol, s["side"], entry, sl, tp1, tp2,
                     "A+" if score >= 5 else "A", [], f"TP1 {cfg.smc_tp1_r:g}R, TP2 {cfg.smc_tp2_r:g}R",
                     now_ms, False, entry, {"side": s["side"], "score": score, "sl_pct": round(risk_pct, 3)})
        sig.card = card
        return sig
