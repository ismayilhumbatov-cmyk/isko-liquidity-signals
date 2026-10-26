"""
Backtest: replays the bot's exact strategy on real Binance history and compares versions.

  A  current strategy
  C  + fresh EMA 3 cross (crossed EMA 9 and 21 within the last 3 candles)
  D  C with LONG switched off (SHORT only)
  E  C with LONG only in a strong market (at least half of the tested coins in a 4H uptrend)
  (D and E are taken from the same C replay: each LONG trade is tagged with the market state.)

Run on the server (the live bot keeps running):
    cd /opt/liquidity-sweep-bot
    .venv/bin/python backtest.py          7 days, 30 coins (about 10 min)
    .venv/bin/python backtest.py 90       90 days, 20 coins (about 30-40 min)
    .venv/bin/python backtest.py 30 range 5-minute high/low break strategy, 30 days
    .venv/bin/python backtest.py 90 htf   1H high/low break with 15m entry, 90 days
    .venv/bin/python backtest.py 30 orb   New York opening range break, 30 days
    .venv/bin/python backtest.py 90 smc   Smart Money liquidity sweep + OB/FVG, 90 days

What is simulated: 4H trend, 15m check, 1H structure shift, BTC shift pause, LONG only with BTC 4H up,
1m cross/retest/big candle, volume and buyers/sellers share, limit entry (3 min to fill),
stop, TP1 (50%) with stop to entry, TP2, 30 min cooldown per coin.
Not simulated (no reliable history): open interest / funding / long-short check, the learning model.
These affect all versions in the same way.
Money result: 50 USDT x 10 = 500 USDT per trade; fees 0.02% limit entry and TP exits, 0.05% stop exits.
"""
import asyncio
import os
import sys
import time

import ccxt.async_support as ccxt

from bot.config import Config
from bot.notifier import Telegram
from bot.state import HISTORY, SymbolState
from bot.strategy import Strategy
from bot.strategy_range import HtfRangeStrategy, OrbStrategy, RangeStrategy
from bot.strategy_smc import SmcStrategy
from bot.symbols import select_symbols
from bot.utils import C, H, L, T, TF_MS

DAYS = float(os.getenv("BT_DAYS", "7"))
COINS = int(os.getenv("BT_COINS", "0"))     # 0 = automatic: 30 coins up to 30 days, 20 coins beyond
POSITION = 500.0
MAKER, TAKER = 0.0002, 0.0005
VARIANTS = [("A", "Current strategy", "off"),
            ("C", "+ fresh EMA 3 cross", "fresh")]
STRONG_BREADTH = 0.5
STRATEGY = "ema"          # "range" = the 5-minute high/low break strategy


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


async def klines(ex, symbol, tf, start, end):
    out, cur, step = [], start, TF_MS[tf]
    while cur < end:
        raw = await ex.fapiPublicGetKlines({"symbol": symbol, "interval": tf, "startTime": cur,
                                            "endTime": end, "limit": 1500})
        if not raw:
            break
        for r in raw:
            out.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])])
        cur = int(raw[-1][0]) + step
        await asyncio.sleep(0.6)            # stay well inside Binance limits (the live bot shares them)
    return [c for c in out if c[T] + step <= end]


async def download(ex, symbol, start, end, have_4h=None):
    data = {"4h": have_4h} if have_4h is not None else {}
    for tf in ("1m", "15m", "1h", "4h", "1d"):
        if tf in data:
            continue
        warm = (HISTORY[tf] + 5) * TF_MS[tf]
        data[tf] = await klines(ex, symbol, tf, start - warm, end)
    return data


def outcome(sig, minutes, k, cfg):
    """Follows one signal over the next 1m candles. Returns (result, R, USDT, exit_index)."""
    long_ = sig.side == "LONG"
    sign = 1 if long_ else -1
    entry, sl, tp1, tp2 = sig.entry, sig.sl, sig.tp1, sig.tp2
    j, n = k + 1, len(minutes)
    # limit entry must fill within LIMIT_VALID_MIN candles
    filled = not sig.limit
    while not filled and j < n and j <= k + cfg.limit_valid_min:
        c = minutes[j]
        if (c[L] <= entry) if long_ else (c[H] >= entry):
            filled = True
            break
        j += 1
    if not filled:
        return ("not_filled", 0.0, 0.0, min(j, n - 1)) if j < n else None
    half = POSITION / 2
    fees = POSITION * (MAKER if sig.limit else TAKER)
    risk = abs(entry - sl) or 1e-12
    r1, r2 = abs(tp1 - entry) / risk, abs(tp2 - entry) / risk
    usdt, tp1_done, stop = 0.0, False, sl
    while j < n:
        c = minutes[j]
        hit_stop = (c[L] <= stop) if long_ else (c[H] >= stop)
        hit_tp1 = (c[H] >= tp1) if long_ else (c[L] <= tp1)
        hit_tp2 = (c[H] >= tp2) if long_ else (c[L] <= tp2)
        if hit_stop:                                      # careful: stop counts first
            size = half if tp1_done else POSITION
            usdt += size * (stop - entry) / entry * sign - size * TAKER
            r = 0.5 * r1 if tp1_done else -1.0
            return ("BE" if tp1_done else "SL", r, usdt - fees, j)
        if not tp1_done and hit_tp1:
            tp1_done = True
            usdt += half * (tp1 - entry) / entry * sign - half * MAKER
            stop = entry
        if tp1_done and hit_tp2:
            usdt += half * (tp2 - entry) / entry * sign - half * MAKER
            return ("TP2", 0.5 * r1 + 0.5 * r2, usdt - fees, j)
        j += 1
    return None                                           # still open when the data ends


class Replay:
    """Feeds history into SymbolState objects minute by minute, exactly as the live bot sees it."""

    def __init__(self, sym, data, start):
        self.st = SymbolState(sym, sym)
        self.data = data
        self.ptr = {}
        for tf in ("15m", "1h", "4h", "1d"):
            arr = data[tf]
            i = 0
            while i < len(arr) and arr[i][T] + TF_MS[tf] <= start:
                self.st.candles[tf].append(arr[i])
                i += 1
            self.ptr[tf] = i
        m = data["1m"]
        i = 0
        while i < len(m) and m[i][T] + TF_MS["1m"] <= start:
            i += 1
        for c in m[max(0, i - HISTORY["1m"]):i]:
            self.st.candles["1m"].append(c)
        self.ptr["1m"] = i
        self.st.last_closed_checked = self.st.candles["1m"][-1][T] if self.st.candles["1m"] else 0

    def advance(self, t_close):
        """Adds every candle that has closed by t_close."""
        for tf in ("1m", "15m", "1h", "4h", "1d"):
            arr, i = self.data[tf], self.ptr[tf]
            while i < len(arr) and arr[i][T] + TF_MS[tf] <= t_close:
                self.st.candles[tf].append(arr[i])
                i += 1
            self.ptr[tf] = i


def cached(strategy):
    """Speed-up: 4H/15m/1H results only change when those candles close."""
    cache = {}
    orig = {"trend_4h": strategy.trend_4h, "check_15m": strategy.check_15m, "structure_1h": strategy.structure_1h}

    def key(st, tf):
        return (st.symbol, tf, st.candles[tf][-1][T] if st.candles[tf] else 0)

    def memo(k, fn):
        if k not in cache:
            cache[k] = fn()
        return cache[k]

    strategy.trend_4h = lambda st: memo(key(st, "4h") + ("t",), lambda: orig["trend_4h"](st))
    strategy.check_15m = lambda st, side: memo(key(st, "15m") + (side,), lambda: orig["check_15m"](st, side))
    strategy.structure_1h = lambda st: memo(key(st, "1h") + ("s",), lambda: orig["structure_1h"](st))
    if hasattr(strategy, "trend_tf"):
        ttf = strategy.trend_tf
        strategy.trend_tf = lambda st, tf: memo(key(st, tf) + ("tf",), lambda: ttf(st, tf))
    if hasattr(strategy, "trend_1h"):
        t1 = strategy.trend_1h
        strategy.trend_1h = lambda st: memo(key(st, "1h") + ("t1",), lambda: t1(st))
    return strategy


def breadth_timeline(all_4h, cfg):
    """Share of tested coins in a 4H uptrend, for every 4H close time."""
    from bot.indicators import ema_series
    n = cfg.trend_slope_bars
    votes = {}                                   # close_time -> [up, total]
    for arr in all_4h.values():
        closes = [c[C] for c in arr]
        ema = ema_series(closes, cfg.trend_ema)
        for i, c in enumerate(arr):
            t = c[T] + TF_MS["4h"]
            v = votes.setdefault(t, [0, 0])
            v[1] += 1
            if i >= n and ema[i] is not None and ema[i - n] is not None and closes[i] > ema[i] > ema[i - n]:
                v[0] += 1
    times = sorted(votes)
    return times, [votes[t][0] / votes[t][1] for t in times]


def breadth_at(tl, t):
    import bisect
    times, vals = tl
    i = bisect.bisect_right(times, t) - 1
    return vals[i] if i >= 0 else None


def run_coin(sym, data, btc_data, start, end, mode, tl=None):
    cfg = Config()
    if STRATEGY == "smc":
        cfg.smc_min_score = int(mode)
        cfg.cooldown_min = cfg.smc_cooldown_min
        strategy = cached(SmcStrategy(cfg))
    elif STRATEGY == "orb":
        cfg.orb_range_min = float(mode)
        strategy = cached(OrbStrategy(cfg))
    elif STRATEGY == "htf":
        cfg.htf_level_bars = int(mode)
        cfg.cooldown_min = cfg.htf_cooldown_min
        strategy = cached(HtfRangeStrategy(cfg))
    elif STRATEGY == "range":
        cfg.range_5m_bars = int(mode)
        strategy = cached(RangeStrategy(cfg))
    else:
        cfg.confirm_ema_mode = mode
        strategy = cached(Strategy(cfg))
    coin = Replay(sym, data, start)
    btc = coin if sym == "BTCUSDT" else Replay("BTCUSDT", btc_data, start)
    strategy.btc = btc.st
    minutes = data["1m"]
    trades, busy_until = [], 0
    k = coin.ptr["1m"]
    while k < len(minutes):
        t_close = minutes[k][T] + TF_MS["1m"]
        if t_close > end:
            break
        coin.advance(t_close)
        if btc is not coin:
            btc.advance(t_close)
        if t_close < busy_until:
            coin.st.setup = None
            coin.st.last_closed_checked = minutes[k][T]
            if coin.st.candles["15m"]:
                coin.st.htf_checked = coin.st.candles["15m"][-1][T]
            if coin.st.candles["1h"]:
                coin.st.smc_1h_checked = coin.st.candles["1h"][-1][T]
            k += 1
            continue
        sig = strategy.evaluate(coin.st, t_close)
        if sig:
            res = outcome(sig, minutes, k, cfg)
            if res:
                kind, r, usdt, j = res
                b = breadth_at(tl, t_close) if tl else None
                trades.append({"symbol": sym, "side": sig.side, "kind": kind, "r": r, "usdt": usdt, "time": t_close,
                               "sl_pct": abs(sig.entry - sig.sl) / sig.entry * 100,
                               "strong": b is not None and b >= STRONG_BREADTH})
                busy_until = max(minutes[j][T] + TF_MS["1m"], t_close + cfg.cooldown_min * 60_000)
            else:
                busy_until = end + 1
        k += 1
    return trades


def summary(trades):
    done = [t for t in trades if t["kind"] != "not_filled"]
    n = len(done)
    if not n:
        return {"n": 0, "nf": len(trades), "tp1": 0, "r": 0, "rps": 0, "usdt": 0, "long": 0, "short": 0}
    return {
        "n": n, "nf": len(trades) - n,
        "tp1": sum(1 for t in done if t["kind"] != "SL") / n * 100,
        "r": sum(t["r"] for t in done), "rps": sum(t["r"] for t in done) / n,
        "usdt": sum(t["usdt"] for t in done),
        "long": sum(t["usdt"] for t in done if t["side"] == "LONG"),
        "short": sum(t["usdt"] for t in done if t["side"] == "SHORT"),
    }


async def main():
    cfg = Config()
    ex = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
    tg = Telegram(cfg.telegram_token, cfg.telegram_chat_id)
    try:
        pairs = await select_symbols(ex, cfg)
        coins = COINS or (30 if DAYS <= 30 else 20)
        ids = list(pairs)[:coins]
        if "BTCUSDT" not in ids:
            ids = ["BTCUSDT"] + ids[:-1]
        end = int(time.time() * 1000) // 60_000 * 60_000
        start = end - int(DAYS * 86_400_000)
        t0 = time.time()
        log(f"Backtest: {len(ids)} coins, last {DAYS:g} days.")
        log("Step 1/2: 4H history of all coins (for the market-strength check)...")
        all_4h = {}
        for sym in ids:
            try:
                all_4h[sym] = await klines(ex, sym, "4h", start - (HISTORY["4h"] + 5) * TF_MS["4h"], end)
            except Exception as e:
                log(f"{sym}: 4H download failed ({e}), skipped")
        tl = breadth_timeline(all_4h, cfg)
        log("Step 2/2: BTC, then each coin (download + replay)...")
        btc_data = await download(ex, "BTCUSDT", start, end, all_4h.get("BTCUSDT"))
        variants = {"range": [("R1", "Lines = last 1 five-minute candle", "1"),
                              ("R3", "Lines = last 3 five-minute candles", "3")],
                    "htf": [("H1", "Lines = last 1-hour candle", "1"),
                            ("H4", "Lines = last 4 hours", "4")],
                    "orb": [("O5", "Lines = first 5 min after New York open", "5"),
                            ("O15", "Lines = first 15 min after New York open", "15")],
                    "smc": [("S3", "Minimum score 3/5", "3"),
                            ("S4", "Minimum score 4/5", "4"),
                            ("S5", "Score 5/5 only", "5")]}.get(STRATEGY, VARIANTS)
        results = {v[0]: [] for v in variants}
        done_coins = 0
        for n, sym in enumerate(ids, 1):
            if sym not in all_4h:
                continue
            try:
                data = btc_data if sym == "BTCUSDT" else await download(ex, sym, start, end, all_4h[sym])
            except Exception as e:
                log(f"{sym}: download failed ({e}), skipped")
                continue
            line = []
            for code, _, mode in variants:
                tr = run_coin(sym, data, btc_data, start, end, mode, tl)
                results[code] += tr
                line.append(f"{code}:{len([t for t in tr if t['kind'] != 'not_filled'])}")
            done_coins += 1
            left = (time.time() - t0) / n * (len(ids) - n) / 60
            log(f"[{n}/{len(ids)}] {sym}  trades {' '.join(line)}  (about {left:.0f} min left)")
            del data

        if STRATEGY in ("range", "htf", "orb", "smc"):
            rows = {code: (name, summary(results[code])) for code, name, _ in variants}
            pools = {code: results[code] for code, _, _ in variants}
        else:
            c = results["C"]
            rows = {
                "A": ("Current strategy", summary(results["A"])),
                "C": ("+ fresh EMA 3 cross (LONG on)", summary(c)),
                "D": ("C, SHORT only (LONG off)", summary([t for t in c if t["side"] == "SHORT"])),
                "E": ("C, LONG only in a strong market", summary([t for t in c if t["side"] == "SHORT" or t["strong"]])),
            }
            pools = {"A": results["A"], "C": c, "D": [t for t in c if t["side"] == "SHORT"],
                     "E": [t for t in c if t["side"] == "SHORT" or t["strong"]]}
        title = {"range": "5m high/low break strategy",
                 "htf": "1H high/low break, 15m entry (Daily + 4H trend)",
                 "orb": "New York opening range break (1H + 4H trend, 1m entry)",
                 "smc": "Smart Money liquidity sweep + OB/FVG (4H bias, 1H setup, 15m confirmation)"}.get(STRATEGY, "EMA retest strategy")
        text = [f"🧪 <b>Backtest — {title}, last {DAYS:g} days, {done_coins} coins</b>",
                "500 USDT per trade, fees included (market entry 0.05%)." if STRATEGY in ("range", "htf", "orb", "smc")
                else "500 USDT per trade, fees included. OI/funding check and model not included.", ""]
        if STRATEGY not in ("range", "htf", "orb", "smc"):
            text.insert(2, f"Strong market = at least {STRONG_BREADTH * 100:.0f}% of tested coins in a 4H uptrend.")
        for code, (name, s_) in rows.items():
            per_day = s_["usdt"] / DAYS
            text.append(f"<b>{code}. {name}</b>")
            text.append(f"Trades {s_['n']} (+{s_['nf']} not filled) • reached TP1 {s_['tp1']:.0f}%")
            text.append(f"Result {s_['r']:+.1f}R ({s_['rps']:+.2f}R per trade) • <b>{s_['usdt']:+.2f} USDT</b> "
                        f"({per_day:+.2f} per day)")
            text.append(f"LONG {s_['long']:+.2f} • SHORT {s_['short']:+.2f} USDT")
            text.append("")
        # period by period for the best version, to see if it holds in different markets
        best = max(rows, key=lambda k: rows[k][1]["usdt"])
        period = 30 if DAYS > 30 else 7
        if DAYS >= 2 * period:
            pool = pools[best]
            text.append(f"<b>{best} by {period}-day period:</b>")
            for k in range(int(DAYS // period)):
                lo, hi = start + k * period * 86_400_000, start + (k + 1) * period * 86_400_000
                part = [t for t in pool if lo <= t.get("time", 0) < hi and t["kind"] != "not_filled"]
                text.append(f"Days {k * period + 1}–{(k + 1) * period}: {len(part)} trades, "
                            f"{sum(t['usdt'] for t in part):+.2f} USDT")
            text.append("")
        text.append(f"Best in USDT: <b>{best}</b>  •  run time {(time.time() - t0) / 60:.0f} min")
        msg = "\n".join(text)
        print("\n" + msg.replace("<b>", "").replace("</b>", ""))
        await tg.send(msg)
    finally:
        await ex.close()
        await tg.close()


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg.isdigit():
            DAYS = float(arg)
        elif arg.lower() in ("range", "ema", "htf", "orb", "smc"):
            STRATEGY = arg.lower()
    asyncio.run(main())
