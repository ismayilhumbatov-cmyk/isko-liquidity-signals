"""
Backtest: replays the bot's exact strategy on real Binance history and compares versions.

  A  current strategy
  B  + EMA 3/9/21 alignment on the big candle
  C  + fresh EMA 3 cross (crossed EMA 9 and 21 within the last 3 candles)

Run on the server (takes about 10-20 minutes, the live bot keeps running):
    cd /opt/liquidity-sweep-bot
    .venv/bin/python backtest.py

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
from bot.symbols import select_symbols
from bot.utils import C, H, L, T, TF_MS

DAYS = float(os.getenv("BT_DAYS", "7"))
COINS = int(os.getenv("BT_COINS", "30"))
POSITION = 500.0
MAKER, TAKER = 0.0002, 0.0005
VARIANTS = [("A", "Current strategy", "off"),
            ("B", "+ EMA 3/9/21 alignment", "align"),
            ("C", "+ fresh EMA 3 cross", "fresh")]


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


async def download(ex, symbol, start, end):
    data = {}
    for tf in ("1m", "15m", "1h", "4h"):
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
    usdt, tp1_done, stop = 0.0, False, sl
    while j < n:
        c = minutes[j]
        hit_stop = (c[L] <= stop) if long_ else (c[H] >= stop)
        hit_tp1 = (c[H] >= tp1) if long_ else (c[L] <= tp1)
        hit_tp2 = (c[H] >= tp2) if long_ else (c[L] <= tp2)
        if hit_stop:                                      # careful: stop counts first
            size = half if tp1_done else POSITION
            usdt += size * (stop - entry) / entry * sign - size * TAKER
            r = 0.75 if tp1_done else -1.0
            return ("BE" if tp1_done else "SL", r, usdt - fees, j)
        if not tp1_done and hit_tp1:
            tp1_done = True
            usdt += half * (tp1 - entry) / entry * sign - half * MAKER
            stop = entry
        if tp1_done and hit_tp2:
            usdt += half * (tp2 - entry) / entry * sign - half * MAKER
            return ("TP2", 2.25, usdt - fees, j)
        j += 1
    return None                                           # still open when the data ends


class Replay:
    """Feeds history into SymbolState objects minute by minute, exactly as the live bot sees it."""

    def __init__(self, sym, data, start):
        self.st = SymbolState(sym, sym)
        self.data = data
        self.ptr = {}
        for tf in ("15m", "1h", "4h"):
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
        for tf in ("1m", "15m", "1h", "4h"):
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
    return strategy


def run_coin(sym, data, btc_data, start, end, mode):
    cfg = Config()
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
            k += 1
            continue
        sig = strategy.evaluate(coin.st, t_close)
        if sig:
            res = outcome(sig, minutes, k, cfg)
            if res:
                kind, r, usdt, j = res
                trades.append({"symbol": sym, "side": sig.side, "kind": kind, "r": r, "usdt": usdt,
                               "sl_pct": abs(sig.entry - sig.sl) / sig.entry * 100})
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
        ids = list(pairs)[:COINS]
        if "BTCUSDT" not in ids:
            ids = ["BTCUSDT"] + ids[:-1]
        end = int(time.time() * 1000) // 60_000 * 60_000
        start = end - int(DAYS * 86_400_000)
        log(f"Backtest: {len(ids)} coins, last {DAYS:g} days. Downloading BTC first...")
        btc_data = await download(ex, "BTCUSDT", start, end)
        results = {v[0]: [] for v in VARIANTS}
        for n, sym in enumerate(ids, 1):
            try:
                data = btc_data if sym == "BTCUSDT" else await download(ex, sym, start, end)
            except Exception as e:
                log(f"{sym}: download failed ({e}), skipped")
                continue
            line = []
            for code, _, mode in VARIANTS:
                tr = run_coin(sym, data, btc_data, start, end, mode)
                results[code] += tr
                line.append(f"{code}:{len([t for t in tr if t['kind'] != 'not_filled'])}")
            log(f"[{n}/{len(ids)}] {sym}  trades {' '.join(line)}")
            del data

        rows = {code: summary(results[code]) for code, _, _ in VARIANTS}
        text = [f"🧪 <b>Backtest — last {DAYS:g} days, {len(ids)} coins</b>",
                "500 USDT per trade, fees included. OI/funding check and model not included.", ""]
        for code, name, _ in VARIANTS:
            s = rows[code]
            text.append(f"<b>{code}. {name}</b>")
            text.append(f"Trades {s['n']} (+{s['nf']} not filled) • reached TP1 {s['tp1']:.0f}%")
            text.append(f"Result {s['r']:+.1f}R ({s['rps']:+.2f}R per trade) • <b>{s['usdt']:+.2f} USDT</b>")
            text.append(f"LONG {s['long']:+.2f} • SHORT {s['short']:+.2f} USDT")
            text.append("")
        best = max(rows, key=lambda c: rows[c]["usdt"])
        text.append(f"Best in USDT: <b>{best}</b>")
        msg = "\n".join(text)
        print("\n" + msg.replace("<b>", "").replace("</b>", ""))
        await tg.send(msg)
    finally:
        await ex.close()
        await tg.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        DAYS = float(sys.argv[1])
    asyncio.run(main())
