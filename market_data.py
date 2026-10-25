"""
Market data from Binance USDT-M Futures (public data, no keys).

* Start-up: loads history over REST for every coin: 300 x 1m, 200 x 15m, 200 x 1H, 300 x 4H candles.
* Live: WebSocket kline streams (1m, 15m, 1H, 4H) for every coin. Binance pushes every price
  change instantly, so a closed 1m candle reaches the bot within about a second.
* Safety net: if the WebSocket keeps failing, the bot switches to REST polling by itself.
"""
import asyncio
import json
import logging
import time

from .state import HISTORY, TIMEFRAMES, SymbolState
from .utils import C, T, TF_MS, now_ms

log = logging.getLogger("data")

MAINNET_WS = "wss://fstream.binance.com"
STREAMS_PER_CONNECTION = 150


def _push_closed(store, candle: list) -> bool:
    """Add a closed candle (or replace the same one). Returns True if it is new."""
    if store and store[-1][T] == candle[T]:
        store[-1] = candle
        return False
    if not store or candle[T] > store[-1][T]:
        store.append(candle)
        return True
    return False


class MarketFeed:
    def __init__(self, cfg, exchange, states: dict):
        self.cfg = cfg
        self.ex = exchange
        self.states: dict[str, SymbolState] = states
        self.use_rest = cfg.data_source == "rest"

    # ------------------------------------------------------------ REST
    async def _klines(self, st: SymbolState, tf: str, limit: int) -> list:
        """Binance klines incl. taker-buy volume (column 9). Falls back to plain OHLCV."""
        try:
            raw = await self.ex.fapiPublicGetKlines({"symbol": st.symbol, "interval": tf, "limit": limit})
            return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])]
                    for r in raw]
        except AttributeError:
            rows = await self.ex.fetch_ohlcv(st.ccxt_symbol, tf, limit=limit)
            return [[int(r[0]), *map(float, r[1:6])] for r in rows]

    async def _fetch(self, st: SymbolState, tf: str, limit: int) -> None:
        rows = await self._klines(st, tf, limit)
        now = now_ms()
        store = st.candles[tf]
        if limit > 10:
            store.clear()
        live = None
        for r in rows:
            candle = list(r)
            if r[0] + TF_MS[tf] <= now:
                _push_closed(store, candle)
            else:
                live = candle
        st.live[tf] = live
        if tf == "1m" and live:
            st.last_price = live[C]
            if limit <= 10:  # REST mode: only the current price is known
                st.tick_high = max(st.tick_high or live[C], live[C])
                st.tick_low = min(st.tick_low or live[C], live[C])
        st.updated = True

    async def bootstrap(self, only: list | None = None) -> None:
        """Load history for all coins (or only the given new coins)."""
        targets = only if only is not None else list(self.states.values())
        for st in targets:
            for tf in TIMEFRAMES:
                for attempt in range(3):
                    try:
                        await self._fetch(st, tf, HISTORY[tf])
                        break
                    except Exception as e:
                        log.warning("history %s %s failed (%s), retrying", st.symbol, tf, e)
                        await asyncio.sleep(2)
            st.last_closed_checked = st.candles["1m"][-1][T] if st.candles["1m"] else 0
        log.info("History loaded for %d coins", len(targets))

    async def _resync_recent(self) -> None:
        """After a reconnect, fill candles missed while disconnected."""
        for st in list(self.states.values()):
            for tf in TIMEFRAMES:
                try:
                    await self._fetch(st, tf, 5)
                except Exception as e:
                    log.warning("resync %s %s failed: %s", st.symbol, tf, e)

    # ------------------------------------------------------- WebSocket
    def _apply_kline(self, k: dict) -> None:
        st = self.states.get(k["s"])
        tf = k["i"]
        if st is None or tf not in st.candles:
            return
        candle = [int(k["t"]), float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"]),
                  float(k.get("V", 0) or 0)]
        if tf == "1m":
            price = candle[C]
            st.last_price = price
            st.tick_high = price if st.tick_high is None else max(st.tick_high, price)
            st.tick_low = price if st.tick_low is None else min(st.tick_low, price)
            st.updated = True
        if k["x"]:
            _push_closed(st.candles[tf], candle)
            st.live[tf] = None
        else:
            st.live[tf] = candle

    async def _ws_worker(self, symbols: list[str]) -> None:
        """One live connection. It connects to the short base address and then subscribes
        to the coins in small groups (a very long address with all streams is ignored by Binance).
        If no candle data arrives for 60 seconds, it reconnects; after 3 tries -> REST polling."""
        import websockets

        base = self.cfg.ws_url or MAINNET_WS
        streams = [f"{s.lower()}@kline_{tf}" for s in symbols for tf in TIMEFRAMES]
        failures, first = 0, True
        while not self.use_rest:
            try:
                async with websockets.connect(f"{base}/stream", ping_interval=20, ping_timeout=20,
                                              max_size=2 ** 22) as ws:
                    for n in range(0, len(streams), 50):
                        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams[n:n + 50], "id": n + 1}))
                        await asyncio.sleep(0.3)
                    log.info("WebSocket connected (%d coins, %d streams)", len(symbols), len(streams))
                    got_data = False
                    while True:
                        if self.use_rest:
                            return
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        if data.get("e") != "kline":
                            continue
                        self._apply_kline(data["k"])
                        if not got_data:
                            got_data, failures = True, 0
                            log.info("WebSocket data flowing (%d coins)", len(symbols))
                            if not first:
                                await self._resync_recent()
                            first = False
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failures += 1
                reason = "no data for 60s" if isinstance(e, asyncio.TimeoutError) else str(e)
                log.warning("WebSocket problem (%d): %s", failures, reason)
                if failures >= 3:
                    log.warning("WebSocket not delivering data -> switching to REST polling")
                    self.use_rest = True
                    return
                await asyncio.sleep(min(30, 2 ** failures))

    async def _rest_loop(self) -> None:
        log.info("REST polling every %ss", self.cfg.rest_poll_sec)
        last_slow = 0.0
        while True:
            started = time.time()
            refresh_slow = started - last_slow > 120
            for st in list(self.states.values()):
                try:
                    await self._fetch(st, "1m", 3)
                    if refresh_slow:
                        await self._fetch(st, "15m", 3)
                        await self._fetch(st, "1h", 3)
                        await self._fetch(st, "4h", 3)
                        await self._fetch(st, "1d", 2)
                except Exception as e:
                    log.warning("REST poll %s: %s", st.symbol, e)
            if refresh_slow:
                last_slow = started
            interval = max(self.cfg.rest_poll_sec, len(self.states) * 0.06)
            await asyncio.sleep(max(1.0, interval - (time.time() - started)))

    async def run(self) -> None:
        if not self.use_rest:
            ids = list(self.states)
            per_conn = STREAMS_PER_CONNECTION // len(TIMEFRAMES)
            chunks = [ids[i:i + per_conn] for i in range(0, len(ids), per_conn)]
            await asyncio.gather(*(self._ws_worker(c) for c in chunks))
        await self._rest_loop()
