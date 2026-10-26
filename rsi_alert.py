"""
RSI alert bot (information only, no trade levels).

Every 5 minutes, right after the 5-minute candle closes, it checks all Binance USDT-M futures coins:

  🔴 OVERBOUGHT : 5-minute RSI(14) > RSI_OB_5M
                  (plus 4H RSI(14) > RSI_OB_4H only if RSI_OB_4H is above 0; 0 = 4H not required)
  🟢 OVERSOLD   : 5-minute RSI(14) below a level that depends on the coin's market cap
                    market cap < MCAP_SMALL (50M)          -> RSI < RSI_OS_SMALL (20)
                    MCAP_SMALL .. MCAP_LARGE (50M-100M)    -> RSI < RSI_OS_MID (25)
                    market cap > MCAP_LARGE (100M)         -> RSI < RSI_OS_LARGE (30)
                    market cap unknown (very small coins)  -> the strictest level (RSI_OS_SMALL)

When several coins qualify at the same time, the most extreme one is sent first
(how far RSI is beyond its level). A coin is announced once when it enters a zone; it can be announced
again only after its 5-minute RSI has moved back out by RSI_RESET_GAP points.

RSI uses Wilder's smoothing over 200 candles, so it matches Binance and TradingView.
The 5-minute RSI uses the candle that just closed; the 4H RSI includes the current 4H candle.
Market caps come from CoinGecko's public API (refreshed every 6 hours).
"""
import asyncio
import logging
import time

import aiohttp

from .indicators import rsi_series
from .symbols import select_symbols
from .utils import fmt_price, fmt_usd, utc_str

log = logging.getLogger("rsi")
COINGECKO = "https://api.coingecko.com/api/v3/coins/markets"


def base_of(symbol: str) -> str:
    """BTCUSDT -> BTC, 1000PEPEUSDT -> PEPE, 1000000MOGUSDT -> MOG."""
    b = symbol[:-4] if symbol.endswith("USDT") else symbol
    for prefix in ("1000000", "10000", "1000", "1M"):
        if b.startswith(prefix) and len(b) > len(prefix):
            return b[len(prefix):]
    return b


class RsiAlerter:
    def __init__(self, cfg, exchange, telegram):
        self.cfg, self.ex, self.tg = cfg, exchange, telegram
        self.caps: dict[str, float] = {}       # base symbol -> market cap in USD
        self.caps_ts = 0.0
        self.pairs: dict = {}
        self.pairs_ts = 0.0
        self.active: dict[tuple, bool] = {}    # (symbol, "OB"/"OS") -> currently in the zone (already announced)
        self.session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------ market caps
    async def refresh_caps(self) -> None:
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        headers = {"accept": "application/json"}
        if self.cfg.coingecko_api_key:
            headers["x-cg-demo-api-key"] = self.cfg.coingecko_api_key
        caps: dict[str, float] = {}
        for page in range(1, self.cfg.mcap_pages + 1):
            for attempt in range(4):
                try:
                    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page}
                    async with self.session.get(COINGECKO, params=params, headers=headers) as r:
                        if r.status == 429:
                            await asyncio.sleep(30 * (attempt + 1))
                            continue
                        rows = await r.json(content_type=None)
                    for row in rows or []:
                        sym, cap = str(row.get("symbol", "")).upper(), row.get("market_cap") or 0
                        if sym and cap and cap > caps.get(sym, 0):     # same ticker: keep the biggest coin
                            caps[sym] = float(cap)
                    break
                except Exception as e:
                    log.warning("CoinGecko page %d failed: %s", page, e)
                    await asyncio.sleep(10)
            await asyncio.sleep(6)                                   # stay inside the free rate limit
        if caps:
            self.caps, self.caps_ts = caps, time.time()
            log.info("Market caps loaded for %d coins", len(caps))

    def cap_of(self, symbol: str) -> float | None:
        return self.caps.get(base_of(symbol))

    def os_level(self, cap: float | None) -> tuple[float, str]:
        cfg = self.cfg
        if cap is None:
            return cfg.rsi_os_small, "market cap unknown (small coin)"
        if cap < cfg.mcap_small:
            return cfg.rsi_os_small, f"market cap below {fmt_usd(cfg.mcap_small)}"
        if cap <= cfg.mcap_large:
            return cfg.rsi_os_mid, f"market cap {fmt_usd(cfg.mcap_small)}–{fmt_usd(cfg.mcap_large)}"
        return cfg.rsi_os_large, f"market cap above {fmt_usd(cfg.mcap_large)}"

    # ----------------------------------------------------------------- scan
    async def _klines(self, symbol: str, tf: str, limit: int) -> list:
        raw = await self.ex.fapiPublicGetKlines({"symbol": symbol, "interval": tf, "limit": limit})
        return [(int(r[0]), float(r[4])) for r in raw]           # (open time, close)

    async def _rsi(self, symbol: str, now_ms: int):
        m5 = await self._klines(symbol, "5m", 200)
        h4 = await self._klines(symbol, "4h", 200)
        m5 = [c for c in m5 if c[0] + 300_000 <= now_ms]          # closed 5m candles only
        if len(m5) < 50 or len(h4) < 50:
            return None
        r5 = float(rsi_series([c[1] for c in m5], 14)[-1])
        r4 = float(rsi_series([c[1] for c in h4], 14)[-1])        # includes the current 4H candle
        return r5, r4

    async def scan(self) -> None:
        now = int(time.time() * 1000)
        try:
            tickers = {t["symbol"]: t for t in await self.ex.fapiPublicGetTicker24hr()}
        except Exception:
            tickers = {}
        symbols = list(self.pairs)
        results: dict = {}
        sem = asyncio.Semaphore(8)

        async def one(sym):
            async with sem:
                try:
                    results[sym] = await self._rsi(sym, now)
                except Exception as e:
                    log.debug("%s: %s", sym, e)

        await asyncio.gather(*(one(s) for s in symbols))
        cfg = self.cfg
        alerts = []
        for sym, val in results.items():
            if not val:
                continue
            r5, r4 = val
            cap = self.cap_of(sym)
            # ---- overbought
            key = (sym, "OB")
            need_4h = cfg.rsi_ob_4h > 0
            if r5 > cfg.rsi_ob_5m and (not need_4h or r4 > cfg.rsi_ob_4h):
                if not self.active.get(key):
                    self.active[key] = True
                    alerts.append((r5 - cfg.rsi_ob_5m, "OB", sym, r5, r4, cap, None))
            elif r5 < cfg.rsi_ob_5m - cfg.rsi_reset_gap:
                self.active[key] = False
            # ---- oversold
            level, why = self.os_level(cap)
            key = (sym, "OS")
            if r5 < level:
                if not self.active.get(key):
                    self.active[key] = True
                    alerts.append((level - r5, "OS", sym, r5, r4, cap, (level, why)))
            elif r5 > level + cfg.rsi_reset_gap:
                self.active[key] = False

        alerts.sort(key=lambda a: -a[0])                          # most extreme first
        log.info("Scan: %d coins, %d new alerts", len(results), len(alerts))
        for k, (_, kind, sym, r5, r4, cap, os_info) in enumerate(alerts):
            if k >= cfg.rsi_max_alerts:
                rest = alerts[k:]
                await self.tg.send("➕ More coins in the zone: " + ", ".join(
                    f"#{a[2]} {a[3]:.1f}" for a in rest))
                break
            await self.tg.send(self.message(kind, sym, r5, r4, cap, os_info, tickers.get(sym, {}), now))
            await asyncio.sleep(1.1)

    def message(self, kind, sym, r5, r4, cap, os_info, tk, now) -> str:
        price = float(tk.get("lastPrice") or 0)
        chg = float(tk.get("priceChangePercent") or 0)
        vol = float(tk.get("quoteVolume") or 0)
        cap_txt = fmt_usd(cap) if cap else "unknown"
        if kind == "OB":
            head = f"🔴 <b>RSI OVERBOUGHT — #{sym}</b>"
            rsi = (f"RSI 5m: <b>{r5:.1f}</b> (above {self.cfg.rsi_ob_5m:g}) • RSI 4H: {r4:.1f}"
                   + (f" (above {self.cfg.rsi_ob_4h:g})" if self.cfg.rsi_ob_4h > 0 else ""))
        else:
            level, why = os_info
            head = f"🟢 <b>RSI OVERSOLD — #{sym}</b>"
            rsi = f"RSI 5m: <b>{r5:.1f}</b> (below {level:g}, {why}) • RSI 4H: {r4:.1f}"
        return "\n".join([
            head, rsi,
            f"Price: <code>{fmt_price(price)}</code> • 24h change: {chg:+.2f}%",
            f"Market cap: {cap_txt} • 24h futures volume: {fmt_usd(vol)} USDT",
            f"🕒 {utc_str(now)}",
        ])

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        self.pairs = await select_symbols(self.ex, self.cfg)
        self.pairs_ts = time.time()
        await self.refresh_caps()
        known = sum(1 for s in self.pairs if self.cap_of(s))
        await self.tg.send(
            f"🤖 <b>RSI alert bot started</b>\n"
            f"Watching {len(self.pairs)} futures coins ({known} with market cap data)\n"
            f"🔴 Overbought: RSI 5m > {self.cfg.rsi_ob_5m:g}"
            + (f" and RSI 4H > {self.cfg.rsi_ob_4h:g}" if self.cfg.rsi_ob_4h > 0 else "") + "\n"
            f"🟢 Oversold (RSI 5m): below {self.cfg.rsi_os_small:g} under {fmt_usd(self.cfg.mcap_small)} • "
            f"below {self.cfg.rsi_os_mid:g} for {fmt_usd(self.cfg.mcap_small)}–{fmt_usd(self.cfg.mcap_large)} • "
            f"below {self.cfg.rsi_os_large:g} above {fmt_usd(self.cfg.mcap_large)}\n"
            f"Most extreme coin first • information only")
        while True:
            now = time.time()
            nxt = (now // 300 + 1) * 300 + 4                         # 4 seconds after each 5m close
            await asyncio.sleep(nxt - now)
            try:
                if time.time() - self.pairs_ts > 3600:
                    self.pairs = await select_symbols(self.ex, self.cfg)
                    self.pairs_ts = time.time()
                if time.time() - self.caps_ts > 6 * 3600:
                    asyncio.create_task(self.refresh_caps())
                await self.scan()
            except Exception:
                log.exception("RSI scan failed")

    async def close(self):
        if self.session:
            await self.session.close()
