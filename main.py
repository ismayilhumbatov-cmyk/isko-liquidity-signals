"""
Trend + EMA 50 Retest SIGNAL bot for Binance USDT-M perpetual futures.

The bot never trades. It reads public Binance Futures data and sends signals to Telegram:
  4H trend  ->  15m check  ->  1m EMA 50 cross, retest, big candle   (🟢 LONG / 🔴 SHORT)
then reports TP1, TP2, stop loss or breakeven for every signal.

Run:  python main.py      (settings: settings.env from GitHub, Telegram details: .env)
"""
import asyncio
import html
import logging
import signal
import time

import ccxt.async_support as ccxt

from bot.config import Config
from bot.learner import Learner
from bot.market_data import MarketFeed
from bot.notifier import Telegram
from bot.state import SymbolState
from bot.strategy import Strategy
from bot.strategy_range import HtfRangeStrategy, OrbStrategy, RangeStrategy
from bot.strategy_smc import SmcStrategy
from bot.symbols import select_symbols
from bot.tracker import SignalTracker
from bot.utils import Journal, fmt_usd, now_ms, setup_logging, utc_str

log = logging.getLogger("main")
BOT_NAME = "Trend + EMA 50 Retest signal bot"


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tg = Telegram(cfg.telegram_token, cfg.telegram_chat_id)
        self.journal = Journal()
        self.ex = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
        self.strategy = {"range": RangeStrategy, "orb": OrbStrategy, "htf": HtfRangeStrategy,
                         "smc": SmcStrategy}.get(cfg.strategy, Strategy)(cfg)
        if cfg.strategy == "smc":
            cfg.cooldown_min = cfg.smc_cooldown_min          # 5 hours per coin after a signal
            if cfg.max_pairs == 0:
                cfg.max_pairs = cfg.smc_top_coins            # only the most liquid coins
        elif cfg.strategy == "htf":
            cfg.cooldown_min = cfg.htf_cooldown_min
        self.tracker = SignalTracker(cfg, self.tg, self.journal)
        self.learner = Learner(cfg)
        self.states: dict[str, SymbolState] = {}
        self.feed: MarketFeed | None = None
        self.feed_task: asyncio.Task | None = None
        self.funding: dict = {}          # symbol -> last funding rate in % (refreshed every 5 min)
        self.funding_ts = 0.0

    # ------------------------------------------------------------ coins
    async def pair_refresher(self) -> None:
        """Every PAIR_REFRESH_MIN: add coins that now have enough volume, drop coins that lost it."""
        while True:
            await asyncio.sleep(self.cfg.pair_refresh_min * 60)
            try:
                await self._refresh_pairs()
            except Exception:
                log.exception("coin list refresh failed")

    async def _refresh_pairs(self) -> None:
        fresh = await select_symbols(self.ex, self.cfg)
        for pid, (_, vol) in fresh.items():
            if pid in self.states:
                self.states[pid].volume_24h = vol
        added = [pid for pid in fresh if pid not in self.states]
        removed = [pid for pid in self.states if pid not in fresh and not self.tracker.has_open(pid)]
        if not added and not removed:
            return
        for pid in removed:
            self.states.pop(pid, None)
        new_states = []
        for pid in added:
            sym, vol = fresh[pid]
            st = SymbolState(pid, sym, volume_24h=vol)
            self.states[pid] = st
            new_states.append(st)
        await self.feed.bootstrap(new_states)
        self.strategy.btc = self.states.get("BTCUSDT")
        self._restart_feed()
        log.info("Coins updated: +%d %s | -%d %s | now %d", len(added), ",".join(added),
                 len(removed), ",".join(removed), len(self.states))

    # ------------------------------------------------------------- feed
    def _restart_feed(self) -> None:
        if self.feed_task and not self.feed_task.done():
            self.feed_task.cancel()
        self.feed_task = asyncio.create_task(self.feed.run())
        self.feed_task.add_done_callback(self._feed_stopped)

    def _feed_stopped(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        log.error("Data feed stopped: %s - restarting in 10s", task.exception())
        asyncio.get_running_loop().call_later(10, self._restart_feed)

    # --------------------------------------------------------- watchdog
    def newest_1m_age_min(self) -> float:
        newest = max((st.candles["1m"][-1][0] for st in self.states.values() if st.candles["1m"]), default=0)
        # a 1m candle closes 1 minute after it opens
        return (now_ms() - newest - 60_000) / 60_000 if newest else 999.0

    async def data_watchdog(self) -> None:
        """If no fresh 1m candle arrives for 3 minutes, switch to REST polling and tell the user."""
        await asyncio.sleep(180)
        warned = False
        while True:
            await asyncio.sleep(60)
            try:
                age = self.newest_1m_age_min()
                if age > 3 and self.feed and not self.feed.use_rest:
                    log.warning("No fresh 1m candles for %.0f min -> REST polling", age)
                    self.feed.use_rest = True
                    self._restart_feed()
                    await self.tg.send("⚠️ Live data stopped. Switched to direct price checks every ~15 seconds. "
                                       "Signals keep coming, a few seconds later than live.")
                elif age > 5 and self.feed and self.feed.use_rest and not warned:
                    warned = True
                    await self.tg.send(f"⚠️ No fresh market data for {age:.0f} minutes. Check the server internet.")
                elif age <= 2:
                    warned = False
            except Exception:
                log.exception("data watchdog failed")

    # ----------------------------------------------------------- status
    async def status_reporter(self) -> None:
        """First report 20 minutes after start, then every STATUS_HOURS."""
        if self.cfg.status_hours <= 0:
            return
        await asyncio.sleep(20 * 60)
        while True:
            try:
                await self._send_status()
            except Exception:
                log.exception("status report failed")
            await asyncio.sleep(self.cfg.status_hours * 3600)

    def _model_line(self) -> str:
        m = self.learner.model or {}
        if not self.cfg.use_learning:
            return "🤖 Model: off in settings"
        if "weights" not in m:
            try:
                with open("logs/dataset.csv", encoding="utf-8") as f:
                    have = max(0, sum(1 for _ in f) - 1)
            except OSError:
                have = 0
            return f"🤖 Model: collecting data — {have} of {self.cfg.learn_min_signals} signals recorded"
        state = "ON" if self.learner.active else "OFF (not better than rules)"
        return f"🤖 Model: {state} • skipped this hour: {self.strategy.stats['blocked_model']}"

    def _hours_line(self) -> str:
        best, worst = self.tracker.best_worst("all")
        if not best:
            return ""
        fmt = lambda rows: ", ".join(f"{h:02d}–{(h + 1) % 24:02d} {v[1]:+.1f}R ({v[0]})" for h, v in rows)
        return f"\nBy hour (UTC, since start): best {fmt(best)}  •  worst {fmt(worst)}"

    async def daily_summary(self) -> None:
        """Every day at 00:05 UTC: yesterday's results hour by hour."""
        while True:
            now = time.time()
            nxt = (now // 86400 + 1) * 86400 + 300
            await asyncio.sleep(nxt - now)
            try:
                day = self.tracker.take_day()
                if not day:
                    continue
                total = sum(v[1] for v in day.values())
                n = sum(v[0] for v in day.values())
                wins = sum(v[2] for v in day.values())
                lines = [f"{int(h):02d}–{(int(h) + 1) % 24:02d}  {v[1]:+.2f}R  ({v[0]} signals, {v[2]} in profit)"
                         for h, v in sorted(day.items(), key=lambda x: int(x[0]))]
                await self.tg.send(
                    f"📅 <b>Daily summary — {utc_str(int((nxt - 86400) * 1000))[:10]}</b>\n"
                    f"Finished signals: {n}  •  in profit: {wins} ({wins / n * 100:.0f}%)\n"
                    f"Total: <b>{total:+.2f}R</b>\n\n<b>By hour (UTC):</b>\n" + "\n".join(lines)
                )
            except Exception:
                log.exception("daily summary failed")

    async def _send_status(self) -> None:
        now = now_ms()
        up = down = flat = long_ok = short_ok = shifted = 0
        for st in list(self.states.values()):
            t = self.strategy.trend_4h(st)
            up += t == "UP"
            down += t == "DOWN"
            flat += t is None
            side, why, _ = self.strategy.direction(st, now)
            long_ok += side == "LONG"
            short_ok += side == "SHORT"
            shifted += why.startswith("1H")
        s = self.strategy.stats
        data = "REST polling (slower)" if self.feed and self.feed.use_rest else "live"
        age = self.newest_1m_age_min()
        bdir, bend = self.strategy.btc_shift(now)
        if bdir:
            paused = "SHORT" if bdir == "bull" else "LONG"
            btc_line = f"BTC shifted {'up' if bdir == 'bull' else 'down'}: {paused} paused until {utc_str(bend)[11:16]} UTC"
        else:
            btc_line = "BTC: no recent structure shift, no pause"
        text = (
            f"📊 <b>Status</b>\n"
            f"Coins: {len(self.states)}  •  data: {data}  •  newest 1m candle: {max(age, 0):.0f} min old\n"
            f"1m candles checked: {s['candles']}\n"
            f"4H trend: {up} up • {down} down • {flat} no trend\n"
            f"1H structure shifted against 4H: {shifted} coins\n"
            f"{btc_line}\n"
            f"Ready now: {long_ok} LONG • {short_ok} SHORT\n"
            f"Crosses with trend: {s['crosses']}  •  Retests: {s['retests']}\n"
            f"Blocked by 1H shift: {s['blocked_1h']}  •  by BTC shift: {s['blocked_btc']}"
            f"  •  LONG without BTC uptrend: {s['blocked_btc_trend']}  •  LONG mode: {s['blocked_long_mode']}\n"
            f"Skipped: weak buyers/sellers {s['weak_taker']}  •  open interest/funding {s['blocked_oi']}\n"
            f"Expired: {s['expired']}  •  Failed: {s['failed']}  •  Weak volume: {s['low_volume']}\n"
            f"Signals sent: {s['signals']}  •  Open: {self.tracker.real_open()}"
            f"\n{self._model_line()}"
            f"{self._hours_line()}"
        )
        log.info("STATUS %s", text.replace("\n", " | "))
        await self.tg.send(text)
        s.clear()

    # ---------------------------------------------- open interest + funding
    async def market_context(self, sig) -> tuple[bool, str, dict]:
        """Does the move have fuel? Uses open interest (OI), funding and the long/short ratio.

        SHORT (LONG is the mirror image):
          * Skip if OI is falling over the last 30 min (no new positions behind the move).
          * Skip if funding is extremely negative (shorts very crowded).
          * Squeeze warning, skip: OI is slowing down (last 15 min weaker than the 15 min before),
            funding is strongly negative AND accounts are heavily short (long/short ratio low).
          * Fuel confirmed: OI rising in both halves and funding negative in each of the last
            3 periods -> new shorts are entering, the downtrend is likely to continue (quality A+).
        Unknown data never blocks a signal."""
        cfg = self.cfg
        info = {"oi_change_pct": None, "oi_first_half_pct": None, "oi_second_half_pct": None,
                "funding_pct": None, "funding_steady": None, "long_short_ratio": None, "fuel": None}
        if not cfg.use_oi_filter:
            return True, "", info
        sym, short = sig.symbol, sig.side == "SHORT"
        sign = -1 if short else 1          # fund * sign > 0 = funding on the crowded side of this trade
        try:
            if time.time() - self.funding_ts > 300:
                rows = await self.ex.fapiPublicGetPremiumIndex()
                self.funding = {r["symbol"]: float(r.get("lastFundingRate") or 0) * 100 for r in rows}
                self.funding_ts = time.time()
            if sym in self.funding:
                info["funding_pct"] = round(self.funding[sym], 4)
        except Exception as e:
            log.warning("funding data unavailable: %s", e)
        try:
            hist = await self.ex.fapiPublicGetFundingRate({"symbol": sym, "limit": 3})
            rates = [float(r["fundingRate"]) * 100 for r in hist]
            if len(rates) == 3:
                info["funding_steady"] = all(r * sign > 0 for r in rates)   # same side each time
        except Exception as e:
            log.warning("funding history unavailable for %s: %s", sym, e)
        try:
            points = max(3, cfg.oi_window_min // 5 + 1)
            hist = await self.ex.fapiDataGetOpenInterestHist({"symbol": sym, "period": "5m", "limit": points})
            oi = [float(h["sumOpenInterest"]) for h in hist]
            if len(oi) >= 3 and oi[0] > 0:
                mid = len(oi) // 2
                info["oi_change_pct"] = round((oi[-1] - oi[0]) / oi[0] * 100, 2)
                info["oi_first_half_pct"] = round((oi[mid] - oi[0]) / oi[0] * 100, 2)
                info["oi_second_half_pct"] = round((oi[-1] - oi[mid]) / oi[mid] * 100, 2) if oi[mid] else None
        except Exception as e:
            log.warning("open interest unavailable for %s: %s", sym, e)
        try:
            ls = await self.ex.fapiDataGetGlobalLongShortAccountRatio({"symbol": sym, "period": "5m", "limit": 1})
            if ls:
                info["long_short_ratio"] = round(float(ls[-1]["longShortRatio"]), 3)
        except Exception as e:
            log.warning("long/short ratio unavailable for %s: %s", sym, e)

        oi, oi1, oi2 = info["oi_change_pct"], info["oi_first_half_pct"], info["oi_second_half_pct"]
        fund, ratio = info["funding_pct"], info["long_short_ratio"]
        crowd = "shorts" if short else "longs"

        if oi is not None and oi < cfg.oi_min_change_pct:
            return False, f"open interest falling ({oi:+.2f}% in {cfg.oi_window_min} min)", info
        if fund is not None and fund * sign >= cfg.funding_extreme_pct:
            return False, f"funding extreme ({fund:+.4f}%), {crowd} very crowded", info
        slowing = oi1 is not None and oi2 is not None and oi2 < oi1
        strong_funding = fund is not None and fund * sign >= cfg.funding_strong_pct
        skewed = ratio is not None and (ratio <= cfg.ls_crowded_short if short else ratio >= cfg.ls_crowded_long)
        if slowing and strong_funding and skewed:
            return False, (f"squeeze warning: open interest slowing ({oi1:+.2f}% → {oi2:+.2f}%), "
                           f"funding {fund:+.4f}%, long/short ratio {ratio:.2f}"), info

        fuel = (oi1 is not None and oi2 is not None and oi1 > 0 and oi2 > 0 and info["funding_steady"] is True)
        info["fuel"] = fuel
        parts = []
        if oi is not None:
            parts.append(f"open interest {oi:+.2f}% in {cfg.oi_window_min} min")
        if fund is not None:
            parts.append(f"funding {fund:+.4f}%")
        if ratio is not None:
            parts.append(f"long/short {ratio:.2f}")
        line = ("Positioning: " + ", ".join(parts)) if parts else ""
        if fuel:
            who = "shorts" if short else "longs"
            steady = "negative" if short else "positive"
            line += f"\n🔥 Fuel: new {who} entering (open interest rising, funding steadily {steady})"
            sig.quality = "A+"
        return True, line, info

    # ------------------------------------------------------ self-learning
    async def learning_loop(self) -> None:
        """Checks every 6 hours whether the model is due for (re)training; trains in the background."""
        await asyncio.sleep(600)
        while True:
            try:
                if self.learner.due():
                    report = await asyncio.to_thread(self.learner.train)
                    if report:
                        log.info("model: %s", report.replace("\n", " | "))
                        await self.tg.send(report)
            except Exception:
                log.exception("model training failed")
            await asyncio.sleep(6 * 3600)

    # --------------------------------------------------------- strategy
    def update_breadth(self) -> None:
        """Share of all scanned coins whose 4H is in an uptrend (used by LONG_MODE=strong)."""
        states = list(self.states.values())
        if states:
            up = sum(1 for st in states if self.strategy.trend_4h(st) == "UP")
            self.strategy.breadth_up = up / len(states)

    async def evaluator(self) -> None:
        """Every EVAL_INTERVAL_SEC: follow open signals, then check new closed 1m candles."""
        last_breadth = 0.0
        while True:
            now = now_ms()
            if time.time() - last_breadth > 300:
                self.update_breadth()
                last_breadth = time.time()
            for st in list(self.states.values()):
                if not st.updated:
                    continue
                st.updated = False
                try:
                    await self.tracker.on_tick(st)
                    st.tick_high = st.tick_low = None
                    if self.tracker.has_open(st.symbol) or time.time() < st.cooldown_until:
                        st.setup = None
                        st.last_closed_checked = st.candles["1m"][-1][0] if st.candles["1m"] else 0
                        st.htf_checked = st.candles["15m"][-1][0] if st.candles["15m"] else 0
                        st.smc_1h_checked = st.candles["1h"][-1][0] if st.candles["1h"] else 0
                        continue
                    sig = self.strategy.evaluate(st, now)
                    if sig:
                        st.cooldown_until = time.time() + self.cfg.cooldown_min * 60
                        ok, line, info = await self.market_context(sig)
                        sig.features.update(info)
                        if sig.card:
                            # SMC: open interest is shown as information; only extreme funding blocks
                            ok = ok or not line.startswith("funding extreme")
                            if line and ok:
                                sig.card["reasons"][-1] += "; " + line.replace("Positioning: ", "").split("\n")[0]
                        if not ok:
                            self.strategy.stats["blocked_oi"] += 1
                            self.strategy.stats["signals"] -= 1
                            log.info("%s %s skipped: %s", sig.symbol, sig.side, line)
                            continue
                        if line:
                            sig.reasons.append(line)
                        prob = self.learner.probability(sig.features) if self.cfg.strategy == "ema" else None
                        if self.learner.active and prob is not None and prob < self.learner.model["threshold"]:
                            # the model says no: follow it silently so the model keeps learning
                            self.strategy.stats["blocked_model"] += 1
                            self.strategy.stats["signals"] -= 1
                            log.info("%s %s skipped by model (%.0f%% < %.0f%%)", sig.symbol, sig.side,
                                     prob * 100, self.learner.model["threshold"] * 100)
                            await self.tracker.publish(sig, st.volume_24h, shadow=True, model_prob=prob)
                            continue
                        if self.learner.active and prob is not None:
                            sig.reasons.append(f"🤖 Model: {prob * 100:.0f}% chance of profit "
                                               f"(minimum {self.learner.model['threshold'] * 100:.0f}%)")
                        await self.tracker.publish(sig, st.volume_24h, model_prob=prob)
                except Exception:
                    log.exception("evaluation error on %s", st.symbol)
            await asyncio.sleep(self.cfg.eval_interval_sec)

    # -------------------------------------------------------------- run
    async def run(self) -> None:
        cfg = self.cfg
        pairs = await select_symbols(self.ex, cfg)
        if not pairs:
            raise SystemExit("No coins pass the volume filter - check MIN_24H_VOLUME_USDT")
        self.states = {pid: SymbolState(pid, sym, volume_24h=vol) for pid, (sym, vol) in pairs.items()}
        log.info("Loading history for %d coins (a few minutes)...", len(pairs))
        self.feed = MarketFeed(cfg, self.ex, self.states)
        self.strategy.btc = self.states.get("BTCUSDT")
        await self.feed.bootstrap()

        if cfg.strategy == "smc":
            await self.tg.send(
                f"🤖 <b>Smart Money signal bot started</b>\n"
                f"Scanning the {len(pairs)} most liquid Binance futures coins\n"
                f"4H structure → 1H liquidity sweep → displacement → OB/FVG retest → 15m confirmation\n"
                f"Minimum confluence score {cfg.smc_min_score}/5 • 5h cooldown per coin\n"
                f"Signals only, no trading")
        elif cfg.strategy == "orb":
            await self.tg.send(
                f"🤖 <b>New York Open Range signal bot started</b>\n"
                f"Scanning {len(pairs)} coins with 24h futures volume ≥ {fmt_usd(cfg.min_24h_volume)} USDT\n"
                f"4H + 1H trend → first {cfg.orb_range_min:g} min after 9:30 New York → "
                f"1m break + retest + big candle with volume (weekdays)\n"
                f"🟢 = LONG   🔴 = SHORT   (signals only, no trading)")
        elif cfg.strategy == "range":
            await self.tg.send(
                f"🤖 <b>5m High/Low Break signal bot started</b>\n"
                f"Scanning {len(pairs)} coins with 24h futures volume ≥ {fmt_usd(cfg.min_24h_volume)} USDT\n"
                f"4H + 1H trend → 5m high/low lines → 1m break + retest + big candle with volume\n"
                f"🟢 = LONG   🔴 = SHORT   (signals only, no trading)")
        else:
          await self.tg.send(
            f"🤖 <b>{BOT_NAME} started</b>\n"
            f"Scanning {len(pairs)} coins with 24h futures volume ≥ {fmt_usd(cfg.min_24h_volume)} USDT\n"
            f"4H trend → {'15m check → ' if cfg.use_15m_filter else ''}"
            f"{'1H shift filter → ' if cfg.use_shift_filter else ''}{'BTC shift filter → ' if cfg.use_btc_filter else ''}"
            f"{'OI/funding check → ' if cfg.use_oi_filter else ''}"
            f"1m EMA {cfg.entry_ema} "
            f"cross + retest + big candle (within {cfg.retest_window_min:g} min)\n"
            f"🟢 = LONG   🔴 = SHORT   (signals only, no trading)"
        )
        self.journal.write("start", strategy="ema_retest", pairs=list(pairs))
        self._restart_feed()
        await asyncio.gather(self.evaluator(), self.pair_refresher(), self.status_reporter(),
                             self.data_watchdog(), self.daily_summary(),
                             self.learning_loop())

    async def close(self) -> None:
        for closer in (self.ex.close, self.tg.close):
            try:
                await closer()
            except Exception:
                pass


async def main() -> None:
    cfg = Config()
    setup_logging(cfg.log_level)
    bot = Bot(cfg)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, task.cancel)
        except NotImplementedError:
            pass
    error = None
    try:
        await bot.run()
    except asyncio.CancelledError:
        log.info("Stopping...")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log.exception("Bot crashed")
        raise
    finally:
        if error:
            await bot.tg.send(f"⚠️ Bot error, restarting in 10 seconds:\n<code>{html.escape(error[:300])}</code>")
        else:
            await bot.tg.send(f"⏹ {BOT_NAME} stopped.")
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
