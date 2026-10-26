# Binance USDT-M Futures Signal Bot

**Current mode: RSI alert bot (`STRATEGY=rsi` in settings.env).** Every 5 minutes it reports coins where the 5-minute RSI is above 70 and the 4H RSI is above 70 (🔴 overbought), or where the 5-minute RSI is below 20 / 25 / 30 depending on market cap under 50M / 50–100M / above 100M (🟢 oversold). The most extreme coin comes first. Information only.

The other strategies below remain available and testable with `backtest.py`.

# Trend + EMA 50 Retest Signal Bot

**Signals only.** The bot never places orders and needs no Binance API keys. It reads public Binance Futures data, sends 🟢 LONG / 🔴 SHORT signals to Telegram, and then reports ✅ TP1, 🎯 TP2, 🛑 stop loss or ⚖️ breakeven for each signal.

## The strategy

**1. 4H trend decides the direction.**
Price above a rising 4H EMA 50 means uptrend, so LONG only. Price below a falling 4H EMA 50 means downtrend, so SHORT only. No clear trend means no trade on that coin.

**2. 15m confirms it.**
For LONG, the last 15m candle must close above the 15m EMA 50. For SHORT, below it.

**Structure shift filters.**
- **1H shift:** if the coin's 1H chart closes above its last swing high, the structure has turned up, so no SHORT signals. If it closes below its last swing low, no LONG signals.
- **BTC shift:** when Bitcoin's 15m structure shifts up, SHORT signals on all coins pause for 60 minutes. When it shifts down, LONG signals pause.

**Extra filters.**
- LONG signals only while Bitcoin's 4H is in an uptrend.
- On the big candle, market sellers (SHORT) or buyers (LONG) must make up at least 55% of the volume.
- Open interest must not be falling over the last 30 minutes, and funding must not be extreme against the trade.

**Limit entry.** The entry is a limit (Post Only) order slightly better than the candle close, so the entry fee is 0.02% instead of 0.05%. If price doesn't return to it within 3 minutes, the bot sends "Not filled" and the signal is cancelled.

**Self-learning model.** Every finished signal is saved in `logs/dataset.csv` with about 30 numbers (candle size, volume, sellers'/buyers' share, stop size, RSI on 1m/15m/1H, volume trend, open interest, funding, long/short ratio, BTC and coin 1H change, hour) and its result. After 500 signals the bot trains a model by itself every 7 days, tests it on the last 7 days it has not seen, and switches it on only if it clearly beats the normal rules. When on, each signal shows its win probability and weak signals are skipped (and still followed silently, so the model keeps learning). Every retraining sends a short report to Telegram.

**3. 1m gives the entry, all within 15 minutes:**
- **Cross:** a 1m candle closes through the EMA 50 in the trend direction.
- **Retest:** a later candle comes back and touches the EMA 50 without closing through it. It can be the very next candle.
- **Big candle:** a separate candle after the retest closes strongly in the trend direction. Its body is at least 1.5× the average and its volume is above average. It closes beyond the EMA 50 and beyond the retest candle. **Entry is at its close.**

The setup is cancelled if price closes back through the EMA 50, if 15 minutes pass, or if the 4H or 15m direction changes.

**4. Stop and targets.**
The stop sits just beyond the retest wick, never closer than 0.3% from entry. TP1 at 1.5R closes half and moves the stop to entry. TP2 at 3R closes the rest. (R = distance from entry to stop.)

## Coins

All Binance USDT-M perpetual crypto coins with at least 3 million USDT futures volume in 24 hours. Stablecoins, gold and stock contracts are excluded. The list refreshes every 60 minutes.

## Install or update (Hetzner black console, no Shift keys needed)

```
curl --proto-default https raw.githubusercontent.com/ismayilhumbatov-cmyk/isko-liquidity-signals/main/install.sh -o i.sh
bash i.sh
```

## Change settings

Open `settings.env` on GitHub, click the pencil icon, change a number, commit, then run the two install lines once. Each setting is explained in the file.

## Telegram messages

- **🤖 started / ⏹ stopped:** the bot starts or stops (normal during updates)
- **🟢 LONG / 🔴 SHORT:** a signal, with entry, stop loss, TP1, TP2 and the reasons
- **✅ 🎯 🛑 ⚖️:** results of each signal
- **📊 Status** every 6 hours: trend count, crosses, retests, signals
- **⚠️ Bot error:** the bot hit an error and restarts by itself

## Backtest (compare versions on real history)

```
cd /opt/liquidity-sweep-bot
.venv/bin/python backtest.py
```

Downloads Binance history and replays the strategy minute by minute: A (current), C (+ fresh EMA 3 cross), D (C with LONG off), E (C with LONG only in a strong market). Default 7 days and 30 coins (about 10 minutes). For 3 months: `.venv/bin/python backtest.py 90` (20 coins, about 30–40 minutes, with a 30-day breakdown). Other strategies: `backtest.py 30 range` (5m lines, 1m entry), `backtest.py 90 htf` (1H lines, 15m entry, Daily + 4H trend) `backtest.py 90 smc` (Smart Money liquidity sweep + Order Block/FVG retest, minimum score 3, 4 and 5) and `backtest.py 30 orb` (New York opening range: lines from the first 5 or 15 minutes after 9:30 New York time, 1m entry, weekdays, one signal per coin per day). The live bot keeps running. The result is printed and sent to Telegram.
