"""
Self-learning signal filter (machine learning).

What it does, by itself, without anyone's help:
  1. Waits until LEARN_MIN_SIGNALS finished signals are recorded in logs/dataset.csv.
  2. Every LEARN_EVERY_DAYS days it trains a model on all recorded signals:
     "given these numbers (candle size, volume, sellers' share, stop size, RSI, open interest,
     funding, hour ...), how likely is this signal to end in profit?"
  3. It tests the model on the most recent LEARN_TEST_DAYS days, which the model has NOT seen.
  4. The model is switched ON only if, on those unseen days, the signals it would have kept
     earned at least LEARN_MIN_GAIN_R more per signal than all signals together, while keeping
     at least LEARN_MIN_KEEP of them. Otherwise the bot keeps working with its normal rules.
  5. When ON, each signal gets a win probability; signals below the chosen level are skipped.
     Skipped signals are still followed silently and recorded, so the model keeps learning
     from complete, unbiased data.

The model is a logistic regression written with numpy only: small, fast, transparent,
and hard to over-fit on a few hundred signals.
"""
import csv
import json
import logging
import math
import os
import time

import numpy as np

log = logging.getLogger("learner")

DATASET = "logs/dataset.csv"
MODEL_FILE = "state/model.json"

# numbers the model learns from (all already "turned" so that higher = more in the trade's direction)
NUMERIC = [
    "body_x", "vol_x", "taker_share", "sl_pct", "atr1m_pct", "retest_bars", "confirm_bars",
    "gap_4h_pct", "gap_15m_pct", "gap_1m_pct", "log_volume_24h",
    "oi_change_pct", "oi_first_half_pct", "oi_second_half_pct", "funding_dir_pct", "log_ls_ratio",
    "fuel", "is_long", "btc_trend_dir", "hour_sin", "hour_cos",
    "rsi_1m_dir", "rsi_15m_dir", "rsi_1h_dir", "vol_5_20", "coin_change_1h_dir", "btc_change_1h_dir",
]


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def make_row(feat: dict) -> dict:
    """Turns recorded features into model inputs (direction-adjusted where it matters)."""
    long_ = feat.get("side") == "LONG"
    d = 1 if long_ else -1
    hour = _f(feat.get("hour_utc")) or 0
    fund = _f(feat.get("funding_pct"))
    ls = _f(feat.get("long_short_ratio"))
    vol24 = _f(feat.get("volume_24h_m"))
    btc = feat.get("btc_trend_4h")

    def rsi_dir(key):
        r = _f(feat.get(key))
        return None if r is None else (r if long_ else 100 - r)

    def signed(key):
        v = _f(feat.get(key))
        return None if v is None else v * d

    fuel = feat.get("fuel")
    return {
        "body_x": _f(feat.get("body_x")), "vol_x": _f(feat.get("vol_x")),
        "taker_share": _f(feat.get("taker_share")), "sl_pct": _f(feat.get("sl_pct")),
        "atr1m_pct": _f(feat.get("atr1m_pct")), "retest_bars": _f(feat.get("retest_bars")),
        "confirm_bars": _f(feat.get("confirm_bars")), "gap_4h_pct": _f(feat.get("gap_4h_pct")),
        "gap_15m_pct": _f(feat.get("gap_15m_pct")), "gap_1m_pct": _f(feat.get("gap_1m_pct")),
        "log_volume_24h": math.log10(vol24 + 0.1) if vol24 is not None else None,
        "oi_change_pct": _f(feat.get("oi_change_pct")),
        "oi_first_half_pct": _f(feat.get("oi_first_half_pct")),
        "oi_second_half_pct": _f(feat.get("oi_second_half_pct")),
        "funding_dir_pct": None if fund is None else fund * d,
        "log_ls_ratio": math.log(ls) * d if ls and ls > 0 else None,
        "fuel": 1.0 if str(fuel) in ("True", "1", "true") else 0.0 if fuel not in (None, "") else None,
        "is_long": 1.0 if long_ else 0.0,
        "btc_trend_dir": (1.0 if btc == "UP" else -1.0 if btc == "DOWN" else 0.0) * d,
        "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
        "rsi_1m_dir": rsi_dir("rsi_1m"), "rsi_15m_dir": rsi_dir("rsi_15m"), "rsi_1h_dir": rsi_dir("rsi_1h"),
        "vol_5_20": _f(feat.get("vol_5_20")),
        "coin_change_1h_dir": signed("coin_change_1h_pct"),
        "btc_change_1h_dir": signed("btc_change_1h_pct"),
    }


class Learner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self._load()

    # --------------------------------------------------------------- state
    def _load(self) -> None:
        try:
            if os.path.exists(MODEL_FILE):
                with open(MODEL_FILE, encoding="utf-8") as f:
                    self.model = json.load(f)
        except Exception as e:
            log.warning("model file unreadable: %s", e)

    def _save(self) -> None:
        os.makedirs("state", exist_ok=True)
        with open(MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(self.model, f)

    @property
    def active(self) -> bool:
        return bool(self.cfg.use_learning and self.model and self.model.get("active"))

    def due(self) -> bool:
        if not self.cfg.use_learning:
            return False
        last = (self.model or {}).get("trained_at", 0)
        return time.time() - last >= self.cfg.learn_every_days * 86400

    # ------------------------------------------------------------- predict
    def probability(self, features: dict) -> float | None:
        if not self.model or "weights" not in self.model:
            return None
        m = self.model
        row = make_row(features)
        x = np.array([(row[k] if row[k] is not None else m["fill"][i]) for i, k in enumerate(NUMERIC)])
        x = (x - np.array(m["mean"])) / np.array(m["std"])
        z = float(np.dot(x, m["weights"]) + m["bias"])
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    # --------------------------------------------------------------- train
    @staticmethod
    def _read() -> list:
        if not os.path.exists(DATASET):
            return []
        with open(DATASET, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("outcome") not in (None, "", "not_filled")
                    and r.get("result_r") not in (None, "")]
        return rows

    @staticmethod
    def _fit(X, y, l2=1.0, steps=3000, lr=0.1):
        w = np.zeros(X.shape[1])
        b = 0.0
        n = len(y)
        for _ in range(steps):
            p = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
            g = p - y
            w -= lr * (X.T @ g / n + l2 * w / n)
            b -= lr * g.mean()
        return w, b

    def _prepare(self, rows, stats=None):
        raw = [[make_row(r)[k] for k in NUMERIC] for r in rows]
        if stats is None:
            cols = list(zip(*raw))
            fill = [float(np.median([v for v in c if v is not None])) if any(v is not None for v in c) else 0.0
                    for c in cols]
            X0 = np.array([[v if v is not None else fill[i] for i, v in enumerate(r)] for r in raw])
            mean, std = X0.mean(axis=0), X0.std(axis=0)
            std[std == 0] = 1.0
            stats = (fill, mean, std)
        fill, mean, std = stats
        X = np.array([[v if v is not None else fill[i] for i, v in enumerate(r)] for r in raw])
        return (X - mean) / std, stats

    def train(self) -> str:
        """Trains, tests on unseen recent days, decides ON/OFF. Returns a report for Telegram."""
        cfg = self.cfg
        rows = self._read()
        n = len(rows)
        if n < cfg.learn_min_signals:
            self.model = {**(self.model or {}), "trained_at": time.time() - (cfg.learn_every_days - 1) * 86400}
            return ""   # not enough data yet; check again tomorrow, no message
        rows.sort(key=lambda r: int(float(r["time_utc"])))
        cut_ms = int(float(rows[-1]["time_utc"])) - cfg.learn_test_days * 86_400_000
        train = [r for r in rows if int(float(r["time_utc"])) < cut_ms]
        test = [r for r in rows if int(float(r["time_utc"])) >= cut_ms]
        if len(test) < cfg.learn_min_test or len(train) < cfg.learn_min_signals // 2:
            self.model = {**(self.model or {}), "trained_at": time.time() - (cfg.learn_every_days - 1) * 86400}
            return ""

        r_tr = np.array([float(r["result_r"]) for r in train])
        r_te = np.array([float(r["result_r"]) for r in test])
        X_tr, stats = self._prepare(train)
        X_te, _ = self._prepare(test, stats)
        w, b = self._fit(X_tr, (r_tr > 0).astype(float))
        p_tr = 1 / (1 + np.exp(-np.clip(X_tr @ w + b, -30, 30)))
        p_te = 1 / (1 + np.exp(-np.clip(X_te @ w + b, -30, 30)))

        # pick the probability level on the TRAINING data only
        best_thr, best_avg = 0.0, r_tr.mean()
        for q in np.arange(0.05, 0.80, 0.05):
            thr = float(np.quantile(p_tr, q))
            keep = p_tr >= thr
            if keep.mean() >= cfg.learn_min_keep and r_tr[keep].mean() > best_avg:
                best_thr, best_avg = thr, r_tr[keep].mean()

        # judge it on the UNSEEN recent days
        keep_te = p_te >= best_thr
        base_avg, base_win = r_te.mean(), (r_te > 0).mean()
        kept = int(keep_te.sum())
        model_avg = r_te[keep_te].mean() if kept else 0.0
        model_win = (r_te[keep_te] > 0).mean() if kept else 0.0
        share = kept / len(r_te)
        good = best_thr > 0 and share >= cfg.learn_min_keep and model_avg - base_avg >= cfg.learn_min_gain_r

        # final model: same settings, trained on everything (incl. the test days)
        X_all, stats_all = self._prepare(rows)
        w_all, b_all = self._fit(X_all, (np.array([float(r["result_r"]) for r in rows]) > 0).astype(float))
        fill, mean, std = stats_all
        top = sorted(zip(NUMERIC, w_all), key=lambda t: -abs(t[1]))[:4]
        self.model = {
            "active": bool(good), "trained_at": time.time(), "threshold": best_thr,
            "weights": [float(v) for v in w_all], "bias": float(b_all),
            "fill": [float(v) for v in fill], "mean": [float(v) for v in mean], "std": [float(v) for v in std],
            "signals": n, "test": {"base_avg": float(base_avg), "model_avg": float(model_avg),
                                   "base_win": float(base_win), "model_win": float(model_win), "kept": share},
            "top": [[k, float(v)] for k, v in top],
        }
        self._save()
        state = "ON ✅" if good else "OFF (not better than the normal rules)"
        tops = ", ".join(f"{k} {'+' if v > 0 else '−'}" for k, v in top)
        return (
            f"🤖 <b>Model retrained on {n} signals</b>\n"
            f"Tested on the last {cfg.learn_test_days:g} days (not seen in training, {len(test)} signals):\n"
            f"• All signals: {base_win * 100:.0f}% in profit, {base_avg:+.2f}R per signal\n"
            f"• Kept by model ({share * 100:.0f}%): {model_win * 100:.0f}% in profit, {model_avg:+.2f}R per signal\n"
            f"Model: <b>{state}</b>\n"
            f"Most important: {tops}"
        )
