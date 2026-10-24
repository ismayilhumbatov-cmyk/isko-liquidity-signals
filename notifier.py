"""
Telegram messages.

🟢 = LONG signal     🔴 = SHORT signal
✅ = TP1 hit         🎯 = TP2 hit       🛑 = stop loss      ⚖️ = breakeven exit
"""
import asyncio
import html
import logging

import aiohttp

from .utils import fmt_price, fmt_usd, pct, utc_str

log = logging.getLogger("telegram")
LINE = "━━━━━━━━━━━━━━━"


class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat_id = token, chat_id
        self.session: aiohttp.ClientSession | None = None

    async def close(self):
        if self.session:
            await self.session.close()

    async def send(self, text: str) -> None:
        if not self.token or not self.chat_id:
            log.info("(Telegram not configured)\n%s", text)
            return
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        for attempt in range(4):
            try:
                async with self.session.post(url, json=payload) as r:
                    if r.status == 200:
                        return
                    body = await r.json(content_type=None)
                    wait = (body.get("parameters") or {}).get("retry_after", 2 * (attempt + 1))
                    log.warning("Telegram error %s: %s", r.status, body.get("description"))
                    if r.status == 400:
                        return
                    await asyncio.sleep(wait)
            except asyncio.TimeoutError:
                # the message may already have been delivered; retrying would send it twice
                log.warning("Telegram send timed out (not retried to avoid a duplicate)")
                return
            except Exception as e:
                log.warning("Telegram send failed: %s", e)
                await asyncio.sleep(2 * (attempt + 1))


def _e(s) -> str:
    return html.escape(str(s))


def signal_message(sig, cfg, volume_24h: float = 0.0) -> str:
    """Main signal card: green for LONG, red for SHORT."""
    long_ = sig.side == "LONG"
    head = "🟢🟢 <b>LONG</b>" if long_ else "🔴🔴 <b>SHORT</b>"
    lines = [
        f"{head}  <b>#{_e(sig.symbol)}</b>   Quality: <b>{sig.quality}</b>",
        LINE,
        (f"Entry (limit, Post Only): <code>{fmt_price(sig.entry)}</code>  • valid {cfg.limit_valid_min:g} min"
         if sig.limit else f"Entry:      <code>{fmt_price(sig.entry)}</code>"),
        *( [f"Signal price: {fmt_price(sig.signal_price)}"] if sig.limit else [] ),
        f"Stop Loss:  <code>{fmt_price(sig.sl)}</code>  ({pct(sig.entry, sig.sl):+.2f}%)",
        f"TP1 ({cfg.tp1_close_pct:g}%): <code>{fmt_price(sig.tp1)}</code>  ({pct(sig.entry, sig.tp1):+.2f}%, {sig.r_of(sig.tp1):.1f}R)",
        f"TP2 (rest): <code>{fmt_price(sig.tp2)}</code>  ({pct(sig.entry, sig.tp2):+.2f}%, {sig.r_of(sig.tp2):.1f}R)",
        f"Leverage: {cfg.leverage}x • Risk: {cfg.risk_pct:g}% of balance",
        f"24h volume: {fmt_usd(volume_24h)} USDT",
        LINE,
        "<b>Why:</b>",
    ]
    lines += [f"• {_e(r)}" for r in sig.reasons]
    lines += [f"• Targets: {_e(sig.target_note)}", LINE, f"🕒 {utc_str(sig.created_ms)}  |  Signal only"]
    return "\n".join(lines)


def not_filled_message(s, minutes: float) -> str:
    side_icon = "🟢" if s.side == "LONG" else "🔴"
    return (f"⌛ <b>Not filled</b> — {side_icon} #{_e(s.symbol)} {s.side}\n"
            f"Price did not come back to the limit {fmt_price(s.entry)} within {minutes:g} min. "
            f"Cancel the order — no trade.")


def exit_message(s, reason: str, share: float, price: float, r: float, moved_be: bool) -> str:
    icon = {"TP1": "✅", "TP2": "🎯", "SL": "🛑", "BE": "⚖️"}.get(reason, "ℹ️")
    name = {"TP1": "TP1 hit", "TP2": "TP2 hit", "SL": "Stop loss hit", "BE": "Breakeven"}.get(reason, reason)
    side_icon = "🟢" if s.side == "LONG" else "🔴"
    move = (price - s.entry) / s.entry * 100 * (1 if s.side == "LONG" else -1)
    lines = [
        f"{icon} <b>{name}</b> — {side_icon} #{_e(s.symbol)} {s.side}",
        f"Price <code>{fmt_price(price)}</code> ({move:+.2f}% from entry, {r:+.1f}R) • {share * 100:.0f}% of the position",
    ]
    if moved_be:
        lines.append("Move stop to entry (breakeven) for the rest.")
    return "\n".join(lines)


def closed_message(s) -> str:
    icon = "🏆" if s.result_r > 0.05 else "📉" if s.result_r < -0.05 else "⚖️"
    side_icon = "🟢" if s.side == "LONG" else "🔴"
    return (f"{icon} <b>Signal finished</b> — {side_icon} #{_e(s.symbol)} {s.side}\n"
            f"Result: <b>{s.result_r:+.2f}R</b>")
