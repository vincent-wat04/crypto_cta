"""
Telegram notification helper for the live trading runner.

Setup
-----
1. Create a bot via @BotFather → get TELEGRAM_BOT_TOKEN
2. Send any message to your bot, then call get_chat_id() to find your TELEGRAM_CHAT_ID
3. Set env vars (or pass directly):
       export TELEGRAM_BOT_TOKEN="7xxxxxxxxxx:AAxxx..."
       export TELEGRAM_CHAT_ID="-1001234567890"   # channel / group or personal chat id

Usage
-----
    notifier = TelegramNotifier.from_env()
    await notifier.send("Hello from bot")             # async
    notifier.send_sync("Hello from thread")           # sync (blocks)
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    from telegram.error import TelegramError
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False


class TelegramNotifier:
    """
    Thin wrapper around python-telegram-bot v20+ async API.

    All public async methods are safe to call from within an asyncio event loop.
    `send_sync` wraps in a thread-pool future for use from sync contexts (e.g. callbacks).
    """

    def __init__(self, token: str, chat_id: str, disable_web_preview: bool = True):
        if not _HAS_TELEGRAM:
            raise ImportError("pip install python-telegram-bot")
        if not token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must not be empty")

        self.chat_id = str(chat_id)
        self._bot = Bot(token=token)
        self._disable_web_preview = disable_web_preview
        self._enabled = True

    # ─── Factory ───

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        """Build from environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        return cls(token=token, chat_id=chat_id)

    @classmethod
    def from_env_optional(cls) -> Optional["TelegramNotifier"]:
        """
        Like from_env() but returns None (instead of raising) when env vars are absent.
        Use this in the runner so it degrades gracefully if Telegram is not configured.
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            logger.debug("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; notifications disabled")
            return None
        try:
            return cls(token=token, chat_id=chat_id)
        except Exception as e:
            logger.warning("TelegramNotifier init failed: %s", e)
            return None

    # ─── Core send ───

    async def send(self, text: str, parse_mode: str = ParseMode.HTML) -> bool:
        """Send a message asynchronously. Returns True on success."""
        if not self._enabled:
            return False
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=self._disable_web_preview,
            )
            return True
        except TelegramError as e:
            logger.error("Telegram send failed: %s", e)
            return False

    def send_sync(self, text: str) -> bool:
        """
        Send from a synchronous context (e.g. thread callback).
        If a running event loop exists, schedules the coroutine on it.
        Otherwise creates a temporary loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule on existing loop (fire-and-forget)
                asyncio.run_coroutine_threadsafe(self.send(text), loop)
                return True
            return loop.run_until_complete(self.send(text))
        except Exception as e:
            logger.error("send_sync error: %s", e)
            return False

    async def get_chat_id(self) -> str:
        """
        Helper: fetch the most recent chat_id that messaged this bot.
        Useful for first-time setup when you don't know your chat_id yet.
        """
        updates = await self._bot.get_updates()
        if not updates:
            return "(no messages yet — send any message to your bot first)"
        last = updates[-1]
        chat = last.message.chat if last.message else None
        if chat:
            return f"chat_id={chat.id}  title={getattr(chat, 'title', chat.first_name)}"
        return "(could not extract chat_id)"

    # ─── Formatted message builders ───

    async def send_trade(
        self,
        action: str,
        old_pos: float,
        new_pos: float,
        price: float,
        close_pnl_bps: float,
        est_fee_bps: float,
        est_net_pnl: float,
        cum_pnl_estimated: float,
        symbol: str = "SOL/USDC",
    ) -> bool:
        direction = "📈 LONG" if new_pos > old_pos else "📉 SHORT" if new_pos < old_pos else "⬜ FLAT"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        text = (
            f"<b>{direction} — {action}</b>  [{symbol}]\n"
            f"🕐 {ts}\n"
            f"─────────────────\n"
            f"Pos:   <code>{old_pos:+.1f} → {new_pos:+.1f}</code>\n"
            f"Price: <code>{price:.4f}</code>\n"
            f"Close PnL: <code>{close_pnl_bps:+.2f} bps</code>\n"
            f"Est fee:   <code>{est_fee_bps:.2f} bps</code>\n"
            f"Net PnL:   <code>{est_net_pnl:+.2f} bps</code>\n"
            f"Cum (est): <code>{cum_pnl_estimated:+.1f} bps</code>"
        )
        return await self.send(text)

    async def send_fill(
        self,
        trade_idx: int,
        fill_status: str,
        contracts: float,
        avg_fill_price: float,
        actual_fee_bps: float,
        est_fee_bps: float,
        net_pnl: float,
        cum_pnl_actual: float,
        fill_stats_line: str,
    ) -> bool:
        emoji = {"maker": "✅", "taker": "⚡", "partial": "🔶", "failed": "❌", "error": "🚨"}.get(
            fill_status, "❓"
        )
        text = (
            f"{emoji} <b>Fill #{trade_idx} [{fill_status.upper()}]</b>\n"
            f"Qty:   <code>{contracts:.4f}</code>  @ <code>{avg_fill_price:.4f}</code>\n"
            f"Fee:   actual <code>{actual_fee_bps:.2f}</code>  est <code>{est_fee_bps:.2f}</code>  "
            f"diff <code>{actual_fee_bps - est_fee_bps:+.2f} bps</code>\n"
            f"Net PnL: <code>{net_pnl:+.2f} bps</code>\n"
            f"Cum (actual): <code>{cum_pnl_actual:+.1f} bps</code>\n"
            f"<i>{fill_stats_line}</i>"
        )
        return await self.send(text)

    async def send_startup(self, config_summary: str) -> bool:
        text = f"🚀 <b>Runner started</b>\n<pre>{config_summary}</pre>"
        return await self.send(text)

    async def send_shutdown(self, pnl_estimated: float, pnl_actual: float,
                            n_trades: int, fill_stats_line: str) -> bool:
        text = (
            f"🛑 <b>Runner stopped</b>\n"
            f"Trades: <code>{n_trades}</code>\n"
            f"PnL (est):    <code>{pnl_estimated:+.1f} bps</code>\n"
            f"PnL (actual): <code>{pnl_actual:+.1f} bps</code>\n"
            f"<i>{fill_stats_line}</i>"
        )
        return await self.send(text)

    async def send_error(self, context: str, error: str) -> bool:
        text = f"🚨 <b>ERROR</b> [{context}]\n<code>{error}</code>"
        return await self.send(text)

    async def send_warmup_done(self, n_bars: int, symbol: str) -> bool:
        text = f"🔥 <b>Warmup complete</b> — {symbol}\nAccumulated <code>{n_bars}</code> bars, trading active."
        return await self.send(text)
