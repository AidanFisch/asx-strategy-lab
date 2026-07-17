"""
Telegram notifier. Sends messages via the free Telegram Bot API.

Credentials come from the environment (see config.TELEGRAM_*). If they're not
set, this runs in DRY-RUN mode: it prints the message instead of sending, so the
daily scan works end-to-end without any setup.

Setup (free, ~5 min):
  1. Message @BotFather -> /newbot -> copy the bot token.
  2. Message your new bot once, then open
     https://api.telegram.org/bot<TOKEN>/getUpdates  and read your chat id.
  3. setx TELEGRAM_BOT_TOKEN "<token>"  &&  setx TELEGRAM_CHAT_ID "<id>"
     (open a fresh terminal so the env vars load.)
"""

from __future__ import annotations

import logging

import requests

import config

log = logging.getLogger("telegram")

API = "https://api.telegram.org/bot{token}/sendMessage"


def is_configured() -> bool:
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def send_message(text: str, disable_preview: bool = True) -> bool:
    """
    Send a Telegram message. Returns True if actually sent, False on dry-run or error.
    Never raises — a notification failure must not crash the scan.
    """
    if not is_configured():
        log.info("[DRY-RUN telegram] %s", text.replace("\n", " | "))
        return False

    url = API.format(token=config.TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": disable_preview,
    }
    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code == 200 and r.json().get("ok"):
            return True
        log.error("telegram send failed: %s %s", r.status_code, r.text[:200])
        return False
    except Exception as e:
        log.error("telegram send error: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = send_message("<b>ASX Strategy Lab</b>\nTelegram test message ✅")
    print("configured:", is_configured(), "| sent:", ok)
