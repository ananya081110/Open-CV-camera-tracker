"""
Retail notification service.

WhatsApp uses the WhatsApp Cloud API when configured.
Telegram is retained as a useful POC fallback.

Environment variables:
    WHATSAPP_ACCESS_TOKEN
    WHATSAPP_PHONE_NUMBER_ID
    WHATSAPP_TO

    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import json
import os
import urllib.request


class RetailNotificationService:

    def __init__(self):
        self.whatsapp_token = os.getenv(
            "WHATSAPP_ACCESS_TOKEN", ""
        ).strip()
        self.whatsapp_phone_number_id = os.getenv(
            "WHATSAPP_PHONE_NUMBER_ID", ""
        ).strip()
        self.whatsapp_to = os.getenv(
            "WHATSAPP_TO", ""
        ).strip()

        self.telegram_token = os.getenv(
            "TELEGRAM_BOT_TOKEN", ""
        ).strip()
        self.telegram_chat_id = os.getenv(
            "TELEGRAM_CHAT_ID", ""
        ).strip()

    def status(self):
        channels = []

        if (
            self.whatsapp_token
            and self.whatsapp_phone_number_id
            and self.whatsapp_to
        ):
            channels.append("WhatsApp")

        if (
            self.telegram_token
            and self.telegram_chat_id
        ):
            channels.append("Telegram")

        if not channels:
            return (
                "Retail notifications: "
                "console fallback only"
            )

        return (
            "Retail notifications: "
            + ", ".join(channels)
            + " configured"
        )

    def send_staff_alert(self, message):
        sent = False

        if (
            self.whatsapp_token
            and self.whatsapp_phone_number_id
            and self.whatsapp_to
        ):
            sent = self._send_whatsapp(message) or sent

        if (
            self.telegram_token
            and self.telegram_chat_id
        ):
            sent = self._send_telegram(message) or sent

        if not sent:
            print("[RETAIL STAFF NOTIFICATION]")
            print(message)

        return sent

    def _send_whatsapp(self, message):
        url = (
            "https://graph.facebook.com/v23.0/"
            f"{self.whatsapp_phone_number_id}/messages"
        )

        payload = json.dumps({
            "messaging_product": "whatsapp",
            "to": self.whatsapp_to,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": message,
            },
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": (
                    f"Bearer {self.whatsapp_token}"
                ),
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:
                return 200 <= response.status < 300
        except Exception as exc:
            print(
                "[WARNING] WhatsApp staff alert failed: "
                f"{exc}"
            )
            return False

    def _send_telegram(self, message):
        url = (
            "https://api.telegram.org/bot"
            f"{self.telegram_token}/sendMessage"
        )

        payload = json.dumps({
            "chat_id": self.telegram_chat_id,
            "text": message,
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:
                return 200 <= response.status < 300
        except Exception as exc:
            print(
                "[WARNING] Telegram staff alert failed: "
                f"{exc}"
            )
            return False
