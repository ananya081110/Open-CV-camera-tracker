import os
import smtplib
import urllib.parse
import urllib.request
import json
from email.message import EmailMessage


class NotificationService:
    """
    Sends administrator security notifications.

    Supported:
    - Telegram
    - Email / SMTP
    - Console fallback
    """

    def __init__(self):
        self.telegram_token = os.getenv(
            "TELEGRAM_BOT_TOKEN", ""
        ).strip()

        self.telegram_chat_id = os.getenv(
            "TELEGRAM_CHAT_ID", ""
        ).strip()

        self.smtp_host = os.getenv(
            "SMTP_HOST", ""
        ).strip()

        self.smtp_port = int(
            os.getenv("SMTP_PORT", "587")
        )

        self.smtp_username = os.getenv(
            "SMTP_USERNAME", ""
        ).strip()

        self.smtp_password = os.getenv(
            "SMTP_PASSWORD", ""
        ).strip()

        self.admin_email = os.getenv(
            "ADMIN_EMAIL", ""
        ).strip()

    def send(
        self,
        title,
        message,
        severity="HIGH"
    ):
        results = []

        # =====================================================
        # TELEGRAM
        # =====================================================

        if (
            self.telegram_token
            and self.telegram_chat_id
        ):
            try:

                self._send_telegram(
                    title,
                    message,
                    severity
                )

                results.append(
                    "TELEGRAM:SENT"
                )

            except Exception as exc:

                results.append(
                    f"TELEGRAM:FAILED:{exc}"
                )

        # =====================================================
        # EMAIL
        # =====================================================

        if (
            self.smtp_host
            and self.smtp_username
            and self.smtp_password
            and self.admin_email
        ):
            try:

                self._send_email(
                    title,
                    message,
                    severity
                )

                results.append(
                    "EMAIL:SENT"
                )

            except Exception as exc:

                results.append(
                    f"EMAIL:FAILED:{exc}"
                )

        # =====================================================
        # CONSOLE FALLBACK
        # =====================================================

        if not results:

            print(
                "\n"
                "========== SECURITY ALERT ==========\n"
                f"Severity: {severity}\n"
                f"{title}\n\n"
                f"{message}\n"
                "====================================\n"
            )

            return "CONSOLE"

        return " | ".join(results)

    # =========================================================
    # TELEGRAM
    # =========================================================

    def _send_telegram(
        self,
        title,
        message,
        severity
    ):

        text = (
            "🚨 AI CAMERA TRACKER\n\n"
            f"Severity: {severity}\n"
            f"{title}\n\n"
            f"{message}"
        )

        query = urllib.parse.urlencode(
            {
                "chat_id": self.telegram_chat_id,
                "text": text,
            }
        )

        url = (
            "https://api.telegram.org/bot"
            f"{self.telegram_token}"
            f"/sendMessage?{query}"
        )

        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "User-Agent":
                "AI-Camera-Tracker/1.0"
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:

            payload = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        if not payload.get("ok"):
            raise RuntimeError(payload)

    # =========================================================
    # EMAIL
    # =========================================================

    def _send_email(
        self,
        title,
        message,
        severity
    ):

        email = EmailMessage()

        email["Subject"] = (
            f"[{severity}] "
            f"AI Camera Tracker - "
            f"{title}"
        )

        email["From"] = (
            self.smtp_username
        )

        email["To"] = (
            self.admin_email
        )

        email.set_content(
            "AI Camera Tracker "
            "Security Alert\n\n"
            f"Severity: {severity}\n"
            f"Alert: {title}\n\n"
            f"{message}\n"
        )

        with smtplib.SMTP(
            self.smtp_host,
            self.smtp_port,
            timeout=15
        ) as server:

            server.starttls()

            server.login(
                self.smtp_username,
                self.smtp_password
            )

            server.send_message(email)