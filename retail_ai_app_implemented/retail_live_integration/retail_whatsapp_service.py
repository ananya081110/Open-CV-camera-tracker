"""Optional WhatsApp alerting for retail operations.

Uses Twilio's WhatsApp sender when credentials are configured. The service is
safe to leave disabled: the AI pipeline continues to run and dashboard alerts
still work locally.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from typing import Iterable


class RetailWhatsAppService:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        self.from_number = self._wa(os.getenv("TWILIO_WHATSAPP_FROM", "").strip())
        self.content_sid = os.getenv("TWILIO_WHATSAPP_CONTENT_SID", "").strip()
        self.enabled = os.getenv("RETAIL_WHATSAPP_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.recipients = self._parse_recipients(os.getenv("RETAIL_WHATSAPP_RECIPIENTS", ""))
        self.zone_recipients = self._parse_zone_recipients(os.getenv("RETAIL_ZONE_WHATSAPP", "{}"))
        self.last_error = ""
        self.last_sent_at = None
        self.sent_count = 0

    @staticmethod
    def _wa(number: str) -> str:
        if not number:
            return ""
        return number if number.startswith("whatsapp:") else f"whatsapp:{number}"

    @staticmethod
    def _parse_recipients(raw: str) -> list[str]:
        return [RetailWhatsAppService._wa(x.strip()) for x in raw.split(",") if x.strip()]

    def _parse_zone_recipients(self, raw: str) -> dict[str, list[str]]:
        try:
            data = json.loads(raw or "{}")
            if not isinstance(data, dict):
                return {}
            return {str(zone): [self._wa(str(x).strip()) for x in values if str(x).strip()] for zone, values in data.items() if isinstance(values, list)}
        except json.JSONDecodeError:
            self.last_error = "RETAIL_ZONE_WHATSAPP must be valid JSON"
            return {}

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.account_sid and self.auth_token and self.from_number)

    def recipients_for_zone(self, zone: str | None = None) -> list[str]:
        if zone and self.zone_recipients.get(zone):
            return self.zone_recipients[zone]
        return self.recipients

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "provider": "Twilio WhatsApp",
            "recipient_count": len(self.recipients),
            "zone_recipient_groups": len(self.zone_recipients),
            "template_configured": bool(self.content_sid),
            "sent_count": self.sent_count,
            "last_sent_at": self.last_sent_at,
            "last_error": self.last_error,
        }

    def send_alert(self, message: str, zone: str | None = None, variables: dict | None = None) -> dict:
        recipients = self.recipients_for_zone(zone)
        if not self.enabled:
            return {"sent": False, "reason": "disabled", "recipients": 0}
        if not self.configured:
            return {"sent": False, "reason": "not_configured", "recipients": len(recipients)}
        if not recipients:
            return {"sent": False, "reason": "no_recipients", "recipients": 0}

        results = []
        for to in recipients:
            try:
                payload = {
                    "From": self.from_number,
                    "To": to,
                }
                if self.content_sid:
                    payload["ContentSid"] = self.content_sid
                    payload["ContentVariables"] = json.dumps(variables or {"1": zone or "Store", "2": message})
                else:
                    payload["Body"] = message

                body = urllib.parse.urlencode(payload).encode("utf-8")
                url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
                token = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode()).decode()
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Authorization": f"Basic {token}", "Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
                results.append({"to": to, "sent": True})
                self.sent_count += 1
            except Exception as exc:
                self.last_error = str(exc)
                results.append({"to": to, "sent": False, "error": str(exc)})

        if any(r["sent"] for r in results):
            import time
            self.last_sent_at = time.time()
        return {"sent": any(r["sent"] for r in results), "results": results}
