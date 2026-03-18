#!/usr/bin/env python3
"""
Alertmanager Telegram Webhook Receiver
Forwards alerts from Alertmanager to Telegram bot
"""

import json
import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT = int(os.environ.get("PORT", 9080))


def escape_markdown(text: str) -> str:
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in escape_chars:
        text = text.replace(char, f"\\{char}")
    return text


def format_alert(alert: dict[str, Any]) -> str:
    status = alert.get("status", "unknown")
    alertname = escape_markdown(alert.get("labels", {}).get("alertname", "Unknown"))
    severity = alert.get("labels", {}).get("severity", "info")
    description = alert.get("annotations", {}).get("description", "")
    summary = alert.get("annotations", {}).get("summary", "")

    emoji = "🔴" if severity == "critical" else "🟡" if severity == "warning" else "🔵"
    status_emoji = "🚨" if status == "firing" else "✅"

    message = f"{status_emoji} *{status.upper()}* {emoji}\n"
    message += f"*Alert:* `{alertname}`\n"
    message += f"*Severity:* {severity.upper()}\n"

    if summary:
        message += f"*Summary:* {escape_markdown(summary)}\n"
    if description:
        message += f"*Details:* {escape_markdown(description)}\n"

    return message


def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "MarkdownV2",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


class AlertmanagerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            alerts = data.get("alerts", [])

            if not alerts:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
                return

            messages = []
            for alert in alerts:
                if alert.get("status") == "firing":
                    messages.append(format_alert(alert))

            if messages:
                full_message = "\n---\n".join(messages)[:4096]
                send_telegram_message(full_message)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            logger.info(f"Processed {len(alerts)} alerts")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Alertmanager Telegram Receiver is running")


def main():
    server = HTTPServer(("0.0.0.0", PORT), AlertmanagerHandler)
    logger.info(f"Starting Alertmanager Telegram receiver on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
