from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import time
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener


@dataclass(frozen=True)
class TelegramConfig:
    """Configuration for sending messages to Telegram."""

    bot_token: str
    chat_id: str
    proxy: str | None = None
    timeout: float = 10.0
    max_retries: int = 5
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    retry_backoff_multiplier: float = 2.0

    def as_proxy_dict(self) -> Dict[str, str] | None:
        if not self.proxy:
            return None
        proxy = self.proxy.strip()
        if not proxy:
            return None
        return {"http": proxy, "https": proxy}


class TelegramClient:
    """Thin client for Telegram Bot API."""

    def __init__(
        self, config: TelegramConfig, logger: logging.Logger | None = None
    ) -> None:
        self._config = config
        self._log = logger or logging.getLogger(__name__)
        handlers = []
        proxy_dict = config.as_proxy_dict()
        if proxy_dict:
            handlers.append(ProxyHandler(proxy_dict))
        self._opener = build_opener(*handlers)

    def send_message(self, text: str) -> None:
        payload = {
            "chat_id": self._config.chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
        encoded = urlencode(payload).encode("utf-8")
        request = Request(
            f"https://api.telegram.org/bot{self._config.bot_token}/sendMessage",
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        delay = self._config.initial_retry_delay
        last_exception: Exception | None = None

        for attempt in range(self._config.max_retries):
            try:
                with self._opener.open(
                    request, timeout=self._config.timeout
                ) as response:
                    body = response.read()
                data = json.loads(body)
                if data.get("ok"):
                    self._log.info(
                        "Sent signal to Telegram chat %s", self._config.chat_id
                    )
                    return
                description = data.get("description", "unknown error")
                parameters = data.get("parameters", {}) or {}
                retry_after = parameters.get("retry_after")
                should_retry = (
                    data.get("error_code") in (429,)
                    or "Too Many Requests" in description
                )
                if should_retry and retry_after:
                    delay = max(delay, float(retry_after))
                if not should_retry or attempt == self._config.max_retries - 1:
                    raise RuntimeError(f"Telegram API error: {description}")
                self._log.warning(
                    "Telegram API error (%s): %s. Retrying in %.1fs...",
                    data.get("error_code"),
                    description,
                    delay,
                )
            except HTTPError as exc:
                last_exception = exc
                body = (
                    exc.read().decode("utf-8", errors="ignore")
                    if hasattr(exc, "read")
                    else ""
                )
                description = body
                try:
                    payload = json.loads(body)
                    description = payload.get("description", body)
                except json.JSONDecodeError:
                    pass
                should_retry = exc.code >= 500 or exc.code in (429, 408)
                if not should_retry or attempt == self._config.max_retries - 1:
                    raise RuntimeError(
                        f"Telegram HTTP error {exc.code}: {description}"
                    ) from exc
                self._log.warning(
                    "Telegram HTTP error %s (%s), retrying in %.1fs...",
                    exc.code,
                    description,
                    delay,
                )
            except URLError as exc:
                last_exception = exc
                if attempt == self._config.max_retries - 1:
                    raise RuntimeError(
                        f"Telegram connection error: {exc.reason or exc}"
                    ) from exc
                self._log.warning(
                    "Telegram connection error '%s', retrying in %.1fs...",
                    exc.reason or exc,
                    delay,
                )
            except Exception as exc:
                last_exception = exc
                if attempt == self._config.max_retries - 1:
                    raise RuntimeError(f"Unexpected Telegram error: {exc}") from exc
                self._log.warning(
                    "Unexpected Telegram error '%s', retrying in %.1fs...", exc, delay
                )

            time.sleep(delay)
            delay = min(
                delay * self._config.retry_backoff_multiplier,
                self._config.max_retry_delay,
            )

        if last_exception:
            raise RuntimeError("Telegram send failed after retries") from last_exception
        raise RuntimeError("Telegram send failed after retries")
