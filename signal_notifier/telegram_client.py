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

    def __post_init__(self) -> None:
        token = self.bot_token.strip()
        chat_id = self.chat_id.strip()
        if not token:
            raise ValueError("Telegram bot token must not be empty")
        if not chat_id:
            raise ValueError("Telegram chat_id must not be empty")
        if self.timeout <= 0:
            raise ValueError("Telegram timeout must be positive")
        if self.max_retries < 1:
            raise ValueError("Telegram max_retries must be at least 1")
        if self.initial_retry_delay <= 0:
            raise ValueError("Telegram initial_retry_delay must be positive")
        if self.max_retry_delay <= 0:
            raise ValueError("Telegram max_retry_delay must be positive")
        if self.retry_backoff_multiplier < 1:
            raise ValueError("Telegram retry_backoff_multiplier must be >= 1")

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
        self._send_message_url = (
            f"https://api.telegram.org/bot{self._config.bot_token}/sendMessage"
        )
        handlers = []
        proxy_dict = config.as_proxy_dict()
        if proxy_dict:
            handlers.append(ProxyHandler(proxy_dict))
        self._opener = build_opener(*handlers)

    @staticmethod
    def _extract_retry_after_seconds(payload: dict) -> float | None:
        parameters = payload.get("parameters", {}) or {}
        retry_after = parameters.get("retry_after")
        if retry_after is None:
            return None
        try:
            value = float(retry_after)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    def send_message(self, text: str) -> None:
        message = str(text)
        if not message.strip():
            raise ValueError("Telegram message text must not be empty")

        payload = {
            "chat_id": self._config.chat_id,
            "text": message,
            "disable_web_page_preview": "true",
        }

        delay = self._config.initial_retry_delay
        last_exception: Exception | None = None

        for attempt in range(self._config.max_retries):
            try:
                encoded = urlencode(payload).encode("utf-8")
                request = Request(
                    self._send_message_url,
                    data=encoded,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
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
                retry_after = self._extract_retry_after_seconds(data)
                should_retry = (
                    data.get("error_code") in (429,)
                    or "Too Many Requests" in description
                )
                if should_retry and retry_after is not None:
                    delay = max(delay, retry_after)
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
                retry_after: float | None = None
                try:
                    body_payload = json.loads(body)
                    description = body_payload.get("description", body)
                    retry_after = self._extract_retry_after_seconds(body_payload)
                except json.JSONDecodeError:
                    pass
                retry_after_header = exc.headers.get("Retry-After")
                if retry_after is None and retry_after_header:
                    try:
                        header_delay = float(retry_after_header)
                        if header_delay > 0:
                            retry_after = header_delay
                    except (TypeError, ValueError):
                        retry_after = None
                should_retry = exc.code >= 500 or exc.code in (429, 408)
                if should_retry and retry_after is not None:
                    delay = max(delay, retry_after)
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
