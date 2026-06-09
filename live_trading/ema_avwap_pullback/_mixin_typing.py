"""Typing helpers for dynamically composed EMA + AVWAP mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


class EmaAvwapMixinTyping:
    if TYPE_CHECKING:
        def __getattr__(self, name: str) -> Any: ...
