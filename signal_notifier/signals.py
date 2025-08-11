from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class SignalSpec:
    """Normalized signal data fed into notifiers."""

    timestamp: datetime
    symbol: str
    timeframe: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    leverage: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

