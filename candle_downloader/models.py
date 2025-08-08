from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Sequence


def to_datetime(milliseconds: int) -> datetime:
    """Convert milliseconds since epoch to an aware UTC datetime."""
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)


def to_milliseconds(moment: datetime) -> int:
    """Convert a datetime to integer milliseconds since epoch."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class Candle:
    """Domain model representing a single OHLCV candle."""

    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_binance(cls, symbol: str, interval: str, payload: Sequence[str | int | float]) -> "Candle":
        """Build a candle instance from the Binance kline payload."""
        open_time_ms = int(payload[0])
        close_time_ms = int(payload[6])
        return cls(
            symbol=symbol,
            interval=interval,
            open_time=to_datetime(open_time_ms),
            close_time=to_datetime(close_time_ms),
            open=float(payload[1]),
            high=float(payload[2]),
            low=float(payload[3]),
            close=float(payload[4]),
            volume=float(payload[5]),
        )

    @property
    def open_time_ms(self) -> int:
        return to_milliseconds(self.open_time)

    def as_row(self) -> List[str]:
        """Serialize the candle into a CSV-friendly row."""
        return [
            self.symbol,
            self.interval,
            str(self.open_time_ms),
            str(to_milliseconds(self.close_time)),
            f"{self.open:.10f}",
            f"{self.high:.10f}",
            f"{self.low:.10f}",
            f"{self.close:.10f}",
            f"{self.volume:.10f}",
        ]


def normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip()


