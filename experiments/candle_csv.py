from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from candle_downloader.models import Candle, to_datetime

CSV_HEADERS = [
    "symbol",
    "interval",
    "open_time_ms",
    "close_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def write_candles_to_csv(path: str | Path, candles: Sequence[Candle]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADERS)
        for candle in candles:
            writer.writerow(candle.as_row())


def read_candles_from_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
    interval: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[Candle]:
    file_path = Path(path)
    candles: list[Candle] = []
    with file_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = set(CSV_HEADERS)
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain headers: {', '.join(CSV_HEADERS)}")

        for row in reader:
            row_symbol = row["symbol"].strip()
            row_interval = row["interval"].strip()
            open_time_ms = int(row["open_time_ms"])
            close_time_ms = int(row["close_time_ms"])

            if symbol and row_symbol != symbol:
                continue
            if interval and row_interval != interval:
                continue
            if start_ms is not None and open_time_ms < start_ms:
                continue
            if end_ms is not None and open_time_ms >= end_ms:
                continue

            candles.append(
                Candle(
                    symbol=row_symbol,
                    interval=row_interval,
                    open_time=to_datetime(open_time_ms),
                    close_time=to_datetime(close_time_ms),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )

    candles.sort(key=lambda candle: candle.open_time_ms)
    return candles
