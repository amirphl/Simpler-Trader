#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from candle_downloader.binance import BinanceClient, BinanceClientConfig, MAX_BATCH
from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle

try:
    import psycopg
except ImportError:
    psycopg = None  # type: ignore[assignment]


def parse_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ms(moment: datetime) -> int:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def align_down(ms: int, interval_ms: int) -> int:
    return (ms // interval_ms) * interval_ms


def expected_times(start_ms: int, end_ms: int, interval_ms: int) -> Iterable[int]:
    cursor = start_ms
    while cursor < end_ms:
        yield cursor
        cursor += interval_ms


def fetch_open_times_sqlite(
    path: str, symbol: str, timeframe: str, start_ms: int, end_ms: int
) -> set[int]:
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            """
            SELECT open_time
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            ORDER BY open_time ASC
            """,
            (symbol, timeframe, start_ms, end_ms),
        )
        return {int(row[0]) for row in cur.fetchall()}
    finally:
        conn.close()


def fetch_open_times_postgres(
    dsn: str, symbol: str, timeframe: str, start_ms: int, end_ms: int
) -> set[int]:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is not installed. Install requirements first: pip install -r requirements.txt"
        )
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT open_time
                FROM candles
                WHERE symbol = %s AND interval = %s AND open_time >= %s AND open_time < %s
                ORDER BY open_time ASC
                """,
                (symbol, timeframe, start_ms, end_ms),
            )
            return {int(row[0]) for row in cur.fetchall()}


def fetch_candles_sqlite(
    path: str, symbol: str, timeframe: str, start_ms: int, end_ms: int
) -> Dict[int, Candle]:
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            """
            SELECT symbol, interval, open_time, close_time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            ORDER BY open_time ASC
            """,
            (symbol, timeframe, start_ms, end_ms),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out: Dict[int, Candle] = {}
    for row in rows:
        candle = Candle(
            symbol=row[0],
            interval=row[1],
            open_time=datetime.fromtimestamp(int(row[2]) / 1000, tz=timezone.utc),
            close_time=datetime.fromtimestamp(int(row[3]) / 1000, tz=timezone.utc),
            open=float(row[4]),
            high=float(row[5]),
            low=float(row[6]),
            close=float(row[7]),
            volume=float(row[8]),
        )
        out[candle.open_time_ms] = candle
    return out


def fetch_candles_postgres(
    dsn: str, symbol: str, timeframe: str, start_ms: int, end_ms: int
) -> Dict[int, Candle]:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is not installed. Install requirements first: pip install -r requirements.txt"
        )
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, interval, open_time, close_time, open, high, low, close, volume
                FROM candles
                WHERE symbol = %s AND interval = %s AND open_time >= %s AND open_time < %s
                ORDER BY open_time ASC
                """,
                (symbol, timeframe, start_ms, end_ms),
            )
            rows = cur.fetchall()

    out: Dict[int, Candle] = {}
    for row in rows:
        candle = Candle(
            symbol=row[0],
            interval=row[1],
            open_time=datetime.fromtimestamp(int(row[2]) / 1000, tz=timezone.utc),
            close_time=datetime.fromtimestamp(int(row[3]) / 1000, tz=timezone.utc),
            open=float(row[4]),
            high=float(row[5]),
            low=float(row[6]),
            close=float(row[7]),
            volume=float(row[8]),
        )
        out[candle.open_time_ms] = candle
    return out


def build_proxies(args: argparse.Namespace) -> Dict[str, str] | None:
    proxies: Dict[str, str] = {}
    if args.proxy:
        proxies["http"] = args.proxy
        proxies["https"] = args.proxy
    if args.http_proxy:
        proxies["http"] = args.http_proxy
    if args.https_proxy:
        proxies["https"] = args.https_proxy
    return proxies if proxies else None


def fetch_binance_candles(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    proxies: Dict[str, str] | None,
) -> Dict[int, Candle]:
    client = BinanceClient(BinanceClientConfig(proxies=proxies))
    try:
        out: Dict[int, Candle] = {}
        cursor = start_ms
        while cursor < end_ms:
            remaining = max((end_ms - cursor) // interval_ms, 1)
            limit = min(MAX_BATCH, remaining)
            chunk_end = cursor + limit * interval_ms
            candles = client.fetch_klines(
                symbol=symbol,
                interval=timeframe,
                start_ms=cursor,
                end_ms=chunk_end,
                limit=limit,
            )
            if not candles:
                cursor = chunk_end
                continue
            for candle in candles:
                if start_ms <= candle.open_time_ms < end_ms:
                    out[candle.open_time_ms] = candle
            next_cursor = candles[-1].open_time_ms + interval_ms
            cursor = next_cursor if next_cursor > cursor else chunk_end
        return out
    finally:
        client.close()


def build_postgres_dsn(args: argparse.Namespace) -> str:
    if args.postgres_dsn:
        return args.postgres_dsn
    return (
        f"host={args.pg_host} "
        f"port={args.pg_port} "
        f"dbname={args.pg_db} "
        f"user={args.pg_user} "
        f"password={args.pg_password} "
        f"sslmode={args.pg_sslmode}"
    )


def print_missing(missing: Sequence[int], preview: int) -> None:
    if not missing:
        print("No missing candles found.")
        return
    print(f"Missing candles: {len(missing)}")
    for ts in missing[:preview]:
        print(f"- {ms_to_iso(ts)}")
    if len(missing) > preview:
        print(f"... and {len(missing) - preview} more")


def candles_equal(a: Candle, b: Candle, epsilon: float) -> bool:
    return (
        a.symbol == b.symbol
        and a.interval == b.interval
        and a.open_time_ms == b.open_time_ms
        and abs(a.close_time.timestamp() - b.close_time.timestamp()) <= (epsilon / 1000)
        and abs(a.open - b.open) <= epsilon
        and abs(a.high - b.high) <= epsilon
        and abs(a.low - b.low) <= epsilon
        and abs(a.close - b.close) <= epsilon
        and abs(a.volume - b.volume) <= epsilon
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check missing candles in candles table for a symbol/timeframe/date range."
    )
    parser.add_argument("--db-kind", choices=("sqlite", "postgres"), required=True)
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--timeframe", required=True, help="e.g. 1m, 15m, 1h, 1d")
    parser.add_argument("--start-date", required=True, help="ISO8601 start datetime")
    parser.add_argument("--end-date", required=True, help="ISO8601 end datetime (exclusive)")
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="How many missing timestamps to print",
    )
    parser.add_argument(
        "--redownload-from-binance",
        action="store_true",
        help="Fetch candles from Binance for full range and compare one-by-one with DB candles",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Float comparison tolerance for one-by-one comparison",
    )

    parser.add_argument(
        "--sqlite-path",
        default="./data/candles.db",
        help="SQLite file path (when --db-kind sqlite)",
    )

    parser.add_argument(
        "--postgres-dsn",
        default="",
        help="Full Postgres DSN/conninfo string (overrides --pg-*)",
    )
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default="postgres")
    parser.add_argument("--pg-db", default="scalp_test")
    parser.add_argument("--pg-sslmode", default="prefer")
    parser.add_argument("--http-proxy", default="")
    parser.add_argument("--https-proxy", default="")
    parser.add_argument("--proxy", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    symbol = args.symbol.strip().upper()
    timeframe = args.timeframe.strip()
    interval_ms = interval_to_milliseconds(timeframe)

    start_ms = align_down(to_ms(parse_datetime(args.start_date)), interval_ms)
    end_ms = align_down(to_ms(parse_datetime(args.end_date)), interval_ms)
    if end_ms <= start_ms:
        print("Error: end_date must be after start_date (after alignment).", file=sys.stderr)
        return 2

    if args.db_kind == "sqlite":
        actual = fetch_open_times_sqlite(
            args.sqlite_path, symbol, timeframe, start_ms, end_ms
        )
        db_candles = fetch_candles_sqlite(
            args.sqlite_path, symbol, timeframe, start_ms, end_ms
        )
    else:
        dsn = build_postgres_dsn(args)
        actual = fetch_open_times_postgres(dsn, symbol, timeframe, start_ms, end_ms)
        db_candles = fetch_candles_postgres(dsn, symbol, timeframe, start_ms, end_ms)

    expected = set(expected_times(start_ms, end_ms, interval_ms))
    missing = sorted(expected - actual)

    print(f"DB kind: {args.db_kind}")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Range: {ms_to_iso(start_ms)} -> {ms_to_iso(end_ms)}")
    print(f"Expected candles: {len(expected)}")
    print(f"Existing candles: {len(actual)}")
    print_missing(missing, preview=max(args.preview, 0))

    mismatches = 0
    if args.redownload_from_binance:
        proxies = build_proxies(args)
        binance_candles = fetch_binance_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            interval_ms=interval_ms,
            proxies=proxies,
        )
        print(f"Binance candles fetched: {len(binance_candles)}")
        missing_in_db_vs_binance = sorted(set(binance_candles) - set(db_candles))
        missing_in_binance_vs_db = sorted(set(db_candles) - set(binance_candles))
        if missing_in_db_vs_binance:
            print(
                f"Missing in DB compared to Binance: {len(missing_in_db_vs_binance)}"
            )
            for ts in missing_in_db_vs_binance[: max(args.preview, 0)]:
                print(f"- {ms_to_iso(ts)}")
        if missing_in_binance_vs_db:
            print(
                f"Missing in Binance compared to DB: {len(missing_in_binance_vs_db)}"
            )
            for ts in missing_in_binance_vs_db[: max(args.preview, 0)]:
                print(f"- {ms_to_iso(ts)}")

        shared = sorted(set(db_candles) & set(binance_candles))
        for ts in shared:
            if not candles_equal(db_candles[ts], binance_candles[ts], args.epsilon):
                mismatches += 1
                if mismatches <= max(args.preview, 0):
                    db_c = db_candles[ts]
                    bi_c = binance_candles[ts]
                    print(f"Mismatch @ {ms_to_iso(ts)}")
                    print(
                        "  DB      "
                        f"o={db_c.open} h={db_c.high} l={db_c.low} c={db_c.close} v={db_c.volume}"
                    )
                    print(
                        "  Binance "
                        f"o={bi_c.open} h={bi_c.high} l={bi_c.low} c={bi_c.close} v={bi_c.volume}"
                    )
        if mismatches:
            print(f"One-by-one mismatches: {mismatches}")
        else:
            print("One-by-one check: all shared candles match.")

    return 1 if (missing or mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
