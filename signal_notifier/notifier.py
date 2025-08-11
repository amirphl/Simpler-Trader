from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from candle_downloader.binance import BinanceClient, interval_to_milliseconds
from candle_downloader.models import Candle, to_milliseconds

from .engulfing_logic import EngulfingSignal, EngulfingSignalDetector
from .telegram_client import TelegramClient


@dataclass(frozen=True)
class SignalNotifierSettings:
    timeframe: str
    symbols: Optional[List[str]] = None
    top_symbols: int = 100
    history_limit: int = 100
    poll_epsilon_minutes: float = 0.1
    state_file: Optional[Path] = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not (10 <= self.history_limit <= 100):
            raise ValueError("history_limit must stay within 10..100 to limit memory usage.")
        if self.poll_epsilon_minutes < 0:
            raise ValueError("poll_epsilon_minutes cannot be negative.")


class SignalNotifier:
    """Continuously evaluates live candles and pushes engulfing signals to Telegram."""

    def __init__(
        self,
        *,
        binance_client: BinanceClient,
        telegram_client: TelegramClient,
        detector: EngulfingSignalDetector,
        settings: SignalNotifierSettings,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._binance = binance_client
        self._telegram = telegram_client
        self._detector = detector
        self._settings = settings
        self._log = logger or logging.getLogger(__name__)

        self._history: Dict[str, List[Candle]] = {}
        self._interval_ms = interval_to_milliseconds(self._settings.timeframe)
        self._interval_seconds = self._interval_ms / 1000.0
        self._epsilon_seconds = self._settings.poll_epsilon_minutes * 60.0
        self._strategy_name = "engulfing"

    def run(self) -> None:
        symbols = self._resolve_symbols()
        if not symbols:
            raise RuntimeError("No symbols available to monitor.")

        self._bootstrap_histories(symbols)
        wait_seconds = self._interval_seconds + self._epsilon_seconds
        self._log.info(
            "Monitoring %s symbols on %s timeframe (poll every %.1fs)",
            len(symbols),
            self._settings.timeframe,
            wait_seconds,
        )

        next_poll = time.monotonic() + wait_seconds
        try:
            while True:
                sleep_for = next_poll - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

                sent = 0
                for symbol in symbols:
                    try:
                        sent += self._process_symbol(symbol)
                    except Exception:  # pragma: no cover - defensive logging
                        self._log.exception("Failed to process %s", symbol)

                if sent:
                    self._log.info("Dispatched %s signal(s) this cycle.", sent)
                next_poll = time.monotonic() + wait_seconds
        except KeyboardInterrupt:
            self._log.info("Signal notifier stopped by user.")

    # ------------------------------------------------------------------ #
    # Candle management
    # ------------------------------------------------------------------ #

    def _resolve_symbols(self) -> List[str]:
        if self._settings.symbols:
            return [symbol.strip().upper() for symbol in self._settings.symbols if symbol.strip()]
        return self._binance.fetch_top_symbols(self._settings.top_symbols)

    def _bootstrap_histories(self, symbols: List[str]) -> None:
        for symbol in symbols:
            try:
                candles = self._load_initial_candles(symbol)
            except Exception:
                self._log.exception("Unable to load initial candles for %s", symbol)
                self._history[symbol] = []
                continue

            if not candles:
                self._log.warning("No historical candles available for %s.", symbol)
                self._history[symbol] = []
                continue

            self._history[symbol] = candles[-self._settings.history_limit :]

    def _load_initial_candles(self, symbol: str) -> List[Candle]:
        if self._settings.history_limit <= 0:
            return []
        end_ms = to_milliseconds(datetime.now(timezone.utc))
        span = self._interval_ms * self._settings.history_limit
        start_ms = max(0, end_ms - span)
        if start_ms >= end_ms:
            return []
        return self._binance.fetch_klines(
            symbol=symbol,
            interval=self._settings.timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=self._settings.history_limit,
        )

    def _process_symbol(self, symbol: str) -> int:
        history = self._history.setdefault(symbol, [])
        if not history:
            # Try to backfill if bootstrap failed earlier.
            history.extend(self._load_initial_candles(symbol))
            history[:] = history[-self._settings.history_limit :]

        last_open_ms = history[-1].open_time_ms if history else None
        new_candles = self._fetch_new_candles(symbol, last_open_ms)
        if not new_candles:
            return 0

        sent = 0
        for candle in new_candles:
            history.append(candle)
            signal = self._detector.evaluate(symbol, history)
            if signal and self._should_send(symbol, signal.entry_time):
                self._dispatch(signal)
                sent += 1
            self._trim_history(history)
        return sent

    def _fetch_new_candles(self, symbol: str, last_open_ms: Optional[int]) -> List[Candle]:
        if last_open_ms is None:
            return []
        start_ms = last_open_ms + self._interval_ms
        now_ms = to_milliseconds(datetime.now(timezone.utc))
        if start_ms >= now_ms:
            return []

        # Allow catching up on at most two missed candles per poll.
        end_ms = now_ms
        limit = 3
        candles = self._binance.fetch_klines(
            symbol=symbol,
            interval=self._settings.timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
        )
        return [candle for candle in candles if candle.open_time_ms > last_open_ms]

    def _trim_history(self, history: List[Candle]) -> None:
        if len(history) > self._settings.history_limit:
            del history[0 : len(history) - self._settings.history_limit]

    # ------------------------------------------------------------------ #
    # State handling
    # ------------------------------------------------------------------ #

    def _state_key(self, symbol: str) -> str:
        return f"{self._strategy_name}::{symbol}::{self._settings.timeframe}"

    def _load_state(self) -> Dict[str, str]:
        path = self._settings.state_file
        if not path or not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            self._log.warning("State file corrupted, ignoring contents.")
            return {}

    def _save_state(self, state: Dict[str, str]) -> None:
        path = self._settings.state_file
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def _should_send(self, symbol: str, entry_time: datetime) -> bool:
        if not self._settings.state_file:
            return True
        state = self._load_state()
        raw = state.get(self._state_key(symbol))
        if raw:
            try:
                last_entry = datetime.fromisoformat(raw)
            except ValueError:
                last_entry = None
            if last_entry and entry_time <= last_entry:
                return False
        state[self._state_key(symbol)] = entry_time.isoformat()
        self._save_state(state)
        return True

    # ------------------------------------------------------------------ #
    # Messaging
    # ------------------------------------------------------------------ #

    def _dispatch(self, signal: EngulfingSignal) -> None:
        message = self._format_signal(signal)
        if self._settings.dry_run:
            self._log.info("DRY RUN - Would send signal:\n%s", message)
            return
        self._telegram.send_message(message)

    def _format_signal(self, signal: EngulfingSignal) -> str:
        lines = [
            f"Symbol: {signal.symbol} ({signal.timeframe})",
            f"Direction: LONG",
            f"Entry Time: {signal.entry_time.isoformat()}",
            f"Entry: {signal.entry_price:.6g}",
            f"Stop Loss: {signal.stop_loss:.6g}",
            f"Take Profit: {signal.take_profit:.6g}",
            f"Leverage: {signal.leverage}",
        ]
        if signal.notes:
            lines.append(f"Notes: {signal.notes}")
        return "\n".join(lines)

