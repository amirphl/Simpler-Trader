"""Live coordinator for the EMA + AVWAP pullback strategy."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Optional
from urllib.request import OpenerDirector, ProxyHandler, build_opener

from candle_downloader.models import Candle
from signal_notifier import TelegramClient

from ..exchange import Exchange
from ..models import TradingState
from .calculations import EmaAvwapCalculationMixin
from .config import Direction, EmaAvwapPullbackLiveConfig
from .data import EmaAvwapDataMixin
from .notifications import EmaAvwapNotificationMixin
from .persistence import EmaAvwapPersistenceMixin
from .positions import EmaAvwapPositionMixin
from .signals import EmaAvwapSignalMixin
from .state import _PendingEntryMeta, _PositionRuntime, _SetupState, _SymbolSnapshot


class EmaAvwapPullbackLiveCoordinator(
    EmaAvwapDataMixin,
    EmaAvwapSignalMixin,
    EmaAvwapPositionMixin,
    EmaAvwapCalculationMixin,
    EmaAvwapPersistenceMixin,
    EmaAvwapNotificationMixin,
):
    """Live coordinator for the EMA + AVWAP pullback strategy."""

    def __init__(
        self,
        exchange: Exchange,
        config: EmaAvwapPullbackLiveConfig | None = None,
        telegram_client: Optional[TelegramClient] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._exchange: Exchange = exchange
        self._cfg: EmaAvwapPullbackLiveConfig = config or EmaAvwapPullbackLiveConfig()
        self._telegram: TelegramClient | None = telegram_client
        self._log: logging.Logger = logger or logging.getLogger(
            name=self.__class__.__name__
        )
        self._state = TradingState()
        self._running = False
        self._last_closed_candle_time_by_symbol: Dict[str, datetime] = {}
        self._last_snapshot_by_symbol: Dict[str, _SymbolSnapshot] = {}
        self._active_setups: Dict[tuple[str, Direction], _SetupState] = {}
        self._last_price_by_setup_key: Dict[tuple[str, Direction], float] = {}
        self._pending_meta_by_key: Dict[str, _PendingEntryMeta] = {}
        self._position_runtime_by_symbol: Dict[str, _PositionRuntime] = {}
        self._position_miss_count_by_symbol: Dict[str, int] = {}
        self._last_tick_trailing_check_ts = 0.0
        self._thread_local = threading.local()
        self._binance_proxies = self._resolve_proxy_map()
        self._binance_opener: OpenerDirector = self._build_binance_opener()
        self._init_persistence()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        self._running = True
        self._log.info(
            "EmaAvwapPullback started (symbols=%s timeframe=%s)",
            ",".join(self._cfg.symbols),
            self._cfg.timeframe,
        )
        while self._running:
            now: datetime = datetime.now(tz=timezone.utc)
            try:
                self._maybe_process_new_candles(now)
                self._on_tick(now)
            except Exception as exc:
                self._log.error("EmaAvwapPullback loop error: %s", exc, exc_info=True)
            time.sleep(max(self._cfg.poll_interval_seconds, 0.2))

    def stop(self) -> None:
        self._running = False
        try:
            self._exchange.close()
        except Exception:
            self._log.debug("Exchange close failed during stop", exc_info=True)

    # ------------------------------------------------------------------
    # Main candle-close processing path
    # ------------------------------------------------------------------

    def _maybe_process_new_candles(self, now: datetime) -> None:
        latest_closed_by_symbol: Dict[str, Candle | None] = {}
        max_workers: int = min(8, max(1, len(self._cfg.symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            candle_futures: dict[Future[Candle | None], str] = {
                executor.submit(self._fetch_latest_closed_candle, symbol): symbol
                for symbol in self._cfg.symbols
            }
            for future in as_completed(fs=candle_futures):
                symbol: str = candle_futures[future]
                try:
                    latest_closed_by_symbol[symbol] = future.result()
                except Exception as exc:
                    self._log.warning(
                        "EmaAvwapPullback: latest candle fetch failed for %s: %s",
                        symbol,
                        exc,
                    )
                    latest_closed_by_symbol[symbol] = None

        new_symbols: list[str] = []
        for symbol in self._cfg.symbols:
            latest_closed: Candle | None = latest_closed_by_symbol.get(symbol)
            if latest_closed is None:
                continue
            last_seen: datetime | None = self._last_closed_candle_time_by_symbol.get(
                symbol
            )
            if last_seen is None or latest_closed.close_time > last_seen:
                self._last_closed_candle_time_by_symbol[symbol] = (
                    latest_closed.close_time
                )
                new_symbols.append(symbol)

        if not new_symbols:
            return

        snapshots: Dict[str, _SymbolSnapshot] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(new_symbols))) as executor:
            snapshot_futures: dict[Future[_SymbolSnapshot | None], str] = {
                executor.submit(self._build_snapshot, symbol): symbol
                for symbol in new_symbols
            }
            for future in as_completed(fs=snapshot_futures):
                symbol: str = snapshot_futures[future]
                try:
                    snapshot: _SymbolSnapshot | None = future.result()
                except Exception as exc:
                    self._log.warning(
                        "EmaAvwapPullback: snapshot build failed for %s: %s",
                        symbol,
                        exc,
                    )
                    continue
                if snapshot is not None:
                    snapshots[symbol] = snapshot

        if not snapshots:
            return

        self._last_snapshot_by_symbol.update(snapshots)
        self._log.info("New AVWAP candle processed for %d symbol(s)", len(snapshots))
        self._sync_positions(now)

        for snapshot in snapshots.values():
            self._manage_position_on_bar(snapshot, now)

        for snapshot in snapshots.values():
            self._cancel_stale_entries(snapshot, now)

        for snapshot in snapshots.values():
            self._process_signal_state(snapshot, now)

        self._activate_due_entries(now)
        self._sync_positions(now)

    def _on_tick(self, now: datetime) -> None:
        self._sync_positions(now)
        self._process_live_setup_crosses(now)
        self._activate_due_entries(now)
        if (
            time.time() - self._last_tick_trailing_check_ts
        ) < self._cfg.trailing_check_interval_seconds:
            return
        self._last_tick_trailing_check_ts: int | float = time.time()
        self._manage_tick_trailing(now)

    # ------------------------------------------------------------------
    # Public fallback HTTP transport
    # ------------------------------------------------------------------

    def _resolve_proxy_map(self) -> dict[str, str]:
        exchange_config = getattr(self._exchange, "_config", None)
        configured = getattr(exchange_config, "proxies", None)
        if configured:
            proxies = {
                str(key): str(value)
                for key, value in dict(configured).items()
                if str(value).strip()
            }
            if "http" in proxies and "https" not in proxies:
                proxies["https"] = proxies["http"]
            return proxies

        all_proxy = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

        proxies: dict[str, str] = {}
        if all_proxy:
            proxies["http"] = all_proxy
            proxies["https"] = all_proxy
        else:
            if http_proxy:
                proxies["http"] = http_proxy
            if https_proxy:
                proxies["https"] = https_proxy
            elif http_proxy:
                proxies["https"] = http_proxy
        return proxies

    def _build_binance_opener(self) -> OpenerDirector:
        if self._binance_proxies:
            return build_opener(ProxyHandler(self._binance_proxies))
        return build_opener()

    def _get_binance_opener(self) -> OpenerDirector:
        opener = getattr(self._thread_local, "binance_opener", None)
        if opener is None:
            opener = self._build_binance_opener()
            self._thread_local.binance_opener = opener
        return opener

    # ------------------------------------------------------------------
