"""Live coordinator for the EMA + AVWAP pullback strategy."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

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
        self._latest_kline_rows_by_symbol: Dict[str, list[list]] = {}
        self._active_setups: Dict[tuple[str, Direction], _SetupState] = {}
        self._last_price_by_setup_key: Dict[tuple[str, Direction], float] = {}
        self._last_middle_by_setup_key: Dict[tuple[str, Direction], float] = {}
        self._pending_meta_by_key: Dict[str, _PendingEntryMeta] = {}
        self._position_runtime_by_symbol: Dict[str, _PositionRuntime] = {}
        self._position_miss_count_by_symbol: Dict[str, int] = {}
        # A failed position sync must never be treated as evidence that the
        # account is flat.  This flag is cleared by the position mixin before
        # any new entry can be submitted again.
        self._position_sync_healthy = True
        self._state_lock_handles = []
        self._last_tick_trailing_check_ts = 0.0
        self._market_data_executor = ThreadPoolExecutor(
            max_workers=min(8, max(1, len(self._cfg.symbols))),
            thread_name_prefix="ema-avwap-market-data",
        )
        self._candle_fetch_futures: Dict[
            str, tuple[int, Future[Candle | None]]
        ] = {}
        self._snapshot_futures: Dict[
            str, tuple[Candle, Future[_SymbolSnapshot | None]]
        ] = {}
        self._last_market_data_poll_slot_by_symbol: Dict[str, int] = {}
        self._init_persistence()

    def __del__(self) -> None:
        # ``stop`` is the normal lifecycle path.  This small backstop prevents
        # a failed construction or an abandoned coordinator from retaining an
        # advisory process lock until interpreter shutdown.
        try:
            self._close_persistence()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        validator = getattr(self._exchange, "validate_ema_avwap_execution", None)
        if callable(validator):
            # The strategy owns one net position per symbol.  Bitunix hedge mode
            # violates that invariant, so reject it before polling or placing
            # anything rather than attempting to infer a position to manage.
            validator()
        self._running = True
        self._log.info(
            "EmaAvwapPullback started (symbols=%s timeframe=%s "
            "entry_mode=%s exit_mode=%s exit_band=%s rigid_stop_loss_pct=%.8f)",
            ",".join(self._cfg.symbols),
            self._cfg.timeframe,
            self._cfg.entry_mode.value,
            self._cfg.exit_mode.value,
            self._cfg.exit_band.value,
            self._cfg.rigid_stop_loss_pct,
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
        self._market_data_executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._exchange.close()
        except Exception:
            self._log.debug("Exchange close failed during stop", exc_info=True)
        finally:
            self._close_persistence()

    # ------------------------------------------------------------------
    # Main candle-close processing path
    # ------------------------------------------------------------------

    def _maybe_process_new_candles(self, now: datetime) -> None:
        snapshots: Dict[str, _SymbolSnapshot] = {}
        self._collect_completed_candle_fetches()
        self._collect_completed_snapshots(snapshots)

        if snapshots:
            self._process_new_snapshots(snapshots, now)
        self._schedule_market_data_fetches(now)

    def _collect_completed_candle_fetches(self) -> None:
        """Schedule snapshot construction only after completed venue-data polls.

        Results are consumed only when their futures are done.  A Bitunix
        timeout therefore cannot hold up position management in ``_on_tick``.
        """
        for symbol, (poll_slot, future) in list(
            self._candle_fetch_futures.items()
        ):
            if not future.done():
                continue
            del self._candle_fetch_futures[symbol]
            try:
                latest_closed = future.result()
            except Exception as exc:
                self._log.warning(
                    "EmaAvwapPullback: latest Bitunix candle fetch failed for %s: %s",
                    symbol,
                    exc,
                )
                continue
            if latest_closed is None or symbol in self._snapshot_futures:
                continue
            last_seen = self._last_closed_candle_time_by_symbol.get(symbol)
            if last_seen is not None and latest_closed.close_time <= last_seen:
                if self._latest_closed_covers_poll_slot(latest_closed, poll_slot):
                    self._last_market_data_poll_slot_by_symbol[symbol] = poll_slot
                continue
            self._snapshot_futures[symbol] = (
                latest_closed,
                self._market_data_executor.submit(self._build_snapshot, symbol),
            )

    def _collect_completed_snapshots(
        self, snapshots: Dict[str, _SymbolSnapshot]
    ) -> None:
        for symbol, (expected, future) in list(self._snapshot_futures.items()):
            if not future.done():
                continue
            del self._snapshot_futures[symbol]
            try:
                snapshot = future.result()
            except Exception as exc:
                self._log.warning(
                    "EmaAvwapPullback: Bitunix snapshot build failed for %s: %s",
                    symbol,
                    exc,
                )
                continue
            if snapshot is None:
                continue
            if snapshot.candle.close_time < expected.close_time:
                self._log.warning(
                    "EmaAvwapPullback: snapshot for %s is stale "
                    "(snapshot_close=%s latest_close=%s); leaving the candle "
                    "unprocessed",
                    symbol,
                    snapshot.candle.close_time.isoformat(),
                    expected.close_time.isoformat(),
                )
                continue
            snapshots[symbol] = snapshot

    def _schedule_market_data_fetches(self, now: datetime) -> None:
        for symbol in self._cfg.symbols:
            if not self._market_data_poll_is_due(now, symbol):
                continue
            if symbol in self._candle_fetch_futures or symbol in self._snapshot_futures:
                continue
            poll_slot = self._market_data_poll_slot(now)
            self._candle_fetch_futures[symbol] = (
                poll_slot,
                self._market_data_executor.submit(
                    self._fetch_latest_closed_candle, symbol
                ),
            )

    def _market_data_poll_slot(self, now: datetime) -> int:
        interval_seconds = min(
            self._cfg.execution_interval_minutes * 60,
            self._timeframe_seconds(self._cfg.timeframe),
        )
        delayed_timestamp = now.timestamp() - self._cfg.candle_ready_delay_seconds
        return int(delayed_timestamp // interval_seconds)

    def _latest_closed_covers_poll_slot(
        self, latest_closed: Candle, poll_slot: int
    ) -> bool:
        """Return whether the latest closed bar covers this scheduled poll.

        Bitunix reports kline ``closeTime`` as one millisecond before the next
        bar's open.  Compare the bar's canonical end boundary instead of that
        inclusive timestamp so a valid bar does not make every later poll in
        the slot look stale.
        """
        poll_interval_seconds = min(
            self._cfg.execution_interval_minutes * 60,
            self._timeframe_seconds(self._cfg.timeframe),
        )
        timeframe_seconds = self._timeframe_seconds(self._cfg.timeframe)
        delayed_poll_timestamp = poll_slot * poll_interval_seconds
        expected_close_time = datetime.fromtimestamp(
            int(delayed_poll_timestamp // timeframe_seconds) * timeframe_seconds,
            tz=timezone.utc,
        )
        candle_end_boundary = latest_closed.open_time + timedelta(
            seconds=timeframe_seconds
        )
        return candle_end_boundary >= expected_close_time

    def _market_data_poll_is_due(
        self, now: datetime, symbol: str | None = None
    ) -> bool:
        slot = self._market_data_poll_slot(now)
        if symbol is not None:
            return self._last_market_data_poll_slot_by_symbol.get(symbol) != slot
        return any(
            self._last_market_data_poll_slot_by_symbol.get(item) != slot
            for item in self._cfg.symbols
        )

    def _process_new_snapshots(
        self, snapshots: Dict[str, _SymbolSnapshot], now: datetime) -> None:

        for symbol, snapshot in snapshots.items():
            self._last_closed_candle_time_by_symbol[symbol] = (
                snapshot.candle.close_time
            )
            self._last_market_data_poll_slot_by_symbol[symbol] = (
                self._market_data_poll_slot(now)
            )
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
        self._manage_live_position_exits(now)

        # Targets and rigid stops are evaluated first above.  Once a position
        # remains open, update its trailing protection at the configured cadence.
        # ``monotonic`` prevents wall-clock changes from delaying or flooding
        # trailing checks.
        trailing_check_ts = time.monotonic()
        if (
            trailing_check_ts - self._last_tick_trailing_check_ts
            < self._cfg.trailing_check_interval_seconds
        ):
            return
        self._last_tick_trailing_check_ts = trailing_check_ts
        self._manage_tick_trailing(now)

    # ------------------------------------------------------------------
