"""Main live trading coordinator with scheduling."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from signal_notifier import TelegramClient

from .exchange import Exchange
from .models import LiveTradingConfig, TradingSignal
from .heiken_ashi_strategy import HeikenAshiLiveStrategy
from .position_manager import PositionManager
from .scanner import SymbolScanner


class LiveTradingCoordinator:
    """Coordinates live trading execution with timeframe alignment."""

    def __init__(
        self,
        config: LiveTradingConfig,
        exchange: Exchange,
        telegram_client: Optional[TelegramClient] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if config.strategy_name != "heiken_ashi":
            raise ValueError(
                "LiveTradingCoordinator only supports heiken_ashi strategy; "
                "use pinbar_magic_coordinator for pinbar_magic_v2."
            )

        self._config = config
        self._exchange = exchange
        self._log = logger or logging.getLogger(__name__)
        self._interval_seconds = config.execution_interval_minutes * 60
        self._candle_delay = timedelta(seconds=config.candle_ready_delay_seconds)
        # Keep shutdown responsive even when next run is far away.
        self._sleep_step_seconds = 5.0

        # Initialize components
        self._position_manager = PositionManager(
            config, exchange, logger, telegram_client=telegram_client
        )
        self._scanner = SymbolScanner(exchange, logger)

        self._strategy = HeikenAshiLiveStrategy(
            config,
            exchange,
            self._position_manager.get_state(),
            logger,
        )

        self._running = False

    def run(self) -> None:
        """Run the live trading loop."""
        self._running = True
        self._log.info("Live trading coordinator started")

        try:
            # Test connection
            if not self._exchange.test_connection():
                raise RuntimeError("Exchange connection test failed")

            self._log.info(f"Connected to {self._config.exchange_name} exchange")
            self._log.info(
                f"Execution interval: {self._config.execution_interval_minutes} minutes"
            )

            # Immediate first run
            current_time = datetime.now(timezone.utc)
            self._log.info(
                "Executing initial strategy cycle at %s", current_time.isoformat()
            )
            self._position_manager.update_positions(current_time)
            cycle_ok = self._execute_strategy_cycle(current_time)
            if cycle_ok:
                self._position_manager.update_execution_time(current_time)
            else:
                self._log.warning(
                    "Initial strategy cycle failed; last_execution_time not updated"
                )

            now_after_cycle = datetime.now(timezone.utc)
            self._position_manager.update_positions(now_after_cycle)
            self._position_manager.cleanup_state(now_after_cycle)

            next_run_time = self._next_aligned_time(now_after_cycle)
            self._log.info("Next execution at %s", next_run_time.isoformat())

            while self._running:
                try:
                    current_time = datetime.now(timezone.utc)

                    if current_time >= next_run_time:
                        scheduled_run_time = next_run_time
                        self._log.info(
                            "Executing strategy at %s (scheduled for %s)",
                            current_time.isoformat(),
                            scheduled_run_time.isoformat(),
                        )

                        self._position_manager.update_positions(current_time)
                        cycle_ok = self._execute_strategy_cycle(scheduled_run_time)
                        if cycle_ok:
                            self._position_manager.update_execution_time(
                                scheduled_run_time
                            )
                        else:
                            self._log.warning(
                                "Strategy cycle failed for scheduled run %s; "
                                "last_execution_time not updated",
                                scheduled_run_time.isoformat(),
                            )

                        now_after_cycle = datetime.now(timezone.utc)
                        self._position_manager.update_positions(now_after_cycle)
                        self._position_manager.cleanup_state(now_after_cycle)

                        next_run_time = self._next_aligned_time(now_after_cycle)
                        self._log.info(
                            "Next execution at %s", next_run_time.isoformat()
                        )
                        continue

                    sleep_seconds = min(
                        self._sleep_step_seconds,
                        max(0.2, (next_run_time - current_time).total_seconds()),
                    )
                    self._sleep_with_stop(sleep_seconds)

                except KeyboardInterrupt:
                    self._log.info("Received interrupt signal")
                    break
                except Exception as e:
                    self._log.error(f"Error in main loop: {e}", exc_info=True)
                    # Back off after unexpected loop errors without blocking stop().
                    self._sleep_with_stop(60.0)

        finally:
            self._running = False
            self._cleanup()

    def _next_aligned_time(self, current_time: datetime) -> datetime:
        """Align to the next interval boundary (divisible by execution_interval_minutes)."""
        # Compute next boundary in UTC to keep alignment consistent
        if current_time.tzinfo is None:
            current_utc = current_time.replace(tzinfo=timezone.utc)
        else:
            current_utc = current_time.astimezone(timezone.utc)
        ts = int(current_utc.timestamp())
        interval = max(self._interval_seconds, 1)
        next_boundary_ts = ((ts // interval) + 1) * interval
        aligned = datetime.fromtimestamp(next_boundary_ts, tz=timezone.utc)
        return aligned + self._candle_delay

    def _execute_strategy_cycle(self, current_time: datetime) -> bool:
        """Execute one complete strategy cycle."""
        try:
            # 1. Scan for top symbols by volume
            self._log.info(
                f"Scanning top {self._config.top_m_symbols} symbols by volume for timeframe {self._config.timeframe}"
            )
            top_symbols = self._scanner.scan_top_symbols_by_volume(
                current_time=current_time,
                limit=self._config.top_m_symbols,
                timeframe=self._config.timeframe,
            )

            if not top_symbols:
                self._log.warning("No symbols found in scan")
                return True

            # 2. Find top movers (gainers and losers)
            self._log.info(
                f"Finding top {self._config.top_n_signals} movers "
                f"(min change: {self._config.price_change_threshold_pct}%)"
            )
            top_gainers, top_losers = self._scanner.find_top_movers(
                symbols=top_symbols,
                top_n=self._config.top_n_signals,
                min_change_pct=self._config.price_change_threshold_pct,
            )

            self._log.info(
                f"Found {len(top_gainers)} gainers and {len(top_losers)} losers"
            )

            if not top_gainers and not top_losers:
                self._log.info("No significant movers found")
                return True

            # 3. Generate trading signals
            long_signals, short_signals = self._strategy.generate_signals(
                top_gainers=top_gainers,
                top_losers=top_losers,
                current_time=current_time,
            )

            # TODO:
            long_signals = long_signals[:20]
            short_signals = short_signals[:20]

            if not long_signals and not short_signals:
                self._log.info("No trading signals generated")
                return True
            # Interleave short and long signals: 1st short, 1st long, 2nd short, 2nd long, ...
            signals: list[TradingSignal] = []
            max_len = max(len(short_signals), len(long_signals))
            for i in range(max_len):
                if i < len(short_signals):
                    signals.append(short_signals[i])
                if i < len(long_signals):
                    signals.append(long_signals[i])

            # 4. Execute signals
            self._log.info(
                "Executing %s trading signals (%s long, %s short)",
                len(signals),
                len(long_signals),
                len(short_signals),
            )
            successful = 0
            for signal in signals:
                if self._position_manager.execute_signal(signal):
                    successful += 1
                    self._log.info(f"Successfully executed signal for {signal.symbol}")
                else:
                    self._log.warning(f"Failed to execute signal for {signal.symbol}")

            self._log.info(
                f"Strategy cycle completed: {successful}/{len(signals)} signals executed"
            )
            return True

        except Exception as e:
            self._log.error(f"Error in strategy cycle: {e}", exc_info=True)
            return False

    def _cleanup(self) -> None:
        """Cleanup resources."""
        self._log.info("Cleaning up resources")
        try:
            close_strategy = getattr(self._strategy, "close", None)
            if callable(close_strategy):
                close_strategy()
        except Exception as e:
            self._log.error(f"Error while closing strategy resources: {e}")
        try:
            self._exchange.close()
        except Exception as e:
            self._log.error(f"Error during cleanup: {e}")

    def _sleep_with_stop(self, seconds: float) -> None:
        """Sleep in small chunks so stop() can interrupt long waits."""
        remaining = max(0.0, seconds)
        while self._running and remaining > 0:
            chunk = min(self._sleep_step_seconds, remaining)
            time.sleep(chunk)
            remaining -= chunk

    def stop(self) -> None:
        """Stop the coordinator gracefully."""
        self._log.info("Stop requested")
        self._running = False
