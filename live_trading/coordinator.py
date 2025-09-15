"""Main live trading coordinator with scheduling."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from signal_notifier import TelegramClient

from .exchange import Exchange
from .models import LiveTradingConfig, TradingSignal
from .position_manager import PositionManager
from .scanner import SymbolScanner
from .strategy import LiveTradingStrategy


class LiveTradingCoordinator:
    """Coordinates live trading execution with timeframe alignment."""

    def __init__(
        self,
        config: LiveTradingConfig,
        exchange: Exchange,
        telegram_client: Optional[TelegramClient] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._exchange = exchange
        self._log = logger or logging.getLogger(__name__)
        self._interval_seconds = config.execution_interval_minutes * 60
        self._candle_delay = timedelta(seconds=config.candle_ready_delay_seconds)

        # Initialize components
        self._position_manager = PositionManager(
            config, exchange, logger, telegram_client=telegram_client
        )
        self._scanner = SymbolScanner(exchange, logger)
        self._strategy = LiveTradingStrategy(
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
            self._execute_strategy_cycle(current_time)
            self._position_manager.update_execution_time(current_time)

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
                        self._execute_strategy_cycle(scheduled_run_time)
                        self._position_manager.update_execution_time(scheduled_run_time)

                        now_after_cycle = datetime.now(timezone.utc)
                        self._position_manager.update_positions(now_after_cycle)
                        self._position_manager.cleanup_state(now_after_cycle)

                        next_run_time = self._next_aligned_time(now_after_cycle)
                        self._log.info(
                            "Next execution at %s", next_run_time.isoformat()
                        )
                        continue

                    sleep_seconds = max(
                        1.0, (next_run_time - current_time).total_seconds()
                    )
                    time.sleep(sleep_seconds)

                except KeyboardInterrupt:
                    self._log.info("Received interrupt signal")
                    break
                except Exception as e:
                    self._log.error(f"Error in main loop: {e}", exc_info=True)
                    time.sleep(60)  # Wait a minute before retrying

        finally:
            self._running = False
            self._cleanup()

    def _next_aligned_time(self, current_time: datetime) -> datetime:
        """Align to the next interval boundary (divisible by execution_interval_minutes)."""
        # Compute next boundary in UTC to keep alignment consistent
        ts = int(current_time.replace(tzinfo=timezone.utc).timestamp())
        interval = max(self._interval_seconds, 1)
        next_boundary_ts = ((ts // interval) + 1) * interval
        aligned = datetime.fromtimestamp(next_boundary_ts, tz=timezone.utc)
        return aligned + self._candle_delay

    def _execute_strategy_cycle(self, current_time: datetime) -> None:
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
                return

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
                return

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
                return
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

        except Exception as e:
            self._log.error(f"Error in strategy cycle: {e}", exc_info=True)

    def _cleanup(self) -> None:
        """Cleanup resources."""
        self._log.info("Cleaning up resources")
        try:
            self._exchange.close()
        except Exception as e:
            self._log.error(f"Error during cleanup: {e}")

    def stop(self) -> None:
        """Stop the coordinator gracefully."""
        self._log.info("Stop requested")
        self._running = False
