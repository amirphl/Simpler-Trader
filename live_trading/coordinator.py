"""Main live trading coordinator with scheduling."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from signal_notifier import TelegramClient

from .exchange import Exchange
from .models import LiveTradingConfig
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
        self._timeframe_delta = timedelta(
            minutes=self._interval_to_minutes(config.timeframe)
        )
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
            self._log.info(f"Timeframe: {self._config.timeframe}")
            next_run_time = self._compute_next_run_time(datetime.now(timezone.utc))
            self._log.info(
                f"Waiting for candle close with {self._config.candle_ready_delay_seconds}s delay; "
                f"next execution at {next_run_time.isoformat()}"
            )

            while self._running:
                try:
                    current_time = datetime.now(timezone.utc)

                    # Catch up any missed intervals; do not skip runs.
                    while self._running and current_time >= next_run_time:
                        scheduled_run_time = next_run_time
                        self._log.info(
                            f"Executing strategy at {current_time.isoformat()} "
                            f"(scheduled for {scheduled_run_time.isoformat()})"
                        )

                        # Refresh positions before executing in case prior cycle closed trades.
                        self._position_manager.update_positions(current_time)

                        self._execute_strategy_cycle(scheduled_run_time)
                        self._position_manager.update_execution_time(scheduled_run_time)

                        # Update existing positions and cleanup right after execution
                        now_after_cycle = datetime.now(timezone.utc)
                        self._position_manager.update_positions(now_after_cycle)
                        self._position_manager.cleanup_state(now_after_cycle)

                        # Schedule next run
                        next_run_time = self._compute_next_run_time(now_after_cycle)
                        self._log.info(f"Next execution at {next_run_time.isoformat()}")
                        current_time = datetime.now(timezone.utc)
                        continue

                    # Sleep until next scheduled run
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

    def _compute_next_run_time(self, current_time: datetime) -> datetime:
        """Compute the next execution time based on timeframe and last run."""
        latest_ready_time = self._latest_ready_time(current_time)
        last_execution = self._position_manager.get_state().last_execution_time
        self._log.debug(
            f"latest_ready_time={latest_ready_time.isoformat()}, "
            f"last_execution={last_execution.isoformat() if last_execution else None}"
        )

        if last_execution is None or last_execution < latest_ready_time:
            return latest_ready_time

        return self._next_ready_time(current_time)

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
        }
        return mapping.get(interval, 60)

    def _latest_ready_time(self, current_time: datetime) -> datetime:
        """Return the most recent timeframe boundary plus readiness delay."""
        start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = current_time - start_of_day
        periods_since_midnight = int(
            elapsed.total_seconds() // self._timeframe_delta.total_seconds()
        )
        latest_boundary = start_of_day + self._timeframe_delta * periods_since_midnight
        self._log.debug(
            f"start_of_day={start_of_day.isoformat()}, elapsed={elapsed}, "
            f"periods_since_midnight={periods_since_midnight}, latest_boundary={latest_boundary.isoformat()}, "
            f"candle_delay={self._candle_delay}"
        )
        return latest_boundary + self._candle_delay

    def _next_ready_time(self, current_time: datetime) -> datetime:
        """Return the next timeframe boundary plus readiness delay."""
        latest_ready = self._latest_ready_time(current_time)
        if current_time < latest_ready:
            return latest_ready
        return latest_ready + self._timeframe_delta

    def _execute_strategy_cycle(self, current_time: datetime) -> None:
        """Execute one complete strategy cycle."""
        try:
            # 1. Scan for top symbols by volume
            self._log.info(
                f"Scanning top {self._config.top_m_symbols} symbols by volume for timeframe {self._config.timeframe}"
            )
            top_symbols = self._scanner.scan_top_symbols_by_volume(
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
            long_signals = long_signals[:10]
            short_signals = short_signals[:10]

            if not long_signals and not short_signals:
                self._log.info("No trading signals generated")
                return
            signals = [*long_signals, *short_signals]

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
