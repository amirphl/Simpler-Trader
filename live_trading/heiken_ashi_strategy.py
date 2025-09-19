"""Heiken Ashi live trading strategy implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from candle_downloader.models import Candle

from .exchange import PositionSide
from .heiken_ashi import (
    calculate_heiken_ashi,
    detect_reversal_signal,
    detect_reversal_signal_v2,
)
from .models import (
    SymbolInfo,
    TradingSignal,
)
from .strategy_shared import BaseLiveTradingStrategy


class HeikenAshiLiveStrategy(BaseLiveTradingStrategy):
    """Live trading strategy based on Heiken Ashi reversals."""

    def analyze_symbol(
        self,
        symbol_info: SymbolInfo,
        current_time: datetime,
    ) -> Optional[TradingSignal]:
        """Analyze a symbol and generate trading signal if conditions are met.

        Args:
            symbol_info: Information about the symbol
            current_time: Current timestamp

        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        symbol = symbol_info.symbol

        # Check if symbol is disabled
        if self._state.is_symbol_disabled(
            symbol, current_time, self._config.disable_symbol_hours
        ):
            self._log.debug(f"Symbol {symbol} is disabled")
            return None

        # Check if position already exists
        if symbol in self._state.active_positions:
            self._log.debug(f"Position already exists for {symbol}")
            return None

        try:
            # Fetch recent candles for Heiken Ashi calculation
            # We need at least W+1 candles for reversal detection
            required_candles = self._config.heiken_ashi_candles_before + 1
            # Pull candles from Binance directly to ensure consistent format/data
            # Fetch atleast 20 candles to enhance Heiken Ashi calculation accuracy
            klines = self._fetch_binance_klines(
                symbol=symbol,
                interval=self._config.timeframe,
                limit=max(required_candles * 2, 20) + 1,  # extra for dropping last
            )

            if len(klines) < required_candles:
                self._log.debug(
                    f"Insufficient candles for {symbol}: {len(klines)} < {required_candles}"
                )
                return None

            # Convert to Candle objects
            candles = [
                Candle.from_binance(symbol, self._config.timeframe, kline)
                for kline in klines
            ]

            inprogress_candle = candles[-1]
            candles_excluding_last_one = candles[
                :-1
            ]  # Drop last candle to avoid in-progress data

            # Calculate Heiken Ashi candles
            # ha_candles = calculate_heiken_ashi(candles)
            ha_candles_excluding_last_one = calculate_heiken_ashi(
                candles_excluding_last_one
            )

            last_closed_candle = candles_excluding_last_one[-1]
            last_closed_ha_candle = ha_candles_excluding_last_one[-1]
            self._log.debug(
                "%s latest original closed candle (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if last_closed_candle.is_bullish() else "Red",
                last_closed_candle.open,
                last_closed_candle.high,
                last_closed_candle.low,
                last_closed_candle.close,
            )
            self._log.debug(
                "%s latest HA closed candle (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if last_closed_ha_candle.is_bullish() else "Red",
                last_closed_ha_candle.ha_open,
                last_closed_ha_candle.ha_high,
                last_closed_ha_candle.ha_low,
                last_closed_ha_candle.ha_close,
            )
            self._log.debug(
                "%s HA candle[-2] (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if ha_candles_excluding_last_one[-2].is_bullish() else "Red",
                ha_candles_excluding_last_one[-2].ha_open,
                ha_candles_excluding_last_one[-2].ha_high,
                ha_candles_excluding_last_one[-2].ha_low,
                ha_candles_excluding_last_one[-2].ha_close,
            )
            self._log.debug(
                "%s HA candle[-3] (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if ha_candles_excluding_last_one[-3].is_bullish() else "Red",
                ha_candles_excluding_last_one[-3].ha_open,
                ha_candles_excluding_last_one[-3].ha_high,
                ha_candles_excluding_last_one[-3].ha_low,
                ha_candles_excluding_last_one[-3].ha_close,
            )

            if len(ha_candles_excluding_last_one) < required_candles:
                return None

            # TODO:
            # Detect reversal signal on closed candles only
            signal_type = detect_reversal_signal(
                ha_candles_excluding_last_one,
                lookback_candles=self._config.heiken_ashi_candles_before,
            )
            # signal_type = detect_reversal_signal_v2(
            #     ha_candles,
            #     lookback_candles=self._config.heiken_ashi_candles_before,
            # )

            if signal_type is None:
                return None

            # Determine position side based on price movement direction
            if signal_type == "LONG":
                # Price was falling (top losers), now reversing up
                # TODO:
                # if symbol_info.price_change_pct >= 0:
                #     return None  # Symbol should be from losers list
                side = PositionSide.LONG
            else:  # SHORT
                # Price was rising (top gainers), now reversing down
                # TODO
                # if symbol_info.price_change_pct <= 0:
                #     return None  # Symbol should be from gainers list
                side = PositionSide.SHORT

            # Entry price is current market price (last in-progress candle close)
            entry_price = inprogress_candle.close

            # Stop loss is the open price of HA candle
            # TODO:
            # stop_loss = last_closed_ha_candle.ha_open

            stop_loss = None

            # Calculate take profit
            if side == PositionSide.LONG:
                take_profit = entry_price * (1 + self._config.take_profit_pct / 100)
            else:
                take_profit = entry_price * (1 - self._config.take_profit_pct / 100)

            # Validate signal
            if side == PositionSide.LONG:
                if stop_loss is not None and stop_loss >= entry_price:
                    fallback_sl = entry_price * 0.99
                    self._log.warning(
                        "Invalid LONG stop loss for %s: stop_loss (%.6f) >= entry (%.6f); "
                        "using fallback 1%% below entry: %.6f",
                        symbol,
                        stop_loss,
                        entry_price,
                        fallback_sl,
                    )
                    stop_loss = fallback_sl
            else:
                if stop_loss is not None and stop_loss <= entry_price:
                    fallback_sl = entry_price * 1.01
                    self._log.warning(
                        "Invalid SHORT stop loss for %s: stop_loss (%.6f) <= entry (%.6f); "
                        "using fallback 1%% above entry: %.6f",
                        symbol,
                        stop_loss,
                        entry_price,
                        fallback_sl,
                    )
                    stop_loss = fallback_sl

            # Create signal
            signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=self._config.leverage,
                margin_mode=self._config.margin_mode,
                reason=f"HA reversal: {signal_type} after {self._config.heiken_ashi_candles_before} candles",
                metadata={
                    "price_change_pct": symbol_info.price_change_pct,
                    "last_ha_candle": {
                        "open": last_closed_ha_candle.ha_open,
                        "high": last_closed_ha_candle.ha_high,
                        "low": last_closed_ha_candle.ha_low,
                        "close": last_closed_ha_candle.ha_close,
                    },
                },
            )

            sl = f"{stop_loss:.6f}" if stop_loss is not None else "None"
            tp = f"{take_profit:.6f}" if take_profit is not None else "None"
            self._log.info(
                f"Signal generated: {symbol} {side.value} @ {entry_price:.6f}, "
                f"SL: {sl}, TP: {tp}"
            )

            return signal

        except Exception as e:
            self._log.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return None

    def analyze_symbol_v2(
        self,
        symbol_info: SymbolInfo,
        current_time: datetime,
    ) -> Optional[TradingSignal]:
        """Analyze a symbol using reconstructed candles from 1m klines."""
        symbol = symbol_info.symbol

        # Check if symbol is disabled
        if self._state.is_symbol_disabled(
            symbol, current_time, self._config.disable_symbol_hours
        ):
            self._log.debug(f"Symbol {symbol} is disabled")
            return None

        # Check if position already exists
        if symbol in self._state.active_positions:
            self._log.debug(f"Position already exists for {symbol}")
            return None

        timeframe_minutes = self._interval_to_minutes(self._config.timeframe)
        if timeframe_minutes <= 0:
            self._log.error(
                "Invalid timeframe minutes for %s: %s", symbol, self._config.timeframe
            )
            return None

        required_candles = max(self._config.heiken_ashi_candles_before + 1, 3)
        lookback_periods = max(
            required_candles + 1, 20 + 1
        )  # fetch at least 21 periods

        end_boundary = current_time.replace(
            second=0, microsecond=0, tzinfo=timezone.utc
        )
        total_minutes = timeframe_minutes * lookback_periods
        start_boundary = end_boundary - timedelta(minutes=total_minutes)

        try:
            minute_candles = self._get_1m_candles(symbol, start_boundary, end_boundary)
            if len(minute_candles) == 0:
                self._log.warning("No 1m klines fetched for %s", symbol)
                return None

            aggregated = self._aggregate_candles(
                minute_candles,
                start_boundary=start_boundary,
                end_boundary=end_boundary,
                timeframe_minutes=timeframe_minutes,
            )

            if len(aggregated) < required_candles:
                self._log.debug(
                    "Insufficient aggregated candles for %s: got %s need %s",
                    symbol,
                    len(aggregated),
                    required_candles,
                )
                return None

            # Drop the most recent candle to avoid any partial period
            # candles = aggregated[:-1]
            # if len(candles) < required_candles:
            #     return None
            candles = aggregated

            ha_candles = calculate_heiken_ashi(candles)
            inprogress_candle = candles[-1]

            # Log last running candle
            self._log.info(
                "%s latest running candle: open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                inprogress_candle.open,
                inprogress_candle.high,
                inprogress_candle.low,
                inprogress_candle.close,
            )

            signal_type = detect_reversal_signal_v2(
                ha_candles,
                lookback_candles=self._config.heiken_ashi_candles_before,
            )

            if not (
                (inprogress_candle.is_bullish() and signal_type == "LONG")
                or (inprogress_candle.is_bearish() and signal_type == "SHORT")
            ):
                return None

            side = PositionSide.LONG if signal_type == "LONG" else PositionSide.SHORT
            entry_price = inprogress_candle.close

            # TODO:
            # stop_loss = last_running_ha_candle.ha_open

            # TODO:
            # if side == PositionSide.LONG and stop_loss >= entry_price:
            #     stop_loss = entry_price * 0.99
            # elif side == PositionSide.SHORT and stop_loss <= entry_price:
            #     stop_loss = entry_price * 0.99

            # TODO:
            stop_loss = None

            if side == PositionSide.LONG:
                take_profit = entry_price * (1 + self._config.take_profit_pct / 100)
            else:
                take_profit = entry_price * (1 - self._config.take_profit_pct / 100)

            signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=self._config.leverage,
                margin_mode=self._config.margin_mode,
                reason=f"HA reversal v2: {signal_type} after {self._config.heiken_ashi_candles_before} candles",
                metadata={
                    "timeframe": self._config.timeframe,
                    "source": "1m_reconstructed",
                    "price_change_pct": symbol_info.price_change_pct,
                },
            )

            sl = f"{signal.stop_loss:.6f}" if signal.stop_loss is not None else "None"
            tp = (
                f"{signal.take_profit:.6f}"
                if signal.take_profit is not None
                else "None"
            )
            self._log.info(
                "V2 Signal generated: %s %s @ %.6f, SL: %s, TP: %s, Timestamp: %s",
                symbol,
                side.value,
                entry_price,
                sl,
                tp,
                signal.timestamp.isoformat(),
            )

            return signal
        except Exception as exc:
            self._log.error(
                "Error in analyze_symbol_v2 for %s: %s", symbol, exc, exc_info=True
            )
            return None

    def generate_signals(
        self,
        top_gainers: List[SymbolInfo],
        top_losers: List[SymbolInfo],
        current_time: datetime,
    ) -> Tuple[List[TradingSignal], List[TradingSignal]]:
        """Generate trading signals for top movers.

        Args:
            top_gainers: Symbols with highest positive price change
            top_losers: Symbols with highest negative price change
            current_time: Current timestamp

        Returns:
            Tuple of (long_signals, short_signals)
        """
        long_signals: List[TradingSignal] = []
        short_signals: List[TradingSignal] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._log.info(f"Analyzing {len(top_losers)} top losers for LONG signals")
        self._log.info(f"Analyzing {len(top_gainers)} top gainers for SHORT signals")

        def process(
            symbol_info: SymbolInfo, expect_side: PositionSide
        ) -> Optional[TradingSignal]:
            signal = self.analyze_symbol_v2(symbol_info, current_time)
            if signal is None:
                return None
            # TODO:
            # if signal.side != expect_side:
            #     return None
            return signal

        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for si in top_losers:
                tasks.append(executor.submit(process, si, PositionSide.LONG))
            for si in top_gainers:
                tasks.append(executor.submit(process, si, PositionSide.SHORT))

            for future in as_completed(tasks):
                sig = future.result()
                if sig is None:
                    continue
                if sig.side == PositionSide.LONG:
                    long_signals.append(sig)
                else:
                    short_signals.append(sig)

        self._log.info(
            "Generated %s trading signals (%s long, %s short)",
            len(long_signals) + len(short_signals),
            len(long_signals),
            len(short_signals),
        )
        for signal in long_signals + short_signals:
            sl = f"{signal.stop_loss:.6f}" if signal.stop_loss is not None else "None"
            tp = (
                f"{signal.take_profit:.6f}"
                if signal.take_profit is not None
                else "None"
            )
            self._log.info(
                "  Signal: %s %s @ %.6f (SL: %s, TP: %s) Reason: %s",
                signal.symbol,
                signal.side.value,
                signal.entry_price,
                sl,
                tp,
                signal.reason,
            )
        return long_signals, short_signals


LiveTradingStrategy = HeikenAshiLiveStrategy
