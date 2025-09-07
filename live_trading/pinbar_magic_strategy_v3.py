"""Pin Bar Magic v3 live trading strategy implementation."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backtest.scalping_FVG_strategy import atr as calc_atr
from backtest.scalping_FVG_strategy import ema as calc_ema
from backtest.scalping_FVG_strategy import sma as calc_sma
from candle_downloader.models import Candle

from .exchange import PositionSide
from .models import PinBarMagicSnapshot, SymbolInfo, TradingSignal
from .strategy_shared import BaseLiveTradingStrategy

_KLINES_RETRIES = 3
_KLINES_RETRY_DELAY_SECONDS = 1.0


class PinBarMagicLiveStrategy(BaseLiveTradingStrategy):
    """Live Pin Bar Magic strategy aligned with backtest pinbar_magic_strategy_v3."""

    def generate_signals_for_symbols(
        self,
        symbols: List[SymbolInfo],
        current_time: datetime,
    ) -> Tuple[List[TradingSignal], Dict[str, PinBarMagicSnapshot]]:
        if not self._has_sufficient_balance():
            self._log.warning(
                "PinBarMagic: skipping signal generation due to insufficient balance"
            )
            return [], {}

        snapshots: Dict[str, PinBarMagicSnapshot] = {}
        signals: List[TradingSignal] = []

        def process(symbol: str) -> Optional[PinBarMagicSnapshot]:
            return self._build_snapshot(symbol, current_time)

        max_workers = min(8, max(1, len(symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process, symbol_info.symbol): symbol_info.symbol
                for symbol_info in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    snapshot = future.result()
                except Exception as exc:
                    self._log.warning(
                        "PinBarMagic: snapshot build failed for %s: %s", symbol, exc
                    )
                    continue
                if snapshot is None:
                    continue
                snapshots[symbol] = snapshot

                if snapshot.long_entry:
                    signal = self._build_entry_signal(
                        snapshot=snapshot, side=PositionSide.LONG
                    )
                    if signal is not None:
                        signals.append(signal)
                if snapshot.short_entry:
                    signal = self._build_entry_signal(
                        snapshot=snapshot, side=PositionSide.SHORT
                    )
                    if signal is not None:
                        signals.append(signal)

        self._log.info(
            "PinBarMagic: evaluated %s symbols, generated %s signals",
            len(symbols),
            len(signals),
        )
        return signals, snapshots

    def _has_sufficient_balance(self) -> bool:
        try:
            balance = float(self._exchange.get_account_balance())
        except Exception as exc:
            if self._is_insufficient_balance_error(exc):
                self._log.warning(
                    "PinBarMagic: get_account_balance indicates insufficient balance: %s",
                    exc,
                )
                return False
            # Non-balance retrieval errors should not block signal generation.
            return True
        return balance > 0

    @staticmethod
    def _is_insufficient_balance_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "insufficient balance",
            "insufficient margin",
            "insufficient available",
            "not enough balance",
            "not enough margin",
            "balance not enough",
            "margin not enough",
            "available balance",
        )
        return any(marker in text for marker in markers)

    def _build_snapshot(
        self,
        symbol: str,
        current_time: datetime,
    ) -> Optional[PinBarMagicSnapshot]:
        cfg = self._config
        min_history = (
            max(
                cfg.slow_sma_period,
                cfg.medium_ema_period,
                cfg.fast_ema_period,
                cfg.atr_period,
            )
            + 2
        )
        # Returns None on total failure, empty list when exchange returned no rows.
        raw = self._fetch_strategy_klines(
            symbol=symbol,
            interval=cfg.timeframe,
            limit=max(min_history + 1, 256),
        )
        if raw is None:
            # Hard fetch failure already logged inside _fetch_strategy_klines.
            return None
        if len(raw) < min_history:
            self._log.debug(
                "PinBarMagic: insufficient klines for %s (%s < %s)",
                symbol,
                len(raw),
                min_history,
            )
            return None

        candles = [Candle.from_binance(symbol, cfg.timeframe, row) for row in raw]
        # Drop the still-open candle; all signal logic runs on closed bars.
        closed = candles[:-1]
        if len(closed) < min_history:
            return None

        closes = [c.close for c in closed]
        fast_ema = calc_ema(closes, cfg.fast_ema_period)
        med_ema = calc_ema(closes, cfg.medium_ema_period)
        slow_sma = calc_sma(closes, cfg.slow_sma_period)
        atr_values = calc_atr(closed, cfg.atr_period)
        idx = len(closed) - 1
        if idx - 1 < 0:
            return None

        fast = fast_ema[idx]
        med = med_ema[idx]
        slow = slow_sma[idx]
        fast_prev = fast_ema[idx - 1]
        med_prev = med_ema[idx - 1]
        atr_prev = atr_values[idx - 1]
        if (
            fast is None
            or med is None
            or slow is None
            or fast_prev is None
            or med_prev is None
            or atr_prev is None
        ):
            return None

        candle = closed[idx]
        prev = closed[idx - 1]
        fast_f = float(fast)
        med_f = float(med)
        slow_f = float(slow)
        fast_prev_f = float(fast_prev)
        med_prev_f = float(med_prev)
        atr_prev_f = float(atr_prev)

        bull_pin = self._is_bullish_pinbar(candle)
        bear_pin = self._is_bearish_pinbar(candle)
        fan_up = fast_f > med_f > slow_f
        fan_down = fast_f < med_f < slow_f
        bull_pierce = self._bull_pierce(candle, fast_f, med_f, slow_f)
        bear_pierce = self._bear_pierce(candle, fast_f, med_f, slow_f)
        long_entry = fan_up and bull_pin and bull_pierce
        short_entry = fan_down and bear_pin and bear_pierce
        crossunder = fast_prev_f > med_prev_f and fast_f < med_f
        crossover = fast_prev_f < med_prev_f and fast_f > med_f
        friday_close = (
            cfg.enable_friday_close
            and candle.close_time.weekday() == 4
            and candle.close_time.hour == cfg.friday_close_hour_utc
        )

        return PinBarMagicSnapshot(
            symbol=symbol,
            timeframe=cfg.timeframe,
            timeframe_minutes=self._interval_to_minutes(cfg.timeframe),
            bar=candle,
            previous_bar=prev,
            fast=fast_f,
            med=med_f,
            slow=slow_f,
            atr_prev=atr_prev_f,
            fast_prev=fast_prev_f,
            med_prev=med_prev_f,
            long_entry=long_entry,
            short_entry=short_entry,
            crossunder=crossunder,
            crossover=crossover,
            friday_close=friday_close,
        )

    def _fetch_strategy_klines(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[List[List]]:
        """Fetch klines with retry logic and a legacy fallback.

        Returns:
            A non-empty list of raw kline rows on success.
            None if every attempt (including the legacy fallback) failed – the
            caller should treat None as a hard failure and skip the symbol.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, _KLINES_RETRIES + 1):
            try:
                rows = self._exchange.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                )
                if rows:
                    return rows
                self._log.warning(
                    "PinBarMagic: exchange.get_klines returned no data for %s (%s) "
                    "(attempt %d/%d)",
                    symbol,
                    interval,
                    attempt,
                    _KLINES_RETRIES,
                )
            except Exception as exc:
                last_exc = exc
                self._log.warning(
                    "PinBarMagic: exchange.get_klines raised for %s (%s) "
                    "(attempt %d/%d): %s",
                    symbol,
                    interval,
                    attempt,
                    _KLINES_RETRIES,
                    exc,
                )
            if attempt < _KLINES_RETRIES:
                time.sleep(_KLINES_RETRY_DELAY_SECONDS)

        # Compatibility fallback to legacy direct-fetch path.
        self._log.info(
            "PinBarMagic: falling back to legacy klines fetch for %s (%s)",
            symbol,
            interval,
        )
        try:
            rows = self._fetch_binance_klines(
                symbol=symbol, interval=interval, limit=limit
            )
            if rows:
                return rows
        except Exception as exc:
            self._log.warning(
                "PinBarMagic: legacy klines fallback also failed for %s (%s): %s",
                symbol,
                interval,
                exc,
            )

        self._log.error(
            "PinBarMagic: all klines fetch attempts failed for %s (%s). "
            "Last primary error: %s",
            symbol,
            interval,
            last_exc,
        )
        return None

    def _build_entry_signal(
        self,
        *,
        snapshot: PinBarMagicSnapshot,
        side: PositionSide,
    ) -> Optional[TradingSignal]:
        """Build an entry signal matching the Pine enterlong / entershort functions.

        Pine reference:
            entryPrice = high[1]  /  low[1]          (stop-entry trigger)
            stopLoss   = low[1]  - atr[1] * atr_mult
                       / high[1] + atr[1] * atr_mult  (initial hard stop)

        stop_loss here is the INITIAL hard risk stop only.  The coordinator
        manages trailing separately in _apply_bar_trailing / _manage_tick_trailing
        and pushes updated levels to the exchange via _update_stop_loss_on_exchange.
        Overwriting stop_loss with a trailing value here would corrupt that logic.

        take_profit is intentionally None: the Pine strategy exits exclusively via
        trailing stop (strategy.exit trail_points/trail_offset).  Placing a fixed TP
        order would conflict with the trailing mechanism.
        """
        cfg = self._config
        prev = snapshot.previous_bar

        if side == PositionSide.LONG:
            entry_price = float(prev.high)
            stop_for_risk = float(prev.low) - snapshot.atr_prev * cfg.atr_multiple
            if stop_for_risk >= entry_price:
                self._log.debug(
                    "PinBarMagic: LONG signal rejected for %s – "
                    "stop (%.6f) >= entry (%.6f)",
                    snapshot.symbol,
                    stop_for_risk,
                    entry_price,
                )
                return None
        else:
            entry_price = float(prev.low)
            stop_for_risk = float(prev.high) + snapshot.atr_prev * cfg.atr_multiple
            if stop_for_risk <= entry_price:
                self._log.debug(
                    "PinBarMagic: SHORT signal rejected for %s – "
                    "stop (%.6f) <= entry (%.6f)",
                    snapshot.symbol,
                    stop_for_risk,
                    entry_price,
                )
                return None

        return TradingSignal(
            timestamp=snapshot.bar.close_time,
            symbol=snapshot.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_for_risk,  # initial hard stop only – NOT the trailing level
            take_profit=None,  # exits via trailing stop, not a fixed TP
            leverage=cfg.leverage,
            margin_mode=cfg.margin_mode,
            reason=f"PinBarMagic v3 {'LONG' if side == PositionSide.LONG else 'SHORT'}",
            strategy="pinbar_magic_v3",
            metadata={
                "strategy": "pinbar_magic_v3",
                "timeframe": cfg.timeframe,
                "signal_bar_close_time": snapshot.bar.close_time.isoformat(),
                "signal_bar_open_time": snapshot.bar.open_time.isoformat(),
                "entry_activation_mode": cfg.entry_activation_mode,
                "entry_cancel_bars": cfg.entry_cancel_bars,
                "trail_points": cfg.trail_points,
                "trail_offset": cfg.trail_offset,
                "risk_equity_include_unrealized": cfg.risk_equity_include_unrealized,
                "risk_equity_mark_source": cfg.risk_equity_mark_source,
                "atr_prev": snapshot.atr_prev,
                "atr_multiple": cfg.atr_multiple,
            },
        )

    # ------------------------------------------------------------------
    # Pin-bar geometry helpers – aligned with Pine Script definitions
    # ------------------------------------------------------------------

    def _is_bullish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        return (
            candle.close > candle.open and (candle.open - candle.low) > 0.66 * rng
        ) or (candle.close < candle.open and (candle.close - candle.low) > 0.66 * rng)

    def _is_bearish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        return (
            candle.close > candle.open and (candle.high - candle.close) > 0.66 * rng
        ) or (candle.close < candle.open and (candle.high - candle.open) > 0.66 * rng)

    def _bull_pierce(
        self, candle: Candle, fast: float, med: float, slow: float
    ) -> bool:
        return any(
            candle.low < ma and candle.open > ma and candle.close > ma
            for ma in (fast, med, slow)
        )

    def _bear_pierce(
        self, candle: Candle, fast: float, med: float, slow: float
    ) -> bool:
        return any(
            candle.high > ma and candle.open < ma and candle.close < ma
            for ma in (fast, med, slow)
        )
