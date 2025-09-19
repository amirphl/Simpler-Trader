"""Pin Bar Magic v2 live trading strategy implementation."""

from __future__ import annotations

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


class PinBarMagicLiveStrategy(BaseLiveTradingStrategy):
    """Live Pin Bar Magic strategy aligned with backtest pinbar_magic_strategy_v2."""

    def generate_signals_for_symbols(
        self,
        symbols: List[SymbolInfo],
        current_time: datetime,
    ) -> Tuple[List[TradingSignal], Dict[str, PinBarMagicSnapshot]]:
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

    def _build_snapshot(
        self,
        symbol: str,
        current_time: datetime,
    ) -> Optional[PinBarMagicSnapshot]:
        cfg = self._config
        min_history = max(
            cfg.slow_sma_period,
            cfg.medium_ema_period,
            cfg.fast_ema_period,
            cfg.atr_period,
        ) + 2
        raw = self._fetch_binance_klines(
            symbol=symbol,
            interval=cfg.timeframe,
            limit=max(min_history + 1, 256),
        )
        if len(raw) < min_history:
            self._log.debug(
                "PinBarMagic: insufficient klines for %s (%s < %s)",
                symbol,
                len(raw),
                min_history,
            )
            return None

        candles = [Candle.from_binance(symbol, cfg.timeframe, row) for row in raw]
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

    def _build_entry_signal(
        self,
        *,
        snapshot: PinBarMagicSnapshot,
        side: PositionSide,
    ) -> Optional[TradingSignal]:
        cfg = self._config
        prev = snapshot.previous_bar
        if side == PositionSide.LONG:
            entry_price = prev.high
            stop_for_risk = prev.low - snapshot.atr_prev * cfg.atr_multiple
            if entry_price <= stop_for_risk:
                return None
        else:
            entry_price = prev.low
            stop_for_risk = prev.high + snapshot.atr_prev * cfg.atr_multiple
            if stop_for_risk <= entry_price:
                return None

        return TradingSignal(
            timestamp=snapshot.bar.close_time,
            symbol=snapshot.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_for_risk,
            take_profit=None,
            leverage=cfg.leverage,
            margin_mode=cfg.margin_mode,
            reason=f"PinBarMagic v2 {'LONG' if side == PositionSide.LONG else 'SHORT'}",
            strategy="pinbar_magic_v2",
            metadata={
                "strategy": "pinbar_magic_v2",
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

    def _is_bullish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        return (
            (candle.close > candle.open and (candle.open - candle.low) > 0.66 * rng)
            or (candle.close < candle.open and (candle.close - candle.low) > 0.66 * rng)
        )

    def _is_bearish_pinbar(self, candle: Candle) -> bool:
        rng = candle.high - candle.low
        if rng <= 0:
            return False
        return (
            (candle.close > candle.open and (candle.high - candle.close) > 0.66 * rng)
            or (candle.close < candle.open and (candle.high - candle.open) > 0.66 * rng)
        )

    def _bull_pierce(self, candle: Candle, fast: float, med: float, slow: float) -> bool:
        return any(
            candle.low < ma and candle.open > ma and candle.close > ma
            for ma in (fast, med, slow)
        )

    def _bear_pierce(self, candle: Candle, fast: float, med: float, slow: float) -> bool:
        return any(
            candle.high > ma and candle.open < ma and candle.close < ma
            for ma in (fast, med, slow)
        )
