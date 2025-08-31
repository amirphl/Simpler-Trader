from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .engulfing_strategy import StopLossMode


@dataclass(frozen=True)
class PinbarStrategyConfig:
    """Configuration for the Bullish Pinbar strategy."""

    symbol: str
    timeframe: str
    leverage: float
    take_profit_pct: float

    stop_loss_mode: StopLossMode = StopLossMode.PERCENT
    stop_loss_pct: float = 0.005
    exchange_fee_pct: float = 0.0004

    min_shadow_body_ratio: float = 0.5
    shadow_dominance_ratio: float = 2.0

    def __post_init__(self) -> None:
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.stop_loss_mode is StopLossMode.PERCENT and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive for percent mode")
        if self.exchange_fee_pct < 0:
            raise ValueError("exchange_fee_pct must be non-negative")
        if self.min_shadow_body_ratio <= 0:
            raise ValueError("min_shadow_body_ratio must be positive")
        if self.shadow_dominance_ratio <= 0:
            raise ValueError("shadow_dominance_ratio must be positive")


@dataclass
class Position:
    """Represents an open long position."""

    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    size: float


class PinbarStrategy(BacktestStrategy):
    """Long-only strategy that enters on bullish pinbar formations."""

    def __init__(self, config: PinbarStrategyConfig) -> None:
        self._config = config
        self._log = logging.getLogger(self.__class__.__name__)

    def name(self) -> str:
        return "PinbarStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(self, context: BacktestContext) -> Sequence[TradePerformance]:
        symbol = self._config.symbol
        timeframe = self._config.timeframe
        candles = context.data.get(symbol, {}).get(timeframe, [])
        self._log.info(
            "Running pinbar strategy",
            extra={"symbol": symbol, "timeframe": timeframe, "length": len(candles)},
        )
        if len(candles) < 2:
            return []

        trades: List[TradePerformance] = []
        position: Position | None = None
        current_capital = context.config.initial_capital

        for idx in range(1, len(candles)):
            candle = candles[idx]

            if position is not None:
                exit_price, exit_reason = self._check_exit(candles[idx], position)
                if exit_price is not None:
                    pnl = self._record_exit(candle, exit_price, exit_reason, position, trades)
                    current_capital += pnl
                    position = None

            if position is None and self._is_bullish_pinbar(candles, idx):
                entry_price = candle.open
                stop_loss = self._compute_stop_loss(entry_price, candle)
                if stop_loss >= entry_price:
                    continue
                take_profit = entry_price * (1.0 + self._config.take_profit_pct)
                position = Position(
                    entry_time=candle.open_time,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=self._config.leverage,
                    size=current_capital,
                )
                exit_price, exit_reason = self._check_exit(candle, position)
                if exit_price is not None and exit_reason is not None:
                    pnl = self._record_exit(
                        candle=candle,
                        exit_price=exit_price,
                        exit_reason=exit_reason or "",
                        position=position,
                        trades=trades,
                    )
                    current_capital += pnl
                    position = None  # type: ignore

        return trades

    def _is_bullish_pinbar(self, candles: Sequence[Candle], idx: int) -> bool:
        if idx == 0:
            return False
        candle = candles[idx]
        previous = candles[idx - 1]
        body = abs(candle.close - candle.open)
        if body <= 0:
            return False
        if not (candle.close > candle.open):
            return False
        if not (previous.open > previous.close and abs(previous.close - previous.open) > body):
            return False
        down_shadow = (candle.open - candle.low) if candle.open > candle.close else (candle.close - candle.low)
        up_shadow = (candle.high - candle.open) if candle.open > candle.close else (candle.high - candle.close)
        if down_shadow <= self._config.min_shadow_body_ratio * body:
            return False
        if up_shadow == 0:
            up_shadow = 1e-9
        return down_shadow > self._config.shadow_dominance_ratio * up_shadow

    def _check_exit(self, candle: Candle, position: Position) -> tuple[float | None, str | None]:
        if candle.low <= position.stop_loss:
            return position.stop_loss, "Stop Loss"
        if candle.high >= position.take_profit:
            return position.take_profit, "Take Profit"
        return None, None

    def _record_exit(
        self,
        candle: Candle,
        exit_price: float,
        exit_reason: str,
        position: Position,
        trades: List[TradePerformance],
    ) -> float:
        pnl, fees_paid, return_pct = self._compute_trade_financials(exit_price, position)
        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=candle.open_time,
                pnl=pnl,
                return_pct=return_pct,
                notes=f"{exit_reason} at {exit_price:.2f}",
                metadata={
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "leverage": position.leverage,
                    "fees": fees_paid,
                },
            )
        )
        return pnl

    def _compute_trade_financials(self, exit_price: float, position: Position) -> tuple[float, float, float]:
        price_change = exit_price - position.entry_price
        gross_pnl = (price_change / position.entry_price) * position.size * position.leverage
        fees_paid = self._calculate_fees(exit_price, position)
        pnl = gross_pnl - fees_paid
        return_pct = (pnl / position.size) * 100.0 if position.size else 0.0
        return pnl, fees_paid, return_pct

    def _calculate_fees(self, exit_price: float, position: Position) -> float:
        fee_pct = self._config.exchange_fee_pct
        if fee_pct <= 0:
            return 0.0
        notional_entry = position.size * position.leverage
        if notional_entry <= 0 or position.entry_price <= 0:
            return 0.0
        quantity = notional_entry / position.entry_price
        entry_fee = notional_entry * fee_pct
        exit_fee = quantity * exit_price * fee_pct
        return entry_fee + exit_fee

    def _compute_stop_loss(self, entry_price: float, candle: Candle) -> float:
        mode = self._config.stop_loss_mode
        if mode is StopLossMode.PERCENT:
            return entry_price * (1.0 - self._config.stop_loss_pct)
        if mode is StopLossMode.CLOSE:
            return candle.close
        if mode is StopLossMode.LOW:
            return candle.low
        if mode is StopLossMode.OPEN:
            return candle.open
        if mode is StopLossMode.BODY:
            body = candle.close - candle.open
            if body <= 0:
                return candle.open
            return candle.close - (self._config.stop_loss_pct * body)
        raise ValueError(f"Unsupported stop-loss mode: {mode}")

