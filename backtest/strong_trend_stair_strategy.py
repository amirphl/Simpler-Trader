from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .indicators import ema as calc_ema


@dataclass(frozen=True)
class StrongTrendStairStrategyConfig:
    symbol: str
    timeframe: str

    leverage: float = 100.0
    position_balance_pct: float = 2.0
    starting_balance_usd: float | None = None
    hard_stop_loss_pct: float = 5.0
    trail_start_pct: float = 2.0
    trail_offset_pct: float = 1.0

    ema_fast_len: int = 50
    ema_mid_len: int = 100
    ema_slow_len: int = 200
    slope_lookback: int = 10

    st_atr_len: int = 10
    st_factor: float = 3.0

    di_len: int = 14
    adx_smooth: int = 14
    adx_min: float = 20.0
    reverse_on_opposite_signal: bool = False

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        timeframe = self.timeframe.strip()
        if not symbol:
            raise ValueError("symbol must not be empty")
        if not timeframe:
            raise ValueError("timeframe must not be empty")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.position_balance_pct <= 0 or self.position_balance_pct > 100:
            raise ValueError("position_balance_pct must be in (0, 100]")
        if self.starting_balance_usd is not None and self.starting_balance_usd <= 0:
            raise ValueError("starting_balance_usd must be positive when provided")
        if self.hard_stop_loss_pct <= 0:
            raise ValueError("hard_stop_loss_pct must be positive")
        if self.trail_start_pct <= 0:
            raise ValueError("trail_start_pct must be positive")
        if self.trail_offset_pct <= 0:
            raise ValueError("trail_offset_pct must be positive")
        if (
            min(
                self.ema_fast_len,
                self.ema_mid_len,
                self.ema_slow_len,
                self.slope_lookback,
                self.st_atr_len,
                self.di_len,
                self.adx_smooth,
            )
            <= 0
        ):
            raise ValueError("period lengths must be positive")
        if self.ema_fast_len >= self.ema_mid_len or self.ema_mid_len >= self.ema_slow_len:
            raise ValueError("EMA lengths must satisfy fast < mid < slow")
        if self.st_factor <= 0:
            raise ValueError("st_factor must be positive")
        if self.adx_min < 0:
            raise ValueError("adx_min must be non-negative")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)


@dataclass
class _PositionState:
    direction: str  # "long" | "short"
    entry_time: datetime
    entry_index: int
    entry_balance_usd: float
    entry_price: float
    qty: float
    initial_stop_price: float
    stop_price: float
    trailing_active: bool = False


@dataclass(frozen=True)
class _SignalSnapshot:
    direction: str  # "long" | "short"
    candle_index: int
    close_price: float
    close_time: datetime


@dataclass(frozen=True)
class _IndicatorSnapshot:
    fast: float
    mid: float
    slow: float
    supertrend: float
    plus_di: float
    minus_di: float
    adx: float
    slow_prev: float


class StrongTrendStairStrategy(BacktestStrategy):
    _LOW_PROFIT_THRESHOLD_PCT = 1.0
    _MID_PROFIT_THRESHOLD_PCT = 2.0
    _MID_TRAIL_OFFSET_PCT = 0.5
    _HIGH_TRAIL_OFFSET_PCT = 1.0

    def __init__(self, config: StrongTrendStairStrategyConfig) -> None:
        self._config = config
        self._starting_balance_usd: float | None = None

    def name(self) -> str:
        return "StrongTrendStairStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        cfg = self._config
        candles = context.data.get(cfg.symbol, {}).get(cfg.timeframe, [])
        if not candles:
            return [], {"note": "no_data"}

        closes = [c.close for c in candles]
        ema_fast = calc_ema(closes, cfg.ema_fast_len)
        ema_mid = calc_ema(closes, cfg.ema_mid_len)
        ema_slow = calc_ema(closes, cfg.ema_slow_len)
        st_line = self._supertrend_line(candles, cfg.st_atr_len, cfg.st_factor)
        di_plus, di_minus, adx = self._dmi(candles, cfg.di_len, cfg.adx_smooth)

        ignore_count = context.ignore_candles.get(cfg.symbol, {}).get(cfg.timeframe, 0)
        min_idx = max(
            ignore_count,
            cfg.ema_slow_len + cfg.slope_lookback,
            cfg.st_atr_len + 2,
            cfg.di_len + cfg.adx_smooth + 2,
        )
        if len(candles) <= min_idx:
            return [], {"note": "insufficient_data", "candles": len(candles)}

        trades: List[TradePerformance] = []
        position: Optional[_PositionState] = None

        account_balance = (
            cfg.starting_balance_usd if cfg.starting_balance_usd is not None else 100.0
        )
        self._starting_balance_usd = account_balance

        stats: Dict[str, Any] = {
            "entries_long": 0,
            "entries_short": 0,
            "entries_skipped_non_positive_balance": 0,
            "stop_exits": 0,
            "forced_close_at_end": 0,
            "reversal_exits": 0,
            "reversal_entries": 0,
            "opposite_signals_ignored": 0,
            "trend_signals_long": 0,
            "trend_signals_short": 0,
            "trailing_activations": 0,
            "trailing_updates": 0,
            "max_locked_usd": 0.0,
            "ending_balance_usd": account_balance,
        }
        last_in_range_candle: Optional[Candle] = None

        for idx in range(min_idx, len(candles)):
            candle = candles[idx]
            if candle.close_time < context.config.start:
                continue
            if candle.open_time > context.config.end:
                break
            last_in_range_candle = candle

            indicator = self._indicator_snapshot(
                index=idx,
                ema_fast=ema_fast,
                ema_mid=ema_mid,
                ema_slow=ema_slow,
                st_line=st_line,
                di_plus=di_plus,
                di_minus=di_minus,
                adx=adx,
            )
            if indicator is None:
                continue

            bull = self._is_bull_trend(candle=candle, indicator=indicator)
            bear = self._is_bear_trend(candle=candle, indicator=indicator)
            if bull:
                stats["trend_signals_long"] += 1
            if bear:
                stats["trend_signals_short"] += 1
            signal = self._build_signal(candle, index=idx, bull=bull, bear=bear)

            if position is not None:
                updated_this_candle = self._apply_trailing_stop(
                    position=position,
                    candle=candle,
                    stats=stats,
                )

                stop_hit, stop_fill = self._stop_hit_price(
                    position,
                    candle,
                    ignore_open_gap=updated_this_candle,
                )
                if stop_hit:
                    trade = self._close_trade(
                        position=position,
                        exit_price=stop_fill,
                        exit_time=candle.close_time,
                        reason="stop_loss",
                    )
                    trades.append(trade)
                    account_balance += trade.pnl
                    stats["stop_exits"] += 1
                    position = None
                    continue

                if signal is not None and signal.direction != position.direction:
                    if cfg.reverse_on_opposite_signal:
                        trade = self._close_trade(
                            position=position,
                            exit_price=signal.close_price,
                            exit_time=signal.close_time,
                            reason="reversal",
                        )
                        trades.append(trade)
                        account_balance += trade.pnl
                        stats["reversal_exits"] += 1
                        position = None

                        reopened = self._open_position(
                            signal=signal,
                            balance=account_balance,
                        )
                        if reopened is not None:
                            position = reopened
                            if signal.direction == "long":
                                stats["entries_long"] += 1
                            else:
                                stats["entries_short"] += 1
                            stats["reversal_entries"] += 1
                        else:
                            stats["entries_skipped_non_positive_balance"] += 1
                    else:
                        stats["opposite_signals_ignored"] += 1
                continue

            if signal is None:
                continue
            opened = self._open_position(signal=signal, balance=account_balance)
            if opened is None:
                stats["entries_skipped_non_positive_balance"] += 1
                continue
            position = opened
            if signal.direction == "long":
                stats["entries_long"] += 1
            else:
                stats["entries_short"] += 1

        if position is not None and last_in_range_candle is not None:
            trade = self._close_trade(
                position=position,
                exit_price=last_in_range_candle.close,
                exit_time=last_in_range_candle.close_time,
                reason="forced_end_close",
            )
            trades.append(trade)
            account_balance += trade.pnl
            stats["forced_close_at_end"] += 1

        stats["ending_balance_usd"] = account_balance
        return trades, stats

    def _build_signal(
        self, candle: Candle, *, index: int, bull: bool, bear: bool
    ) -> Optional[_SignalSnapshot]:
        if bull == bear:
            return None
        return _SignalSnapshot(
            direction="long" if bull else "short",
            candle_index=index,
            close_price=candle.close,
            close_time=candle.close_time,
        )

    def _indicator_snapshot(
        self,
        *,
        index: int,
        ema_fast: Sequence[Optional[float]],
        ema_mid: Sequence[Optional[float]],
        ema_slow: Sequence[Optional[float]],
        st_line: Sequence[Optional[float]],
        di_plus: Sequence[Optional[float]],
        di_minus: Sequence[Optional[float]],
        adx: Sequence[Optional[float]],
    ) -> Optional[_IndicatorSnapshot]:
        slow_prev_idx = index - self._config.slope_lookback
        slow_prev = ema_slow[slow_prev_idx] if slow_prev_idx >= 0 else None
        values = (
            ema_fast[index],
            ema_mid[index],
            ema_slow[index],
            st_line[index],
            di_plus[index],
            di_minus[index],
            adx[index],
            slow_prev,
        )
        if any(value is None for value in values):
            return None
        return _IndicatorSnapshot(
            fast=float(values[0]),
            mid=float(values[1]),
            slow=float(values[2]),
            supertrend=float(values[3]),
            plus_di=float(values[4]),
            minus_di=float(values[5]),
            adx=float(values[6]),
            slow_prev=float(values[7]),
        )

    def _is_bull_trend(
        self,
        *,
        candle: Candle,
        indicator: _IndicatorSnapshot,
    ) -> bool:
        cfg = self._config
        return (
            candle.close > indicator.supertrend
            and candle.close > indicator.slow
            and indicator.fast > indicator.mid > indicator.slow
            and indicator.slow > indicator.slow_prev
            and indicator.adx >= cfg.adx_min
            and indicator.plus_di > indicator.minus_di
        )

    def _is_bear_trend(
        self,
        *,
        candle: Candle,
        indicator: _IndicatorSnapshot,
    ) -> bool:
        cfg = self._config
        return (
            candle.close < indicator.supertrend
            and candle.close < indicator.slow
            and indicator.fast < indicator.mid < indicator.slow
            and indicator.slow < indicator.slow_prev
            and indicator.adx >= cfg.adx_min
            and indicator.minus_di > indicator.plus_di
        )

    def _open_position(
        self, *, signal: _SignalSnapshot, balance: float
    ) -> Optional[_PositionState]:
        qty = self._position_qty(signal.close_price, balance)
        if qty <= 0:
            return None
        initial_stop_price = self._initial_stop_price(
            direction=signal.direction,
            entry_price=signal.close_price,
        )
        return _PositionState(
            direction=signal.direction,
            entry_time=signal.close_time,
            entry_index=signal.candle_index,
            entry_balance_usd=balance,
            entry_price=signal.close_price,
            qty=qty,
            initial_stop_price=initial_stop_price,
            stop_price=initial_stop_price,
        )

    def _apply_trailing_stop(
        self,
        *,
        position: _PositionState,
        candle: Candle,
        stats: Dict[str, Any],
    ) -> bool:
        # Best-case intrabar model: favorable excursion occurs before the stop test.
        favorable_mark = candle.high if position.direction == "long" else candle.low
        new_stop = self._candidate_stop(position, favorable_mark)
        if new_stop is None:
            return False

        prev_stop = position.stop_price
        if position.direction == "long":
            position.stop_price = max(position.stop_price, new_stop)
        else:
            position.stop_price = min(position.stop_price, new_stop)

        if not position.trailing_active:
            position.trailing_active = True
            stats["trailing_activations"] += 1

        if abs(position.stop_price - prev_stop) <= 1e-12:
            return False

        stats["trailing_updates"] += 1
        locked_pnl = max(0.0, self._open_pnl_usd(position, position.stop_price))
        stats["max_locked_usd"] = max(float(stats["max_locked_usd"]), locked_pnl)
        return True

    def _initial_stop_price(self, *, direction: str, entry_price: float) -> float:
        if direction == "long":
            return entry_price * (1.0 - self._config.hard_stop_loss_pct / 100.0)
        return entry_price * (1.0 + self._config.hard_stop_loss_pct / 100.0)

    def _open_pnl_usd(self, position: _PositionState, mark_price: float) -> float:
        if position.direction == "long":
            return (mark_price - position.entry_price) * position.qty
        return (position.entry_price - mark_price) * position.qty

    def _position_margin_usd(self, balance: float) -> float:
        return balance * (self._config.position_balance_pct / 100.0)

    def _position_qty(self, price: float, balance: float) -> float:
        margin_usd = self._position_margin_usd(balance)
        if price <= 0 or margin_usd <= 0:
            return 0.0
        return (margin_usd * self._config.leverage) / price

    def _position_margin_from_qty(self, entry_price: float, qty: float) -> float:
        if self._config.leverage <= 0:
            return 0.0
        return (entry_price * qty) / self._config.leverage

    def _current_return(self, position: _PositionState, mark_price: float) -> float:
        margin = self._position_margin_from_qty(position.entry_price, position.qty)
        if margin <= 0:
            return 0.0
        pnl = self._open_pnl_usd(position, mark_price)
        return (pnl / margin) * 100.0

    def _candidate_stop(
        self, position: _PositionState, favorable_mark: float
    ) -> Optional[float]:
        if position.entry_price <= 0:
            return None
        if position.direction == "long":
            favorable_move_pct = (
                (favorable_mark - position.entry_price) / position.entry_price
            ) * 100.0
        else:
            favorable_move_pct = (
                (position.entry_price - favorable_mark) / position.entry_price
            ) * 100.0
        if favorable_move_pct < self._config.trail_start_pct:
            return None
        ret = self._current_return(position, favorable_mark)
        trail_offset_pct = self._config.trail_offset_pct
        if ret > self._LOW_PROFIT_THRESHOLD_PCT:
            if ret <= self._MID_PROFIT_THRESHOLD_PCT:
                trail_offset_pct = self._MID_TRAIL_OFFSET_PCT
            else:
                trail_offset_pct = self._HIGH_TRAIL_OFFSET_PCT
        if position.direction == "long":
            return favorable_mark * (1.0 - trail_offset_pct / 100.0)
        return favorable_mark * (1.0 + trail_offset_pct / 100.0)

    def _stop_hit_price(
        self,
        position: _PositionState,
        candle: Candle,
        *,
        ignore_open_gap: bool = False,
    ) -> Tuple[bool, float]:
        stop = position.stop_price
        if position.direction == "long":
            if not ignore_open_gap and candle.open <= stop:
                return True, candle.open
            if candle.low <= stop:
                return True, stop
            return False, 0.0
        if not ignore_open_gap and candle.open >= stop:
            return True, candle.open
        if candle.high >= stop:
            return True, stop
        return False, 0.0

    def _close_trade(
        self,
        *,
        position: _PositionState,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ) -> TradePerformance:
        notional = position.entry_price * position.qty
        margin = notional / self._config.leverage if self._config.leverage > 0 else 0.0
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.qty
        else:
            pnl = (position.entry_price - exit_price) * position.qty
        ret_pct = (pnl / margin) * 100 if margin > 0 else 0.0
        return TradePerformance(
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            return_pct=ret_pct,
            notes=reason,
            metadata={
                "direction": position.direction,
                "entry_index": position.entry_index,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "qty": position.qty,
                "initial_stop_price": position.initial_stop_price,
                "stop_at_exit": position.stop_price,
                "position_margin_usd": margin,
                "position_size_pct": self._config.position_balance_pct,
                "entry_balance_usd": position.entry_balance_usd,
                "starting_balance_usd": self._starting_balance_usd,
                "notional_usd": notional,
                "leverage": self._config.leverage,
            },
        )

    @staticmethod
    def _true_range(candles: Sequence[Candle]) -> List[float]:
        out: List[float] = []
        for i, candle in enumerate(candles):
            if i == 0:
                out.append(candle.high - candle.low)
                continue
            prev_close = candles[i - 1].close
            out.append(
                max(
                    candle.high - candle.low,
                    abs(candle.high - prev_close),
                    abs(candle.low - prev_close),
                )
            )
        return out

    @staticmethod
    def _rma(values: Sequence[float], length: int) -> List[Optional[float]]:
        if length <= 0 or len(values) < length:
            return [None] * len(values)
        out: List[Optional[float]] = [None] * len(values)
        seed = sum(values[:length]) / float(length)
        out[length - 1] = seed
        prev = seed
        for i in range(length, len(values)):
            prev = (prev * (length - 1) + values[i]) / float(length)
            out[i] = prev
        return out

    def _supertrend_line(
        self, candles: Sequence[Candle], atr_len: int, factor: float
    ) -> List[Optional[float]]:
        tr = self._true_range(candles)
        atr = self._rma(tr, atr_len)
        n = len(candles)
        out: List[Optional[float]] = [None] * n
        final_upper: List[Optional[float]] = [None] * n
        final_lower: List[Optional[float]] = [None] * n
        for i, candle in enumerate(candles):
            atr_i = atr[i]
            if atr_i is None:
                continue
            hl2 = (candle.high + candle.low) / 2.0
            upper = hl2 + factor * atr_i
            lower = hl2 - factor * atr_i
            if i == 0 or final_upper[i - 1] is None or final_lower[i - 1] is None:
                final_upper[i] = upper
                final_lower[i] = lower
                out[i] = lower
                continue
            prev_fu = float(final_upper[i - 1])
            prev_fl = float(final_lower[i - 1])
            prev_close = candles[i - 1].close
            final_upper[i] = (
                upper if upper < prev_fu or prev_close > prev_fu else prev_fu
            )
            final_lower[i] = (
                lower if lower > prev_fl or prev_close < prev_fl else prev_fl
            )
            prev_st = out[i - 1]
            if prev_st is None:
                out[i] = final_lower[i]
            elif abs(float(prev_st) - prev_fu) < 1e-12:
                out[i] = (
                    final_upper[i]
                    if candle.close <= float(final_upper[i])
                    else final_lower[i]
                )
            else:
                out[i] = (
                    final_lower[i]
                    if candle.close >= float(final_lower[i])
                    else final_upper[i]
                )
        return out

    def _dmi(
        self, candles: Sequence[Candle], di_len: int, adx_smooth: int
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        n = len(candles)
        plus_dm: List[float] = [0.0] * n
        minus_dm: List[float] = [0.0] * n
        tr = self._true_range(candles)
        for i in range(1, n):
            up = candles[i].high - candles[i - 1].high
            down = candles[i - 1].low - candles[i].low
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0

        tr_rma = self._rma(tr, di_len)
        plus_rma = self._rma(plus_dm, di_len)
        minus_rma = self._rma(minus_dm, di_len)
        plus_di: List[Optional[float]] = [None] * n
        minus_di: List[Optional[float]] = [None] * n
        dx_values: List[Optional[float]] = [None] * n
        for i in range(n):
            tr_v = tr_rma[i]
            p_v = plus_rma[i]
            m_v = minus_rma[i]
            if tr_v is None or p_v is None or m_v is None or tr_v <= 0:
                continue
            p_di = 100.0 * (p_v / tr_v)
            m_di = 100.0 * (m_v / tr_v)
            plus_di[i] = p_di
            minus_di[i] = m_di
            denom = p_di + m_di
            if denom > 0:
                dx_values[i] = 100.0 * abs(p_di - m_di) / denom

        adx: List[Optional[float]] = [None] * n
        valid_dx_points = [
            (idx, float(value))
            for idx, value in enumerate(dx_values)
            if value is not None
        ]
        if len(valid_dx_points) < adx_smooth:
            return plus_di, minus_di, adx

        seed_window = valid_dx_points[:adx_smooth]
        prev_adx = sum(point[1] for point in seed_window) / float(adx_smooth)
        seed_index = seed_window[-1][0]
        adx[seed_index] = prev_adx
        for idx, dx_val in valid_dx_points[adx_smooth:]:
            prev_adx = (prev_adx * (adx_smooth - 1) + dx_val) / float(adx_smooth)
            adx[idx] = prev_adx
        return plus_di, minus_di, adx
