"""Decision logging and aggregate reporting for the EMA/AVWAP strategy."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Sequence

from ..base import TradePerformance


class EmaAvwapReportingMixin:
    """Encapsulates side-effect-free reporting and bounded decision logging."""

    def _record_event(
        self,
        *,
        decision_log: List[Dict[str, Any]],
        stats: Dict[str, Any],
        event: str,
        payload: Dict[str, Any],
    ) -> None:
        if len(decision_log) >= self._config.max_decision_log_entries:
            stats["decision_log_truncated_count"] += 1
            return
        item = {"event": event}
        item.update(payload)
        decision_log.append(item)

    def _summarize_trade_stats(self, trades: Sequence[TradePerformance]) -> Dict[str, Any]:
        if not trades:
            return {
                "trade_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_notional_multiple": 0.0,
                "avg_return_pct": 0.0,
                "avg_holding_bars": 0.0,
                "long_trade_count": 0,
                "short_trade_count": 0,
                "gross_profit_long": 0.0,
                "gross_profit_short": 0.0,
                "gross_loss_long": 0.0,
                "gross_loss_short": 0.0,
                "total_net_pnl": 0.0,
            }

        wins = sum(1 for trade in trades if trade.pnl > 0)
        losses = sum(1 for trade in trades if trade.pnl < 0)
        long_trades: List[TradePerformance] = []
        short_trades: List[TradePerformance] = []
        notional_multiples: List[float] = []
        hold_bars: List[int] = []
        gross_profit_long = 0.0
        gross_profit_short = 0.0
        gross_loss_long = 0.0
        gross_loss_short = 0.0

        for trade in trades:
            metadata = dict(trade.metadata or {})
            direction = str(metadata.get("direction", ""))
            notional_multiples.append(float(metadata.get("notional_multiple", 0.0)))
            hold_bars.append(int(metadata.get("holding_bars", 0)))

            if direction == "long":
                long_trades.append(trade)
                if trade.pnl >= 0:
                    gross_profit_long += trade.pnl
                else:
                    gross_loss_long += abs(trade.pnl)
            elif direction == "short":
                short_trades.append(trade)
                if trade.pnl >= 0:
                    gross_profit_short += trade.pnl
                else:
                    gross_loss_short += abs(trade.pnl)

        return {
            "trade_count": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trades),
            "avg_notional_multiple": (
                mean(notional_multiples) if notional_multiples else 0.0
            ),
            "avg_return_pct": mean(trade.return_pct for trade in trades),
            "avg_holding_bars": mean(hold_bars) if hold_bars else 0.0,
            "best_trade_pnl": max(trade.pnl for trade in trades),
            "worst_trade_pnl": min(trade.pnl for trade in trades),
            "long_trade_count": len(long_trades),
            "short_trade_count": len(short_trades),
            "gross_profit_long": gross_profit_long,
            "gross_profit_short": gross_profit_short,
            "gross_loss_long": gross_loss_long,
            "gross_loss_short": gross_loss_short,
            "total_net_pnl": sum(trade.pnl for trade in trades),
        }

    def _config_as_dict(self) -> Dict[str, Any]:
        cfg = self._config
        return {
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "initial_equity": cfg.initial_equity,
            "leverage": cfg.leverage,
            "position_notional_pct": cfg.position_notional_pct,
            "minimum_balance_usdt": cfg.minimum_balance_usdt,
            "max_position_size_pct": cfg.max_position_size_pct,
            "max_entry_notional_usdt": cfg.max_entry_notional_usdt,
            "ema_length": cfg.ema_length,
            "consecutive_count": cfg.consecutive_count,
            "ema_validation_mode": cfg.ema_validation_mode,
            "setup_waiting_replacement_mode": cfg.setup_waiting_replacement_mode,
            "max_setup_age_bars": cfg.max_setup_age_bars,
            "max_entry_deviation_pct": cfg.max_entry_deviation_pct,
            "position_sizing_mode": cfg.position_sizing_mode,
            "entry_mode": cfg.entry_mode.value,
            "exit_mode": cfg.exit_mode.value,
            "exit_band": cfg.exit_band.value,
            "avwap_multiplier_1": cfg.avwap_multiplier_1,
            "avwap_multiplier_2": cfg.avwap_multiplier_2,
            "avwap_multiplier_3": cfg.avwap_multiplier_3,
            "rigid_stop_loss_pct": cfg.rigid_stop_loss_pct,
            "trailing_activation_threshold_pct": cfg.trailing_activation_threshold_pct,
            "trailing_gap_pct": cfg.trailing_gap_pct,
            "maker_fee_pct": cfg.maker_fee_pct,
            "taker_fee_pct": cfg.taker_fee_pct,
            "entry_slippage_pct": cfg.entry_slippage_pct,
            "exit_slippage_pct": cfg.exit_slippage_pct,
            "use_gap_cross_detection": cfg.use_gap_cross_detection,
            "max_decision_log_entries": cfg.max_decision_log_entries,
        }
