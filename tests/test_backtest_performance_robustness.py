from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from backtest.base import BacktestReport, BacktestRunConfig, BacktestStatistics, TradePerformance
from backtest.performance import build_performance_statistics
from backtest.robustness import (
    MonteCarloConfig,
    ParameterCandidate,
    WalkForwardConfig,
    build_walk_forward_windows,
    run_monte_carlo_suite,
    run_remove_best_trades_analysis,
    run_walk_forward,
)


def _trade(
    *,
    entry_offset_days: int,
    exit_offset_days: int,
    pnl: float,
    return_pct: float,
    metadata: Mapping[str, str | float | int | None] | None = None,
) -> TradePerformance:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return TradePerformance(
        entry_time=start + timedelta(days=entry_offset_days),
        exit_time=start + timedelta(days=exit_offset_days),
        pnl=pnl,
        return_pct=return_pct,
        metadata=metadata,
    )


class _FakeBacktester:
    def __init__(self, parameters: Mapping[str, Any], initial_capital: float) -> None:
        self._edge = float(parameters.get("edge", 0.0))
        self._initial_capital = initial_capital

    def run(self, config: BacktestRunConfig) -> BacktestReport:
        days = max((config.end - config.start).days, 1)
        pnl = self._edge * days
        trade = TradePerformance(
            entry_time=config.start + timedelta(hours=1),
            exit_time=config.end - timedelta(hours=1),
            pnl=pnl,
            return_pct=(pnl / self._initial_capital) * 100.0,
            metadata={"qty": 1, "entry_price": 100, "exit_price": 100 + pnl},
        )
        stats = BacktestStatistics(
            total_trades=1,
            winning_trades=1 if pnl > 0 else 0,
            losing_trades=1 if pnl < 0 else 0,
            win_rate=1.0 if pnl > 0 else 0.0,
            gross_profit=max(pnl, 0.0),
            gross_loss=min(pnl, 0.0),
            net_profit=pnl,
            net_profit_pct=(pnl / self._initial_capital) * 100.0,
            average_return_pct=(pnl / self._initial_capital) * 100.0,
            expectancy=pnl,
            equity_curve=[self._initial_capital, self._initial_capital + pnl],
        )
        return BacktestReport(
            strategy_name="FakeStrategy",
            config=config,
            statistics=stats,
            trades=[trade],
        )


class BacktestPerformanceRobustnessTests(unittest.TestCase):
    def test_detailed_performance_statistics_include_requested_metrics(self) -> None:
        trades = [
            _trade(
                entry_offset_days=0,
                exit_offset_days=1,
                pnl=10.0,
                return_pct=10.0,
                metadata={
                    "qty": 2,
                    "entry_raw_price": 10,
                    "entry_price": 10.1,
                    "exit_raw_price": 15,
                    "exit_price": 14.8,
                    "entry_fee": 0.2,
                    "exit_fee": 0.3,
                },
            ),
            _trade(entry_offset_days=2, exit_offset_days=3, pnl=-5.0, return_pct=-5.0),
            _trade(entry_offset_days=4, exit_offset_days=5, pnl=-2.0, return_pct=-2.0),
            _trade(entry_offset_days=6, exit_offset_days=7, pnl=8.0, return_pct=8.0),
        ]

        stats = build_performance_statistics(
            trades=trades,
            initial_capital=100.0,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
            strategy_trials=10,
        )

        self.assertEqual(stats["number_of_trades"], 4)
        self.assertAlmostEqual(stats["average_win"], 9.0)
        self.assertAlmostEqual(stats["average_loss"], -3.5)
        self.assertAlmostEqual(stats["expectancy_per_trade"], 2.75)
        self.assertEqual(stats["longest_losing_streak"], 2)
        self.assertIn("sortino_ratio", stats)
        self.assertIn("calmar_ratio", stats)
        self.assertIn("return_skewness", stats)
        self.assertIn("return_excess_kurtosis", stats)
        self.assertIn("2024-01", stats["monthly_return_distribution_pct"])
        self.assertAlmostEqual(stats["fee_drag"], 0.5)
        self.assertGreater(stats["slippage_drag"], 0.0)
        self.assertEqual(stats["strategy_trials"], 10)

    def test_monte_carlo_suite_and_remove_best_trades_report_risk(self) -> None:
        trades = [
            _trade(entry_offset_days=0, exit_offset_days=1, pnl=12.0, return_pct=12.0),
            _trade(entry_offset_days=2, exit_offset_days=3, pnl=-4.0, return_pct=-4.0),
            _trade(entry_offset_days=4, exit_offset_days=5, pnl=6.0, return_pct=6.0),
            _trade(entry_offset_days=6, exit_offset_days=7, pnl=-3.0, return_pct=-3.0),
        ]
        config = MonteCarloConfig(
            iterations=100,
            seed=7,
            initial_capital=100.0,
            block_size=2,
            drawdown_threshold_pct=10.0,
        )

        suite = run_monte_carlo_suite(
            trades=trades,
            config=config,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
        removal = run_remove_best_trades_analysis(
            trades=trades,
            initial_capital=100.0,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
            removal_percentages=(25.0,),
        )

        self.assertIn("trade_reshuffling", suite)
        self.assertIn("block_bootstrap", suite)
        self.assertIn("slippage", suite)
        self.assertIn("remove_best_trades", suite)
        self.assertIn("max_drawdown_pct_p95", suite["block_bootstrap"])
        self.assertIn("probability_of_drawdown_ge_threshold", suite["trade_reshuffling"])
        removed_stats = removal["remove_top_25_pct"]["statistics"]
        self.assertEqual(removal["remove_top_25_pct"]["removed_trade_count"], 1)
        self.assertLess(removed_stats["net_profit"], sum(trade.pnl for trade in trades))

    def test_walk_forward_selects_on_training_and_combines_oos_trades(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        run_config = BacktestRunConfig(
            start=start,
            end=datetime(2024, 5, 1, tzinfo=timezone.utc),
            initial_capital=100.0,
        )
        wf_config = WalkForwardConfig(
            start=run_config.start,
            end=run_config.end,
            train_window=timedelta(days=30),
            test_window=timedelta(days=15),
            step=timedelta(days=30),
        )

        result = run_walk_forward(
            build_backtester=lambda parameters, initial_capital: _FakeBacktester(
                parameters,
                initial_capital,
            ),
            run_config=run_config,
            walk_forward_config=wf_config,
            candidates=[
                ParameterCandidate("weak", {"edge": 0.1}),
                ParameterCandidate("strong", {"edge": 0.3}),
            ],
        )

        self.assertEqual(len(result.folds), len(build_walk_forward_windows(wf_config)))
        self.assertTrue(result.folds)
        self.assertTrue(
            all(fold.selected_candidate.name == "strong" for fold in result.folds)
        )
        self.assertEqual(
            result.combined_statistics["number_of_trades"],
            len(result.combined_trades),
        )
        self.assertGreater(result.combined_statistics["net_profit"], 0.0)


if __name__ == "__main__":
    unittest.main()
