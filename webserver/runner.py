from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, cast

from backtest import (
    BacktestRunConfig,
    BaseBacktester,
    EngulfingStrategy,
    EngulfingStrategyConfig,
    PinBarMagicStrategy,
    PinBarMagicStrategyConfig,
    PinbarStrategy,
    PinbarStrategyConfig,
)
from backtest.pinbar_magic_strategy_v2 import (
    PinBarMagicStrategyConfigV2,
    PinBarMagicStrategyV2,
)
from backtest.stochastic_fsm_strategy import (
    StochasticRsiFsmConfig,
    StochasticRsiFsmStrategy,
)
from candle_downloader.binance import BinanceClient, BinanceClientConfig
from candle_downloader.downloader import CandleDownloader
from candle_downloader.storage import build_store

from .models import (
    BacktestSubmission,
    EngulfingStrategyParams,
    PinbarMagicStrategyParams,
    PinbarMagicStrategyParamsV2,
    PinbarStrategyParams,
    StochasticFsmParams,
)


def run_backtest_job(
    job_id: str, submission: BacktestSubmission, *, cache_dir: Path
) -> Dict[str, Any]:
    """Execute the requested backtest synchronously and return its report."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    params = submission.params
    proxies = {}
    if params.http_proxy:
        proxies["http"] = params.http_proxy
    if params.https_proxy:
        proxies["https"] = params.https_proxy

    store = build_store("postgres", None)
    client_logger = logging.getLogger(f"web-backtest.{job_id}.binance")
    client = BinanceClient(
        BinanceClientConfig(proxies=proxies or None), logger=client_logger
    )
    downloader = CandleDownloader(client=client, store=store)

    if submission.strategy == "stochastic_fsm":
        stoch_params = cast(StochasticFsmParams, params)
        strategy = StochasticRsiFsmStrategy(
            StochasticRsiFsmConfig(
                symbols=stoch_params.symbols,
                tf_1=stoch_params.base_timeframe,
                tf_2=stoch_params.higher_timeframe,
                tf_3=stoch_params.higher_timeframe_2,
                k_period=stoch_params.k_period,
                k_slowing=stoch_params.k_slowing,
                d_period=stoch_params.d_period,
                use_d_line=stoch_params.use_d_line,
                oversold=stoch_params.oversold,
                overbought=stoch_params.overbought,
                initial_order_usdt=stoch_params.initial_order_usdt,
                initial_leverage=stoch_params.initial_leverage,
                martingale_multiplier=stoch_params.martingale_multiplier,
                martingale_multipliers=tuple(stoch_params.martingale_multipliers),
                martingale_leverages=tuple(stoch_params.martingale_leverages),
                max_concurrent_positions=stoch_params.max_concurrent_positions,
                take_profit_pct=stoch_params.take_profit_pct,
                slippage_pct=stoch_params.slippage_pct,
                maker_fee_pct=stoch_params.maker_fee_pct,
                taker_fee_pct=stoch_params.taker_fee_pct,
                funding_rate_per_day_pct=stoch_params.funding_rate_per_day_pct,
                trailing_activation_pct=stoch_params.trailing_activation_pct,
                trailing_gap_pct=stoch_params.trailing_gap_pct,
                trailing_check_interval_seconds=stoch_params.trailing_check_interval_seconds,
                max_position_days=stoch_params.max_position_days,
                margin_mode=stoch_params.margin_mode,
                aligned_high_stoch_mode=stoch_params.aligned_high_stoch_mode,
                signal_offset=stoch_params.signal_offset,
                enable_take_profit_check=stoch_params.enable_take_profit_check,
                enable_high_exit_cross=stoch_params.enable_high_exit_cross,
                use_midsold_filter=stoch_params.use_midsold_filter,
                enable_reversal_logic=stoch_params.enable_reversal_logic,
                enable_reversal_reentry=stoch_params.enable_reversal_reentry,
                trailing_use_first_entry_price=stoch_params.trailing_use_first_entry_price,
                trailing_use_close_for_stop_activation=stoch_params.trailing_use_close_for_stop_activation,
                take_profit_use_first_entry_price=stoch_params.take_profit_use_first_entry_price,
                enable_grid_martingales=stoch_params.enable_grid_martingales,
                grid_martingales_percent=stoch_params.grid_martingales_percent,
            )
        )
    elif submission.strategy == "pinbar_magic_v2":
        magic_params_v2 = cast(PinbarMagicStrategyParamsV2, params)
        strategy = PinBarMagicStrategyV2(
            PinBarMagicStrategyConfigV2(
                symbol=magic_params_v2.symbol,
                timeframe=magic_params_v2.timeframe,
                leverage=magic_params_v2.leverage,
                equity_risk_pct=magic_params_v2.equity_risk_pct,
                atr_multiple=magic_params_v2.atr_multiple,
                trail_points=magic_params_v2.trail_points,
                trail_offset=magic_params_v2.trail_offset,
                slow_sma_period=magic_params_v2.slow_sma_period,
                medium_ema_period=magic_params_v2.medium_ema_period,
                fast_ema_period=magic_params_v2.fast_ema_period,
                atr_period=magic_params_v2.atr_period,
                entry_cancel_bars=magic_params_v2.entry_cancel_bars,
                trailing_tick_timeframe=magic_params_v2.trailing_tick_timeframe.strip(),
                use_trailing_tick_emulation=magic_params_v2.use_trailing_tick_emulation,
                use_stop_fill_open_gap=magic_params_v2.use_stop_fill_open_gap,
                entry_activation_mode=magic_params_v2.entry_activation_mode.strip().lower(),
                enable_friday_close=magic_params_v2.enable_friday_close,
                friday_close_hour_utc=magic_params_v2.friday_close_hour_utc,
                enable_ema_cross_close=magic_params_v2.enable_ema_cross_close,
                risk_equity_include_unrealized=magic_params_v2.risk_equity_include_unrealized,
                risk_equity_mark_source=magic_params_v2.risk_equity_mark_source.strip().lower(),
            )
        )
    elif submission.strategy == "pinbar_magic":
        magic_params = cast(PinbarMagicStrategyParams, params)
        strategy = PinBarMagicStrategy(
            PinBarMagicStrategyConfig(
                symbol=magic_params.symbol,
                timeframe=magic_params.timeframe,
                leverage=magic_params.leverage,
                equity_risk_pct=magic_params.equity_risk_pct,
                atr_multiple=magic_params.atr_multiple,
                trail_points=magic_params.trail_points,
                trail_offset=magic_params.trail_offset,
                slow_sma_period=magic_params.slow_sma_period,
                medium_ema_period=magic_params.medium_ema_period,
                fast_ema_period=magic_params.fast_ema_period,
                atr_period=magic_params.atr_period,
                entry_cancel_bars=magic_params.entry_cancel_bars,
            )
        )
    elif submission.strategy == "pinbar":
        pinbar_params = cast(PinbarStrategyParams, params)
        strategy = PinbarStrategy(
            PinbarStrategyConfig(
                symbol=pinbar_params.symbol,
                timeframe=pinbar_params.timeframe,
                leverage=pinbar_params.leverage,
                take_profit_pct=pinbar_params.take_profit_pct,
                stop_loss_mode=pinbar_params.stop_loss_mode,
                stop_loss_pct=pinbar_params.stop_loss_pct,
                exchange_fee_pct=pinbar_params.exchange_fee_pct,
                min_shadow_body_ratio=pinbar_params.min_shadow_body_ratio,
                shadow_dominance_ratio=pinbar_params.shadow_dominance_ratio,
            )
        )
    elif submission.strategy == "engulfing":
        engulfing_params = cast(EngulfingStrategyParams, params)
        strategy = EngulfingStrategy(
            EngulfingStrategyConfig(
                symbol=engulfing_params.symbol,
                timeframe=engulfing_params.timeframe,
                window_size=engulfing_params.window_size,
                leverage=engulfing_params.leverage,
                take_profit_pct=engulfing_params.take_profit_pct,
                doji_size=engulfing_params.doji_size,
                stop_loss_mode=engulfing_params.stop_loss_mode,
                stop_loss_pct=engulfing_params.stop_loss_pct,
                skip_large_upper_wick=engulfing_params.skip_large_upper_wick,
                skip_bollinger_cross=engulfing_params.skip_bollinger_cross,
                bollinger_period=engulfing_params.bollinger_period,
                bollinger_stddev=engulfing_params.bollinger_stddev,
                enable_volume_pressure_filter=engulfing_params.enable_volume_pressure_filter,
                volume_window=engulfing_params.volume_window,
                max_volume_pressure_score=engulfing_params.max_volume_pressure_score,
                enable_stochastic_filter=engulfing_params.enable_stochastic_filter,
                stochastic_first_line=engulfing_params.stochastic_first_line,
                stochastic_first_period=engulfing_params.stochastic_first_period,
                stochastic_first_threshold=engulfing_params.stochastic_first_threshold,
                stochastic_second_line=engulfing_params.stochastic_second_line,
                stochastic_second_period=engulfing_params.stochastic_second_period,
                stochastic_second_threshold=engulfing_params.stochastic_second_threshold,
                stochastic_comparison=engulfing_params.stochastic_comparison,
                stochastic_d_smoothing=engulfing_params.stochastic_d_smoothing,
                exchange_fee_pct=engulfing_params.exchange_fee_pct,
            )
        )
    else:
        raise ValueError(f"Unknown strategy: {submission.strategy}")

    backtester = BaseBacktester(strategy=strategy, downloader=downloader, store=store)
    run_config = BacktestRunConfig(
        start=_ensure_utc(submission.start),
        end=_ensure_utc(submission.end),
        initial_capital=submission.initial_capital,
        override_download=submission.override_download,
        risk_free_rate=params.risk_free_rate,
        warmup_days=submission.warmup_days,
    )

    try:
        report = backtester.run(run_config)
        return {
            "job_id": job_id,
            "strategy": submission.strategy,
            "report": report.as_dict(),
        }
    finally:
        store.close()
        client.close()


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
