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
    PinbarStrategy,
    PinbarStrategyConfig,
)
from candle_downloader.binance import BinanceClient, BinanceClientConfig
from candle_downloader.downloader import CandleDownloader
from candle_downloader.storage import build_store

from .models import BacktestSubmission, EngulfingStrategyParams, PinbarStrategyParams


def run_backtest_job(job_id: str, submission: BacktestSubmission, *, cache_dir: Path, store_path: Path) -> Dict[str, Any]:
    """Execute the requested backtest synchronously and return its report."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    params = submission.params
    proxies = {}
    if params.http_proxy:
        proxies["http"] = params.http_proxy
    if params.https_proxy:
        proxies["https"] = params.https_proxy


    store = build_store("sqlite", store_path)
    client_logger = logging.getLogger(f"web-backtest.{job_id}.binance")
    client = BinanceClient(BinanceClientConfig(proxies=proxies or None), logger=client_logger)
    downloader = CandleDownloader(client=client, store=store)

    if submission.strategy == "pinbar":
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
    else:
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

    backtester = BaseBacktester(strategy=strategy, downloader=downloader, store=store)
    run_config = BacktestRunConfig(
        start=_ensure_utc(submission.start),
        end=_ensure_utc(submission.end),
        initial_capital=submission.initial_capital,
        override_download=submission.override_download,
        risk_free_rate=params.risk_free_rate,
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

