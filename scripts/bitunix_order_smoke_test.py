#!/usr/bin/env python3
"""Simple Bitunix order smoke test using existing live-trading config file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

from cmd.live_trading._shared import load_env_config
from live_trading.exchange import ExchangeConfig, MarginMode, PositionSide
from live_trading.exchanges import BitunixExchange


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_proxies(config: Dict[str, str]) -> Optional[Dict[str, str]]:
    proxy = config.get("proxy", "")
    http_proxy = config.get("http_proxy", "")
    https_proxy = config.get("https_proxy", "")
    if proxy:
        return {"http": proxy, "https": proxy}
    out: Dict[str, str] = {}
    if http_proxy:
        out["http"] = http_proxy
    if https_proxy:
        out["https"] = https_proxy
    return out or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Place a simple Bitunix test order using config file secrets."
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("./configs/live_trading.pinbar_magic_v2.env"),
        help="Path to live trading .env config file",
    )
    parser.add_argument("--symbol", default="DOGEUSDT", help="Symbol to trade")
    parser.add_argument("--quantity", type=float, default=1.0, help="Order quantity")
    parser.add_argument(
        "--side",
        choices=["long", "short"],
        default="long",
        help="Order side",
    )
    parser.add_argument(
        "--order-type",
        choices=["market", "limit"],
        default="market",
        help="Order type",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        help="Limit price (required when --order-type=limit)",
    )
    parser.add_argument("--leverage", type=int, default=10, help="Leverage")
    parser.add_argument(
        "--margin-mode",
        choices=["isolated", "cross"],
        default="isolated",
        help="Margin mode",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live mode even if TESTNET=true in config",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Optional take-profit price",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Optional stop-loss price",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("bitunix_order_smoke_test")

    config = load_env_config(args.config_file)
    api_key = config.get("api_key", "")
    api_secret = config.get("api_secret", "")
    exchange_name = str(config.get("exchange", "")).strip().lower()
    if exchange_name and exchange_name != "bitunix":
        raise ValueError(
            f"Config exchange is '{exchange_name}', but this script only supports 'bitunix'."
        )
    if not api_key:
        raise ValueError("Missing API_KEY in config/environment")
    if not api_secret:
        raise ValueError("Missing API_SECRET in config/environment")

    config_testnet = _parse_bool(config.get("testnet", "true") or "true")
    testnet = False if args.live else config_testnet
    exchange_config = ExchangeConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        proxies=_build_proxies(config),
        passphrase=(config.get("api_passphrase", "") or None),
        base_url=(config.get("exchange_base_url", "") or None),
    )

    side = PositionSide.LONG if args.side == "long" else PositionSide.SHORT
    margin_mode = (
        MarginMode.ISOLATED if args.margin_mode == "isolated" else MarginMode.CROSS
    )
    symbol = args.symbol.strip().upper()

    if args.order_type == "limit" and args.price is None:
        raise ValueError("--price is required when --order-type=limit")
    if args.quantity <= 0:
        raise ValueError("--quantity must be > 0")

    exchange = BitunixExchange(exchange_config, log)
    try:
        log.info(
            "Submitting %s %s order: symbol=%s qty=%s mode=%s",
            args.side.upper(),
            args.order_type.upper(),
            symbol,
            args.quantity,
            "testnet" if testnet else "live",
        )
        if args.order_type == "market":
            result = exchange.open_market_position(
                symbol=symbol,
                side=side,
                quantity=args.quantity,
                leverage=args.leverage,
                margin_mode=margin_mode,
                take_profit=args.take_profit,
                stop_loss=args.stop_loss,
            )
        else:
            result = exchange.open_limit_position(
                symbol=symbol,
                side=side,
                quantity=args.quantity,
                price=float(args.price),
                leverage=args.leverage,
                margin_mode=margin_mode,
                take_profit=args.take_profit,
                stop_loss=args.stop_loss,
            )

        print("Order submitted successfully")
        print(f"order_id={result.order_id}")
        print(f"symbol={result.symbol}")
        print(f"side={result.side.value}")
        print(f"type={result.order_type.value}")
        print(f"quantity={result.quantity}")
        print(f"price={result.price}")
        print(f"status={result.status}")
        return 0
    finally:
        exchange.close()


if __name__ == "__main__":
    raise SystemExit(main())
