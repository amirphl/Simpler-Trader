from __future__ import annotations

import unittest
from datetime import datetime, timezone
from argparse import Namespace
from unittest.mock import Mock
from unittest.mock import patch

from cmd.live_trading import _shared
from cmd.live_trading.ema_avwap_pullback_main import build_parser
from live_trading.exchange import ExchangeConfig, MarginMode, OrderResult, OrderType, Position, PositionSide
from live_trading.exchanges.routed import ZecWeexRoutedExchange


def _position(symbol: str) -> Position:
    return Position(
        symbol=symbol,
        side=PositionSide.LONG,
        size=1.0,
        entry_price=100.0,
        leverage=2.0,
        margin_mode=MarginMode.ISOLATED,
        unrealized_pnl=0.0,
        position_id=f"{symbol}-position",
    )


def _order(symbol: str) -> OrderResult:
    return OrderResult(
        order_id=f"{symbol}-order",
        symbol=symbol,
        side=PositionSide.LONG,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=1.0,
        status="NEW",
        timestamp=datetime.now(timezone.utc),
    )


class ZecWeexRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bitunix = Mock()
        self.bitunix._config = ExchangeConfig(api_key="bitunix", api_secret="secret")
        self.weex = Mock()
        self.router = ZecWeexRoutedExchange(self.bitunix, self.weex)

    def test_zec_execution_and_balance_use_weex(self) -> None:
        self.bitunix.fetch_price.return_value = 101.0
        self.weex.fetch_price.return_value = 102.0
        self.bitunix.get_account_balance_for_symbol.return_value = 50.0
        self.weex.get_account_balance_for_symbol.return_value = 75.0

        self.assertEqual(self.router.fetch_price("ETHUSDT"), 101.0)
        self.assertEqual(self.router.fetch_price("ZECUSDT"), 102.0)
        self.assertEqual(self.router.get_account_balance_for_symbol("ETHUSDT"), 50.0)
        self.assertEqual(self.router.get_account_balance_for_symbol("ZECUSDT"), 75.0)
        self.bitunix.fetch_price.assert_called_with("ETHUSDT")
        self.weex.fetch_price.assert_called_with("ZECUSDT")
        self.bitunix.get_account_balance_for_symbol.assert_called_with("ETHUSDT")
        self.weex.get_account_balance_for_symbol.assert_called_with("ZECUSDT")

    def test_zec_orders_and_positions_use_weex(self) -> None:
        self.bitunix.open_limit_position.return_value = _order("ETHUSDT")
        self.weex.open_limit_position.return_value = _order("ZECUSDT")
        self.bitunix.get_current_positions.return_value = [_position("ETHUSDT"), _position("ZECUSDT")]
        self.weex.get_current_positions.return_value = [_position("ZECUSDT"), _position("BTCUSDT")]

        self.router.open_limit_position(
            "ZECUSDT",
            PositionSide.LONG,
            1.0,
            100.0,
            2,
            MarginMode.ISOLATED,
            client_id="zec-entry",
        )

        self.weex.open_limit_position.assert_called_once()
        self.bitunix.open_limit_position.assert_not_called()
        self.assertEqual(
            [position.symbol for position in self.router.get_current_positions()],
            ["ETHUSDT", "ZECUSDT"],
        )

    def test_all_other_symbols_remain_on_bitunix(self) -> None:
        self.bitunix.open_limit_position.return_value = _order("ETHUSDT")

        self.router.open_limit_position(
            "ETHUSDT",
            PositionSide.LONG,
            1.0,
            100.0,
            2,
            MarginMode.ISOLATED,
        )

        self.bitunix.open_limit_position.assert_called_once()
        self.weex.open_limit_position.assert_not_called()

    def test_ema_parser_defaults_to_bitunix(self) -> None:
        self.assertEqual(build_parser().parse_args([]).exchange, "bitunix")

    def test_ema_rejects_spot_before_constructing_an_exchange(self) -> None:
        args = Namespace(
            exchange="bitunix",
            strategy_name="ema_avwap_pullback",
            symbols="ETHUSDT",
            api_key="bitunix-key",
            api_secret="bitunix-secret",
            api_passphrase="",
            trading_mode="spot",
            testnet=False,
            live=True,
            proxy=None,
            http_proxy=None,
            https_proxy=None,
            exchange_base_url=None,
        )

        with self.assertRaisesRegex(ValueError, "requires TRADING_MODE=futures"):
            _shared.create_exchange(args, Mock())

    @patch("live_trading.exchanges.WeexExchange")
    @patch("live_trading.exchanges.BitunixExchange")
    def test_bitunix_ema_config_builds_weex_fallback_for_zec(
        self, bitunix_cls, weex_cls
    ) -> None:
        bitunix = bitunix_cls.return_value
        weex = weex_cls.return_value
        args = Namespace(
            exchange="bitunix",
            strategy_name="ema_avwap_pullback",
            symbols="ETHUSDT,ZECUSDT",
            api_key="bitunix-key",
            api_secret="bitunix-secret",
            api_passphrase="",
            weex_api_key="weex-key",
            weex_api_secret="weex-secret",
            weex_api_passphrase="weex-passphrase",
            trading_mode="futures",
            testnet=False,
            live=True,
            proxy=None,
            http_proxy=None,
            https_proxy=None,
            exchange_base_url=None,
        )

        exchange = _shared.create_exchange(args, Mock())

        self.assertIsInstance(exchange, ZecWeexRoutedExchange)
        bitunix_cls.assert_called_once()
        weex_cls.assert_called_once()
        weex_config = weex_cls.call_args.args[0]
        self.assertEqual(weex_config.api_key, "weex-key")
        self.assertEqual(weex_config.api_secret, "weex-secret")
        self.assertEqual(weex_config.passphrase, "weex-passphrase")
        self.assertEqual(weex_config.trading_mode, "futures")
        self.assertIs(exchange._primary, bitunix)
        self.assertIs(exchange._weex, weex)
