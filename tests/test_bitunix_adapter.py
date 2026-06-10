from __future__ import annotations

import unittest

from live_trading.exchange import ExchangeConfig, MarginMode, PositionSide
from live_trading.exchanges.bitunix.adapter import BitunixExchange


class _FakeBitunixClient:
    def __init__(self, positions):
        self._positions = positions

    def get_pending_positions(self):
        if isinstance(self._positions, Exception):
            raise self._positions
        return self._positions


class _FakeKlineClient:
    def __init__(self, rows):
        self._rows = rows

    def get_kline_history(self, **kwargs):
        return list(self._rows)


class _FakeOrderClient:
    def __init__(self, *, mark_price: float = 100.0):
        self.mark_price = mark_price
        self.orders = []

    def get_trading_pairs(self, symbols=None):
        return {
            "ETHUSDT": {
                "symbol": "ETHUSDT",
                "basePrecision": 3,
                "quotePrecision": 2,
                "minTradeVolume": "0.001",
                "priceProtectScope": "0",
            }
        }

    def fetch_price(self, symbol: str):
        return self.mark_price

    def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"orderId": "order-1"}


def _exchange_with_positions(positions) -> BitunixExchange:
    exchange = BitunixExchange(ExchangeConfig(api_key="key", api_secret="secret"))
    exchange._client = _FakeBitunixClient(positions)  # noqa: SLF001
    return exchange


def _exchange_with_klines(rows) -> BitunixExchange:
    exchange = BitunixExchange(ExchangeConfig(api_key="key", api_secret="secret"))
    exchange._client = _FakeKlineClient(rows)  # noqa: SLF001
    return exchange


def _exchange_with_order_client(client: _FakeOrderClient) -> BitunixExchange:
    exchange = BitunixExchange(ExchangeConfig(api_key="key", api_secret="secret"))
    exchange._client = client  # noqa: SLF001
    return exchange


class BitunixExchangeTests(unittest.TestCase):
    def test_get_current_positions_propagates_unknown_exchange_state(self) -> None:
        exchange = _exchange_with_positions(RuntimeError("temporary API failure"))

        with self.assertRaisesRegex(RuntimeError, "failed to fetch current positions"):
            exchange.get_current_positions()

    def test_get_current_positions_returns_empty_only_for_confirmed_empty_response(
        self,
    ) -> None:
        exchange = _exchange_with_positions([])

        self.assertEqual(exchange.get_current_positions(), [])

    def test_get_current_positions_normalizes_bitunix_position_payload(self) -> None:
        exchange = _exchange_with_positions(
            [
                {
                    "symbol": "ETHUSDT",
                    "side": "SELL",
                    "qty": "0.25",
                    "avgOpenPrice": "2500.5",
                    "leverage": "10",
                    "marginMode": "ISOLATION",
                    "unrealizedPNL": "12.5",
                    "liqPrice": "3200",
                    "positionId": "pos-123",
                }
            ]
        )

        positions = exchange.get_current_positions()

        self.assertEqual(len(positions), 1)
        position = positions[0]
        self.assertEqual(position.symbol, "ETHUSDT")
        self.assertEqual(position.side, PositionSide.SHORT)
        self.assertEqual(position.size, 0.25)
        self.assertEqual(position.entry_price, 2500.5)
        self.assertEqual(position.leverage, 10)
        self.assertEqual(position.margin_mode, MarginMode.ISOLATED)
        self.assertEqual(position.unrealized_pnl, 12.5)
        self.assertEqual(position.liquidation_price, 3200)
        self.assertEqual(position.position_id, "pos-123")

    def test_get_klines_returns_rows_oldest_to_newest(self) -> None:
        exchange = _exchange_with_klines(
            [
                {
                    "time": 3_601_000,
                    "open": "103",
                    "high": "104",
                    "low": "102",
                    "close": "103.5",
                    "baseVol": "10",
                    "quoteVol": "1000",
                },
                {
                    "time": 1_000,
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100.5",
                    "baseVol": "8",
                    "quoteVol": "800",
                },
                {
                    "time": 7_201_000,
                    "open": "106",
                    "high": "107",
                    "low": "105",
                    "close": "106.5",
                    "baseVol": "12",
                    "quoteVol": "1200",
                },
            ]
        )

        rows = exchange.get_klines("ETHUSDT", "1h", limit=3)

        self.assertEqual([row[0] for row in rows], [1_000, 3_601_000, 7_201_000])
        self.assertEqual(
            [row[6] for row in rows], [3_600_999, 7_200_999, 10_800_999]
        )

    def test_open_limit_position_normalizes_attached_long_stop_loss(self) -> None:
        client = _FakeOrderClient(mark_price=100.0)
        exchange = _exchange_with_order_client(client)

        exchange.open_limit_position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=1.23456,
            price=99.987,
            leverage=2,
            margin_mode=MarginMode.ISOLATED,
            stop_loss=100.001,
        )

        self.assertEqual(len(client.orders), 1)
        order = client.orders[0]
        self.assertEqual(order["qty"], 1.234)
        self.assertEqual(order["price"], 99.98)
        self.assertEqual(order["sl_price"], 99.98)

    def test_open_limit_position_normalizes_attached_short_stop_loss(self) -> None:
        client = _FakeOrderClient(mark_price=100.0)
        exchange = _exchange_with_order_client(client)

        exchange.open_limit_position(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            quantity=1.23456,
            price=100.012,
            leverage=2,
            margin_mode=MarginMode.ISOLATED,
            stop_loss=100.001,
        )

        self.assertEqual(len(client.orders), 1)
        order = client.orders[0]
        self.assertEqual(order["price"], 100.02)
        self.assertEqual(order["sl_price"], 100.02)

    def test_open_market_position_normalizes_attached_stop_loss(self) -> None:
        client = _FakeOrderClient(mark_price=100.0)
        exchange = _exchange_with_order_client(client)

        exchange.open_market_position(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            quantity=0.12345,
            leverage=2,
            margin_mode=MarginMode.ISOLATED,
            stop_loss=105.121,
        )

        self.assertEqual(len(client.orders), 1)
        self.assertEqual(client.orders[0]["qty"], 0.123)
        self.assertEqual(client.orders[0]["sl_price"], 105.13)


if __name__ == "__main__":
    unittest.main()
