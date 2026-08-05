from __future__ import annotations

import unittest
from unittest.mock import patch

from live_trading.exchange import ExchangeConfig, MarginMode, Position, PositionSide
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


class _PagingKlineClient:
    def __init__(self, rows):
        self._rows = sorted(rows, key=lambda row: int(row["time"]))
        self.calls: list[dict] = []

    def get_kline_history(self, **kwargs):
        self.calls.append(kwargs)
        end_time = kwargs.get("end_time")
        eligible = [
            row
            for row in self._rows
            if end_time is None or int(row["time"]) <= end_time
        ]
        return eligible[-kwargs["limit"] :]


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


class _FakePositionOrderClient(_FakeOrderClient):
    def __init__(self, *, position_mode: str, positions: list[dict]):
        super().__init__()
        self.position_mode = position_mode
        self.positions = positions

    def get_single_account(self, margin_coin: str):
        del margin_coin
        return {"positionMode": self.position_mode}

    def get_pending_positions(self):
        return self.positions


class _FakeStopClient(_FakeOrderClient):
    def __init__(self):
        super().__init__()
        self.stop_orders: list[dict[str, str]] = []
        self.place_stop_calls: list[dict] = []
        self.modify_stop_calls: list[dict] = []

    def get_pending_tpsl_orders(self, **kwargs):
        return list(self.stop_orders)

    def place_position_tpsl_order(self, **kwargs):
        self.place_stop_calls.append(kwargs)
        self.stop_orders = [
            {
                "positionId": kwargs["position_id"],
                "slPrice": str(kwargs["sl_price"]),
            }
        ]
        return {"orderId": "stop-1"}

    def modify_position_tpsl_order(self, **kwargs):
        self.modify_stop_calls.append(kwargs)
        self.stop_orders = [
            {
                "positionId": kwargs["position_id"],
                "slPrice": str(kwargs["sl_price"]),
            }
        ]
        return {"orderId": "stop-1"}


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
    def test_kline_request_slots_are_paced_across_concurrent_fetches(self) -> None:
        exchange = _exchange_with_klines([])
        exchange._next_kline_request_at = 10.5  # noqa: SLF001

        with (
            patch(
                "live_trading.exchanges.bitunix.adapter.time.monotonic",
                return_value=10.0,
            ),
            patch("live_trading.exchanges.bitunix.adapter.time.sleep") as sleep,
        ):
            exchange._wait_for_kline_request_slot()  # noqa: SLF001

        sleep.assert_called_once_with(0.5)
        self.assertEqual(exchange._next_kline_request_at, 11.0)  # noqa: SLF001

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
                    "side": "SHORT",
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

    def test_get_current_positions_rejects_unknown_position_side(self) -> None:
        exchange = _exchange_with_positions(
            [
                {
                    "symbol": "ETHUSDT",
                    "side": "SIDEWAYS",
                    "qty": "1",
                    "marginMode": "ISOLATION",
                }
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "Invalid Bitunix position payload"):
            exchange.get_current_positions()

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

    def test_get_klines_paginates_history_beyond_bitunix_page_limit(self) -> None:
        client = _PagingKlineClient(
            [
                {
                    "time": (index + 1) * 60_000,
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100.5",
                    "baseVol": "8",
                }
                for index in range(450)
            ]
        )
        exchange = BitunixExchange(ExchangeConfig(api_key="key", api_secret="secret"))
        exchange._client = client  # noqa: SLF001

        rows = exchange.get_klines("ETHUSDT", "1m", limit=450)

        self.assertEqual(len(rows), 450)
        self.assertEqual([call["limit"] for call in client.calls], [200, 200, 50])
        self.assertEqual(rows[0][0], 60_000)
        self.assertEqual(rows[-1][0], 27_000_000)

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
        self.assertIsNone(order["trade_side"])

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
        # Short stops round down to a tick so normalization never widens risk.
        self.assertEqual(client.orders[0]["sl_price"], 105.12)

    def test_stop_is_normalized_conservatively_and_read_back(self) -> None:
        client = _FakeStopClient()
        exchange = _exchange_with_order_client(client)
        position = Position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
            leverage=2.0,
            margin_mode=MarginMode.ISOLATED,
            unrealized_pnl=0.0,
            position_id="pos-1",
        )

        actual = exchange.ensure_position_stop_loss(position, 97.501)

        self.assertEqual(actual, 97.51)
        self.assertEqual(client.place_stop_calls[0]["sl_price"], 97.51)

    def test_open_limit_position_forwards_stable_client_id(self) -> None:
        client = _FakeOrderClient()
        exchange = _exchange_with_order_client(client)

        exchange.open_limit_position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=1.0,
            price=99.0,
            leverage=2,
            margin_mode=MarginMode.ISOLATED,
            client_id="emaavwap-stable-id",
        )

        self.assertEqual(client.orders[0]["client_id"], "emaavwap-stable-id")

    def test_ema_avwap_rejects_hedge_mode_before_trading(self) -> None:
        client = _FakePositionOrderClient(position_mode="HEDGE", positions=[])
        exchange = _exchange_with_order_client(client)

        with self.assertRaisesRegex(RuntimeError, "requires Bitunix ONE_WAY"):
            exchange.validate_ema_avwap_execution()

    def test_close_position_uses_documented_hedge_close_side(self) -> None:
        client = _FakePositionOrderClient(
            position_mode="HEDGE",
            positions=[
                {
                    "symbol": "ETHUSDT",
                    "side": "LONG",
                    "qty": "1",
                    "avgOpenPrice": "100",
                    "leverage": "2",
                    "marginMode": "ISOLATION",
                    "unrealizedPNL": "0",
                    "positionId": "pos-long",
                }
            ],
        )
        exchange = _exchange_with_order_client(client)

        exchange.close_position("ETHUSDT", side=PositionSide.LONG)

        order = client.orders[0]
        self.assertEqual(order["side"], "BUY")
        self.assertEqual(order["trade_side"], "CLOSE")
        self.assertEqual(order["position_id"], "pos-long")

    def test_close_position_uses_opposite_side_in_one_way_mode(self) -> None:
        client = _FakePositionOrderClient(
            position_mode="ONE_WAY",
            positions=[
                {
                    "symbol": "ETHUSDT",
                    "side": "LONG",
                    "qty": "1",
                    "avgOpenPrice": "100",
                    "leverage": "2",
                    "marginMode": "ISOLATION",
                    "unrealizedPNL": "0",
                    "positionId": "pos-one-way",
                }
            ],
        )
        exchange = _exchange_with_order_client(client)

        exchange.close_position("ETHUSDT", side=PositionSide.LONG)

        order = client.orders[0]
        self.assertEqual(order["side"], "SELL")
        self.assertIsNone(order["trade_side"])
        self.assertIsNone(order["position_id"])


if __name__ == "__main__":
    unittest.main()
