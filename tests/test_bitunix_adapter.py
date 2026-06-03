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


def _exchange_with_positions(positions) -> BitunixExchange:
    exchange = BitunixExchange(ExchangeConfig(api_key="key", api_secret="secret"))
    exchange._client = _FakeBitunixClient(positions)  # noqa: SLF001
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


if __name__ == "__main__":
    unittest.main()
