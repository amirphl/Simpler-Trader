from __future__ import annotations

import unittest
from datetime import datetime, timezone

from live_trading.exchange import MarginMode, PositionSide
from live_trading.models import PositionRecord
from live_trading.pinbar_magic_coordinator_v3 import (
    PinBarMagicCoordinatorV3,
    PinBarMagicCoordinatorV3Config,
)


class _CloseFailingExchange:
    def close_position(self, symbol, side=None):
        raise RuntimeError("close rejected")


class PinBarMagicCoordinatorV3Tests(unittest.TestCase):
    def test_close_position_failure_keeps_position_tracked(self) -> None:
        coordinator = PinBarMagicCoordinatorV3(
            exchange=_CloseFailingExchange(),
            config=PinBarMagicCoordinatorV3Config(symbols=("ETHUSDT",)),
        )
        position = PositionRecord(
            position_id="pos-1",
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            entry_price=2500.0,
            quantity=0.1,
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
            take_profit=None,
            stop_loss=2450.0,
            strategy="pinbar_magic_v3",
        )
        coordinator._state.active_positions[position.symbol] = position  # noqa: SLF001

        coordinator._close_position(  # noqa: SLF001
            position,
            datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            "test close",
        )

        self.assertIs(
            coordinator._state.active_positions["ETHUSDT"],  # noqa: SLF001
            position,
        )
        self.assertEqual(position.status, "OPEN")
        self.assertIn("close failed: test close", position.notes)


if __name__ == "__main__":
    unittest.main()
