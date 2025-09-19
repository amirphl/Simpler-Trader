"""Position management and persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from signal_notifier import TelegramClient

from .exchange import Exchange, OrderResult, Position, PositionSide
from .models import (
    LiveTradingConfig,
    PositionRecord,
    TradingSignal,
    TradingState,
)


class PositionManager:
    """Manages positions, state persistence, and order execution."""

    def __init__(
        self,
        config: LiveTradingConfig,
        exchange: Exchange,
        logger: logging.Logger | None = None,
        telegram_client: Optional[TelegramClient] = None,
    ) -> None:
        self._config = config
        self._exchange = exchange
        self._log = logger or logging.getLogger(__name__)
        self._state = TradingState()
        self._telegram = telegram_client
        self._last_trailing_check_ts: float = 0.0
        # Throttle trailing-stop checks to avoid hammering the exchange.
        self._trailing_check_interval_seconds = 15.0

        # Ensure directories exist
        self._config.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._config.positions_db.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Load state
        self._load_state()

    def _init_database(self) -> None:
        """Initialize SQLite database for position tracking."""
        conn = sqlite3.connect(str(self._config.positions_db))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                leverage INTEGER NOT NULL,
                margin_mode TEXT NOT NULL,
                take_profit REAL,
                stop_loss REAL,
                exit_time TEXT,
                exit_price REAL,
                pnl REAL,
                status TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol ON positions(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON positions(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entry_time ON positions(entry_time)
        """)

        conn.commit()
        conn.close()

        self._log.info(f"Position database initialized: {self._config.positions_db}")

    def _load_state(self) -> None:
        """Load trading state from file."""
        if not self._config.state_file.exists():
            self._log.info("No existing state file found, starting fresh")
            return

        try:
            with open(self._config.state_file, "r") as f:
                data = json.load(f)

            # Load disabled symbols
            disabled = data.get("disabled_symbols", {})
            self._state.disabled_symbols = {
                symbol: datetime.fromisoformat(dt_str)
                for symbol, dt_str in disabled.items()
            }

            # Load active positions
            positions = data.get("active_positions", {})
            self._state.active_positions = {
                symbol: self._position_from_dict(pos_dict)
                for symbol, pos_dict in positions.items()
            }

            # Load stats
            self._state.last_execution_time = (
                datetime.fromisoformat(data["last_execution_time"])
                if data.get("last_execution_time")
                else None
            )
            self._state.total_trades = data.get("total_trades", 0)
            self._state.successful_trades = data.get("successful_trades", 0)
            self._state.failed_trades = data.get("failed_trades", 0)

            self._log.info(
                "Loaded state: %s active positions, %s disabled symbols",
                len(self._state.active_positions),
                len(self._state.disabled_symbols),
            )

        except Exception as e:
            self._log.error(f"Failed to load state: {e}")
            self._state = TradingState()

    def _save_state(self) -> None:
        """Save trading state to file."""
        try:
            data = {
                "disabled_symbols": {
                    symbol: dt.isoformat()
                    for symbol, dt in self._state.disabled_symbols.items()
                },
                "active_positions": {
                    symbol: self._position_to_dict(pos)
                    for symbol, pos in self._state.active_positions.items()
                },
                "last_execution_time": (
                    self._state.last_execution_time.isoformat()
                    if self._state.last_execution_time
                    else None
                ),
                "total_trades": self._state.total_trades,
                "successful_trades": self._state.successful_trades,
                "failed_trades": self._state.failed_trades,
            }

            with open(self._config.state_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self._log.error(f"Failed to save state: {e}")

    def _position_to_dict(self, pos: PositionRecord) -> Dict:
        """Convert PositionRecord to dictionary."""
        return {
            "position_id": pos.position_id,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry_time": pos.entry_time.isoformat(),
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "leverage": pos.leverage,
            "margin_mode": pos.margin_mode.value,
            "take_profit": pos.take_profit,
            "stop_loss": pos.stop_loss,
            "risk_amount": pos.risk_amount,
            "trailing_active": pos.trailing_active,
            "trailing_stop": pos.trailing_stop,
            "extreme_since_activation": pos.extreme_since_activation,
            "strategy": pos.strategy,
            "exit_time": pos.exit_time.isoformat() if pos.exit_time else None,
            "exit_price": pos.exit_price,
            "pnl": pos.pnl,
            "status": pos.status,
            "notes": pos.notes,
        }

    def _position_from_dict(self, data: Dict) -> PositionRecord:
        """Convert dictionary to PositionRecord."""
        from .exchange import MarginMode, PositionSide

        return PositionRecord(
            position_id=data["position_id"],
            symbol=data["symbol"],
            side=PositionSide(data["side"]),
            entry_time=self._ensure_aware(datetime.fromisoformat(data["entry_time"])),
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            leverage=data["leverage"],
            margin_mode=MarginMode(data["margin_mode"]),
            take_profit=data["take_profit"],
            stop_loss=data["stop_loss"],
            risk_amount=data.get("risk_amount"),
            trailing_active=bool(data.get("trailing_active", False)),
            trailing_stop=data.get("trailing_stop"),
            extreme_since_activation=data.get("extreme_since_activation"),
            strategy=str(data.get("strategy", "heiken_ashi")),
            exit_time=self._ensure_aware(datetime.fromisoformat(data["exit_time"]))
            if data.get("exit_time")
            else None,
            exit_price=data.get("exit_price"),
            pnl=data.get("pnl"),
            status=data["status"],
            notes=data.get("notes", ""),
        )

    def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal by opening a position.

        Args:
            signal: Trading signal to execute

        Returns:
            True if position opened successfully
        """
        if (
            signal.strategy == "pinbar_magic_v2"
            or signal.metadata.get("strategy") == "pinbar_magic_v2"
        ):
            self._log.warning(
                "PinBarMagic signal received by PositionManager. "
                "Use live_trading.pinbar_magic_coordinator for PinBar Magic v2 execution."
            )
            return False

        symbol = signal.symbol

        # Check max concurrent positions
        if len(self._state.active_positions) >= self._config.max_concurrent_positions:
            self._log.warning(
                f"Max concurrent positions reached ({self._config.max_concurrent_positions}), "
                f"skipping {symbol}"
            )
            return False

        # Check balance
        try:
            balance = self._exchange.get_account_balance()
            position_size = self._config.position_size_usdt

            # Validate position size
            max_position_size = balance * (self._config.max_position_size_pct / 100)
            if position_size > max_position_size:
                self._log.warning(
                    f"Position size {position_size} exceeds max ({max_position_size}), "
                    f"adjusting to max"
                )
                position_size = max_position_size

            # TODO: Read from config
            if position_size < 10:  # Minimum position size
                self._log.warning(f"Position size too small: {position_size}")
                return False

        except Exception as e:
            self._log.error(f"Failed to check balance: {e}")
            self._state.failed_trades += 1
            self._save_state()
            return False

        try:
            # Set leverage and margin mode
            # TODO:
            self._exchange.set_leverage(symbol, signal.leverage)
            # TODO:
            self._exchange.set_margin_mode(symbol, signal.margin_mode)

            # Get latest price for limit entry
            last_price = self._get_last_price(symbol)
            if last_price is None or last_price <= 0:
                self._log.error("Could not fetch valid last price for %s", symbol)
                self._state.failed_trades += 1
                self._save_state()
                return False

            # Open position
            self._log.info(
                f"Opening {signal.side.value} position for {symbol}: "
                f"size={position_size} USDT, leverage={signal.leverage}x"
            )

            # Convert notional (USDT) to base quantity using latest price
            base_quantity = position_size / last_price if last_price > 0 else 0.0
            if base_quantity <= 0:
                self._log.error("Invalid base quantity calculated for %s", symbol)
                self._state.failed_trades += 1
                self._save_state()
                return False

            order_result = self._exchange.open_limit_position(
                symbol=symbol,
                side=signal.side,
                quantity=base_quantity,
                price=last_price,
                leverage=signal.leverage,
                margin_mode=signal.margin_mode,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
            )

            filled_price = order_result.price if order_result.price > 0 else last_price

            # Create position record
            position = PositionRecord(
                position_id=order_result.order_id,
                symbol=symbol,
                side=signal.side,
                entry_time=signal.timestamp,
                entry_price=filled_price,
                quantity=order_result.quantity,
                leverage=signal.leverage,
                margin_mode=signal.margin_mode,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                strategy=signal.strategy,
                status="OPEN",
                notes=signal.reason,
            )

            # Update state
            self._state.active_positions[symbol] = position
            self._state.disable_symbol(
                symbol, signal.timestamp, self._config.disable_symbol_hours
            )
            self._state.total_trades += 1

            # Save to database
            self._save_position_to_db(position)
            self._save_state()
            self._notify_trade_opened(signal, order_result, position_size)

            self._log.info(
                f"Position opened successfully: {symbol} {signal.side.value}"
            )
            return True

        except Exception as e:
            self._log.error(
                f"Failed to execute signal for {symbol}: {e}", exc_info=True
            )
            self._state.failed_trades += 1
            self._save_state()
            return False

    def _get_last_price(self, symbol: str) -> Optional[float]:
        """Fetch last price using exchange data."""
        try:
            if hasattr(self._exchange, "fetch_price"):
                price = self._exchange.fetch_price(symbol)  # type: ignore[attr-defined]
                if price is not None:
                    return float(price)
            tickers = self._exchange.get_24h_tickers()
            for ticker in tickers or []:
                if str(ticker.get("symbol")) == symbol:
                    last = (
                        ticker.get("lastPrice")
                        or ticker.get("last")
                        or ticker.get("markPrice")
                    )
                    if last is not None:
                        return float(last)
        except Exception as exc:
            self._log.warning("Failed to fetch last price for %s: %s", symbol, exc)
        return None

    def _save_position_to_db(self, position: PositionRecord) -> None:
        """Save position record to database."""
        conn = sqlite3.connect(str(self._config.positions_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO positions (
                position_id, symbol, side, entry_time, entry_price,
                quantity, leverage, margin_mode, take_profit, stop_loss,
                exit_time, exit_price, pnl, status, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.position_id,
                position.symbol,
                position.side.value,
                position.entry_time.isoformat(),
                position.entry_price,
                position.quantity,
                position.leverage,
                position.margin_mode.value,
                position.take_profit,
                position.stop_loss,
                position.exit_time.isoformat() if position.exit_time else None,
                position.exit_price,
                position.pnl,
                position.status,
                position.notes,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def update_positions(
        self, current_time: datetime, *, enable_generic_trailing: bool = True
    ) -> None:
        """Update position statuses by checking exchange."""
        now = self._ensure_aware(current_time)
        if not self._state.active_positions:
            return

        self._log.info(
            "Updating positions: %s active", len(self._state.active_positions)
        )

        closed_symbols: list[str] = []
        state_changed = False
        try:
            exchange_positions = self._exchange.get_current_positions()
        except Exception as exc:
            self._log.error("Failed to fetch positions from exchange: %s", exc)
            return

        exchange_positions_by_symbol = {pos.symbol: pos for pos in exchange_positions}

        trailing_allowed = (time.time() - self._last_trailing_check_ts) >= (
            self._trailing_check_interval_seconds
        )

        for symbol, position in list(self._state.active_positions.items()):
            try:
                exchange_position = exchange_positions_by_symbol.get(symbol)
                if exchange_position is None:
                    # Position closed (likely hit TP or SL)
                    self._log.info(f"Position closed for {symbol}")
                    position.status = "CLOSED"
                    position.exit_time = now

                    # Try to get exit details from recent trades (not implemented in base Exchange)
                    # For now, we mark as closed without exact exit price

                    self._save_position_to_db(position)
                    closed_symbols.append(symbol)
                    self._state.successful_trades += 1
                    state_changed = True
                    continue

                if trailing_allowed and enable_generic_trailing:
                    self._log.info(
                        f"Checking trailing stop for {symbol}",
                    )
                    self._maybe_update_trailing_stop(position, exchange_position)

            except Exception as e:
                self._log.error(f"Error updating position for {symbol}: {e}")

        # Remove closed positions from active state
        for symbol in closed_symbols:
            current = self._state.active_positions.get(symbol)
            if current is None:
                continue
            if current.status != "OPEN":
                del self._state.active_positions[symbol]
            # Immediately re-enable the symbol since the position is closed
            # TODO:
            # if symbol in self._state.disabled_symbols:
            #     del self._state.disabled_symbols[symbol]

        if closed_symbols:
            state_changed = True
        if state_changed:
            self._save_state()
        if trailing_allowed:
            self._last_trailing_check_ts = time.time()

    def apply_pinbar_position_management(
        self, snapshots: Dict[str, object], current_time: datetime
    ) -> None:
        """Compatibility no-op. PinBar logic moved to pinbar_magic_coordinator."""
        self._log.warning(
            "apply_pinbar_position_management() is deprecated in PositionManager; "
            "use pinbar_magic_coordinator instead."
        )

    def activate_due_pinbar_entries(self, current_time: datetime) -> None:
        """Compatibility no-op. PinBar logic moved to pinbar_magic_coordinator."""
        self._log.warning(
            "activate_due_pinbar_entries() is deprecated in PositionManager; "
            "use pinbar_magic_coordinator instead."
        )

    def cleanup_state(self, current_time: datetime) -> None:
        """Cleanup expired disabled symbols."""
        self._state.cleanup_disabled_symbols(current_time)
        self._save_state()

    def get_state(self) -> TradingState:
        """Get current trading state."""
        return self._state

    def update_execution_time(self, timestamp: datetime) -> None:
        """Update last execution time."""
        self._state.last_execution_time = timestamp
        self._save_state()

    def _notify_trade_opened(
        self,
        signal: TradingSignal,
        order_result: OrderResult,
        requested_size: float,
    ) -> None:
        """Send Telegram notification for an opened trade."""
        if not self._telegram:
            return

        def _fmt(value: Optional[float], digits: int) -> str:
            if value is None:
                return "N/A"
            return f"{value:.{digits}f}"

        try:
            lines = [
                "Live trade executed",
                f"Symbol: {signal.symbol}",
                f"Side: {signal.side.value}",
                f"Entry price: {_fmt(order_result.price if order_result.price > 0 else signal.entry_price, 6)}",
                f"Requested size (USDT): {_fmt(requested_size, 2)}",
                f"Filled quantity: {_fmt(order_result.quantity, 6)}",
                f"Leverage: {signal.leverage}x ({signal.margin_mode.value})",
                f"Take profit: {_fmt(signal.take_profit, 6)}",
                f"Stop loss: {_fmt(signal.stop_loss, 6)}",
                f"Reason: {signal.reason}",
            ]
            self._telegram.send_message("\n".join(lines))
            self._log.info("Sent Telegram notification for %s", signal.symbol)
        except Exception as exc:  # pragma: no cover - best-effort notification
            self._log.warning("Failed to send Telegram notification: %s", exc)

    def _maybe_update_trailing_stop(
        self, position: PositionRecord, exchange_position: Position
    ) -> None:
        """Adjust stop loss based on realized PnL to trail profits."""
        entry_price = exchange_position.entry_price or position.entry_price
        if entry_price <= 0:
            self._log.debug(
                "Skipping trailing stop for %s due to missing entry price",
                position.symbol,
            )
            return

        current_price = self._get_last_price(position.symbol)
        if current_price is None or current_price <= 0:
            self._log.debug(
                "Skipping trailing stop for %s due to missing current price",
                position.symbol,
            )
            return

        pnl_pct = self._calculate_pnl_pct(entry_price, current_price, position.side)
        if pnl_pct < 1.0:
            # Activation threshold not reached yet.
            return

        desired_stop_pct = max(0.0, pnl_pct - 0.5)
        existing_stop_pct = self._stop_loss_pct(
            entry_price, position.stop_loss, position.side
        )
        # Never move the stop backward.
        if desired_stop_pct <= existing_stop_pct:
            return

        stop_price = self._stop_price_from_pct(
            entry_price, desired_stop_pct, position.side
        )
        if stop_price is None:
            return

        self._log.info(
            "Trailing stop for %s: pnl=%.2f%% -> updating SL to %.2f%% (price=%.6f)",
            position.symbol,
            pnl_pct,
            desired_stop_pct,
            stop_price,
        )

        if not self._update_stop_loss_on_exchange(exchange_position, stop_price):
            return

        position.stop_loss = stop_price
        position.entry_price = entry_price
        self._save_position_to_db(position)
        self._save_state()

    def _calculate_pnl_pct(
        self, entry_price: float, current_price: float, side: PositionSide
    ) -> float:
        """Return PnL percentage based on side and prices."""
        if entry_price <= 0:
            return 0.0
        if side == PositionSide.LONG:
            return ((current_price - entry_price) / entry_price) * 100.0
        return ((entry_price - current_price) / entry_price) * 100.0

    def _stop_loss_pct(
        self, entry_price: float, stop_price: Optional[float], side: PositionSide
    ) -> float:
        """Return current stop-loss distance from entry in percent (0 if not set)."""
        if entry_price <= 0 or not stop_price:
            return 0.0
        if side == PositionSide.LONG:
            return ((stop_price - entry_price) / entry_price) * 100.0
        return ((entry_price - stop_price) / entry_price) * 100.0

    def _stop_price_from_pct(
        self, entry_price: float, stop_pct: float, side: PositionSide
    ) -> Optional[float]:
        """Convert desired stop distance (percent from entry) into a price."""
        if entry_price <= 0:
            return None
        if side == PositionSide.LONG:
            return entry_price * (1 + stop_pct / 100.0)
        return entry_price * (1 - stop_pct / 100.0)

    def _update_stop_loss_on_exchange(
        self, exchange_position: Position, stop_price: float
    ) -> bool:
        """Call exchange adapter to update stop loss, with mock fallback."""
        updater = getattr(self._exchange, "update_stop_loss", None)
        if callable(updater):
            try:
                return bool(updater(exchange_position, stop_price))
            except Exception as exc:
                self._log.warning(
                    "Failed to update stop loss for %s via exchange: %s",
                    exchange_position.symbol,
                    exc,
                )
                return False

        # Mock path when exchange does not yet expose stop-loss updates.
        self._log.info(
            "Simulating stop loss update for %s at %.6f (exchange method missing)",
            exchange_position.symbol,
            stop_price,
        )
        return True

    @staticmethod
    def _ensure_aware(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment
