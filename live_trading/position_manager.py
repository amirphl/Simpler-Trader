"""Position management and persistence."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Optional

from signal_notifier import TelegramClient

from .exchange import Exchange, OrderResult
from .models import LiveTradingConfig, PositionRecord, TradingSignal, TradingState


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
                take_profit REAL NOT NULL,
                stop_loss REAL NOT NULL,
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
            
            self._log.info(f"Loaded state: {len(self._state.active_positions)} active positions, "
                          f"{len(self._state.disabled_symbols)} disabled symbols")
            
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
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            leverage=data["leverage"],
            margin_mode=MarginMode(data["margin_mode"]),
            take_profit=data["take_profit"],
            stop_loss=data["stop_loss"],
            exit_time=datetime.fromisoformat(data["exit_time"]) if data.get("exit_time") else None,
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
            
            if position_size < 1:  # Minimum position size # TODO: Read from config
                self._log.warning(f"Position size too small: {position_size}")
                return False
            
        except Exception as e:
            self._log.error(f"Failed to check balance: {e}")
            self._state.failed_trades += 1
            self._save_state()
            return False
        
        try:
            # Set leverage and margin mode
            self._exchange.set_leverage(symbol, signal.leverage)
            self._exchange.set_margin_mode(symbol, signal.margin_mode)
            
            # Open position
            self._log.info(
                f"Opening {signal.side.value} position for {symbol}: "
                f"size={position_size} USDT, leverage={signal.leverage}x"
            )
            
            order_result = self._exchange.open_market_position(
                symbol=symbol,
                side=signal.side,
                quantity=position_size,
                leverage=signal.leverage,
                margin_mode=signal.margin_mode,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
            )
            
            # Create position record
            position = PositionRecord(
                position_id=order_result.order_id,
                symbol=symbol,
                side=signal.side,
                entry_time=signal.timestamp,
                entry_price=order_result.price,
                quantity=order_result.quantity,
                leverage=signal.leverage,
                margin_mode=signal.margin_mode,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                status="OPEN",
                notes=signal.reason,
            )
            
            # Update state
            self._state.active_positions[symbol] = position
            self._state.disable_symbol(symbol, signal.timestamp, self._config.disable_symbol_hours)
            self._state.total_trades += 1
            
            # Save to database
            self._save_position_to_db(position)
            self._save_state()
            self._notify_trade_opened(signal, order_result, position_size)
            
            self._log.info(f"Position opened successfully: {symbol} {signal.side.value}")
            return True
            
        except Exception as e:
            self._log.error(f"Failed to execute signal for {symbol}: {e}", exc_info=True)
            self._state.failed_trades += 1
            self._save_state()
            return False
    
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
                datetime.now().isoformat(),
            ),
        )
        
        conn.commit()
        conn.close()
    
    def update_positions(self, current_time: datetime) -> None:
        """Update position statuses by checking exchange."""
        if not self._state.active_positions:
            return
        
        self._log.info(f"Updating {len(self._state.active_positions)} active positions")
        
        closed_symbols = []
        
        for symbol, position in self._state.active_positions.items():
            try:
                # Check if position still exists on exchange
                exchange_position = self._exchange.get_position(symbol)
                
                if exchange_position is None:
                    # Position closed (likely hit TP or SL)
                    self._log.info(f"Position closed for {symbol}")
                    position.status = "CLOSED"
                    position.exit_time = current_time
                    
                    # Try to get exit details from recent trades (not implemented in base Exchange)
                    # For now, we mark as closed without exact exit price
                    
                    self._save_position_to_db(position)
                    closed_symbols.append(symbol)
                    self._state.successful_trades += 1
                    
            except Exception as e:
                self._log.error(f"Error updating position for {symbol}: {e}")
        
        # Remove closed positions from active state
        for symbol in closed_symbols:
            del self._state.active_positions[symbol]
        
        if closed_symbols:
            self._save_state()
    
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
        try:
            lines = [
                "Live trade executed",
                f"Symbol: {signal.symbol}",
                f"Side: {signal.side.value}",
                f"Entry price: {order_result.price:.6f}",
                f"Requested size (USDT): {requested_size:.2f}",
                f"Filled quantity: {order_result.quantity:.6f}",
                f"Leverage: {signal.leverage}x ({signal.margin_mode.value})",
                f"Take profit: {signal.take_profit:.6f}",
                f"Stop loss: {signal.stop_loss:.6f}",
                f"Reason: {signal.reason}",
            ]
            self._telegram.send_message("\n".join(lines))
            self._log.info("Sent Telegram notification for %s", signal.symbol)
        except Exception as exc:  # pragma: no cover - best-effort notification
            self._log.warning("Failed to send Telegram notification: %s", exc)
