"""State and position persistence for EMA + AVWAP live trading."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Mapping

from ..exchange import MarginMode, PositionSide
from ..models import PendingEntryRecord, PositionRecord, TradingState
from ._mixin_typing import EmaAvwapMixinTyping
from .state import (
    _AvwapSnapshot,
    _EntryCandidate,
    _PendingEntryMeta,
    _PositionRuntime,
    _SetupState,
    _SizingDecision,
)


class EmaAvwapPersistenceMixin(EmaAvwapMixinTyping):
    def _init_persistence(self) -> None:
        self._cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._cfg.positions_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_positions_db()
        self._load_state()

    def _init_positions_db(self) -> None:
        conn = sqlite3.connect(str(self._cfg.positions_db))
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
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
                    risk_amount REAL,
                    strategy TEXT NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    pnl REAL,
                    status TEXT NOT NULL,
                    notes TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ema_avwap_positions_symbol "
                "ON positions(symbol)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ema_avwap_positions_status "
                "ON positions(status)"
            )
            conn.commit()
        finally:
            conn.close()

    def _load_state(self) -> None:
        if not self._cfg.state_file.exists():
            self._log.info("EmaAvwapPullback: no state file found, starting fresh")
            return
        try:
            with self._cfg.state_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._state = TradingState(
                disabled_symbols={
                    symbol: self._parse_dt(value)
                    for symbol, value in data.get("disabled_symbols", {}).items()
                },
                active_positions={
                    symbol: self._position_from_dict(value)
                    for symbol, value in data.get("active_positions", {}).items()
                },
                pending_entries={
                    key: self._pending_from_dict(value)
                    for key, value in data.get("pending_entries", {}).items()
                },
                last_pinbar_signal_times={
                    key: self._parse_dt(value)
                    for key, value in data.get("last_signal_times", {}).items()
                },
                last_execution_time=self._parse_dt_or_none(
                    data.get("last_execution_time")
                ),
                total_trades=int(data.get("total_trades", 0)),
                successful_trades=int(data.get("successful_trades", 0)),
                failed_trades=int(data.get("failed_trades", 0)),
            )
            self._active_setups = {
                self._setup_key(item["symbol"], item["direction"]): self._setup_from_dict(
                    item
                )
                for item in data.get("active_setups", [])
            }
            self._last_price_by_setup_key = {
                self._setup_key(item["symbol"], item["direction"]): float(
                    item["price"]
                )
                for item in data.get("last_price_by_setup", [])
            }
            self._pending_meta_by_key = {
                key: self._pending_meta_from_dict(value)
                for key, value in data.get("pending_meta", {}).items()
                if value
            }
            self._position_runtime_by_symbol = {
                symbol: self._runtime_from_dict(value)
                for symbol, value in data.get("position_runtime", {}).items()
                if value
            }
            self._position_miss_count_by_symbol = {
                symbol: int(count)
                for symbol, count in data.get("position_miss_counts", {}).items()
            }
            self._log.info(
                "EmaAvwapPullback: loaded state active=%d pending=%d setups=%d",
                len(self._state.active_positions),
                len(self._state.pending_entries),
                len(self._active_setups),
            )
        except Exception as exc:
            self._log.error("EmaAvwapPullback: failed to load state: %s", exc)
            self._state = TradingState()
            self._active_setups = {}
            self._last_price_by_setup_key = {}
            self._pending_meta_by_key = {}
            self._position_runtime_by_symbol = {}
            self._position_miss_count_by_symbol = {}

    def _save_state(self) -> None:
        try:
            data = {
                "disabled_symbols": {
                    symbol: value.isoformat()
                    for symbol, value in self._state.disabled_symbols.items()
                },
                "active_positions": {
                    symbol: self._position_to_dict(position)
                    for symbol, position in self._state.active_positions.items()
                },
                "pending_entries": {
                    key: self._pending_to_dict(pending)
                    for key, pending in self._state.pending_entries.items()
                },
                "last_signal_times": {
                    key: value.isoformat()
                    for key, value in self._state.last_pinbar_signal_times.items()
                },
                "last_execution_time": (
                    self._state.last_execution_time.isoformat()
                    if self._state.last_execution_time
                    else None
                ),
                "total_trades": self._state.total_trades,
                "successful_trades": self._state.successful_trades,
                "failed_trades": self._state.failed_trades,
                "active_setups": [
                    self._setup_to_dict(setup)
                    for setup in self._active_setups.values()
                ],
                "last_price_by_setup": [
                    {"symbol": key[0], "direction": key[1], "price": price}
                    for key, price in self._last_price_by_setup_key.items()
                ],
                "pending_meta": {
                    key: self._pending_meta_to_dict(meta)
                    for key, meta in self._pending_meta_by_key.items()
                },
                "position_runtime": {
                    symbol: self._runtime_to_dict(runtime)
                    for symbol, runtime in self._position_runtime_by_symbol.items()
                },
                "position_miss_counts": self._position_miss_count_by_symbol,
            }
            with self._cfg.state_file.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception as exc:
            self._log.error("EmaAvwapPullback: failed to save state: %s", exc)

    def _save_position_to_db(self, position: PositionRecord) -> None:
        try:
            conn = sqlite3.connect(str(self._cfg.positions_db))
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (
                        position_id, symbol, side, entry_time, entry_price, quantity,
                        leverage, margin_mode, take_profit, stop_loss, risk_amount,
                        strategy, exit_time, exit_price, pnl, status, notes, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        position.risk_amount,
                        position.strategy,
                        position.exit_time.isoformat() if position.exit_time else None,
                        position.exit_price,
                        position.pnl,
                        position.status,
                        position.notes,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            self._log.error(
                "EmaAvwapPullback: failed to persist position %s: %s",
                position.symbol,
                exc,
            )

    def _position_to_dict(self, position: PositionRecord) -> Dict[str, Any]:
        return {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "side": position.side.value,
            "entry_time": position.entry_time.isoformat(),
            "entry_price": position.entry_price,
            "quantity": position.quantity,
            "leverage": position.leverage,
            "margin_mode": position.margin_mode.value,
            "take_profit": position.take_profit,
            "stop_loss": position.stop_loss,
            "risk_amount": position.risk_amount,
            "trailing_active": position.trailing_active,
            "trailing_stop": position.trailing_stop,
            "extreme_since_activation": position.extreme_since_activation,
            "strategy": position.strategy,
            "exit_time": position.exit_time.isoformat() if position.exit_time else None,
            "exit_price": position.exit_price,
            "pnl": position.pnl,
            "status": position.status,
            "notes": position.notes,
        }

    def _position_from_dict(self, data: Mapping[str, Any]) -> PositionRecord:
        return PositionRecord(
            position_id=str(data["position_id"]),
            symbol=str(data["symbol"]),
            side=PositionSide(str(data["side"])),
            entry_time=self._parse_dt(str(data["entry_time"])),
            entry_price=float(data["entry_price"]),
            quantity=float(data["quantity"]),
            leverage=int(data["leverage"]),
            margin_mode=MarginMode(str(data["margin_mode"])),
            take_profit=data.get("take_profit"),
            stop_loss=data.get("stop_loss"),
            risk_amount=data.get("risk_amount"),
            trailing_active=bool(data.get("trailing_active", False)),
            trailing_stop=data.get("trailing_stop"),
            extreme_since_activation=data.get("extreme_since_activation"),
            strategy=str(data.get("strategy", "ema_avwap_pullback")),
            exit_time=self._parse_dt_or_none(data.get("exit_time")),
            exit_price=data.get("exit_price"),
            pnl=data.get("pnl"),
            status=str(data.get("status", "OPEN")),
            notes=str(data.get("notes", "")),
        )

    def _pending_to_dict(self, pending: PendingEntryRecord) -> Dict[str, Any]:
        return {
            "order_key": pending.order_key,
            "symbol": pending.symbol,
            "side": pending.side.value,
            "entry_price": pending.entry_price,
            "quantity": pending.quantity,
            "leverage": pending.leverage,
            "margin_mode": pending.margin_mode.value,
            "risk_amount": pending.risk_amount,
            "stop_for_risk": pending.stop_for_risk,
            "created_time": pending.created_time.isoformat(),
            "signal_time": pending.signal_time.isoformat(),
            "activate_time": pending.activate_time.isoformat(),
            "order_id": pending.order_id,
            "status": pending.status,
            "notes": pending.notes,
        }

    def _pending_from_dict(self, data: Mapping[str, Any]) -> PendingEntryRecord:
        return PendingEntryRecord(
            order_key=str(data["order_key"]),
            symbol=str(data["symbol"]),
            side=PositionSide(str(data["side"])),
            entry_price=float(data["entry_price"]),
            quantity=float(data["quantity"]),
            leverage=int(data["leverage"]),
            margin_mode=MarginMode(str(data["margin_mode"])),
            risk_amount=float(data["risk_amount"]),
            stop_for_risk=float(data["stop_for_risk"]),
            created_time=self._parse_dt(str(data["created_time"])),
            signal_time=self._parse_dt(str(data["signal_time"])),
            activate_time=self._parse_dt(str(data["activate_time"])),
            order_id=data.get("order_id"),
            status=str(data.get("status", "PENDING")),
            notes=str(data.get("notes", "")),
        )

    def _setup_to_dict(self, setup: _SetupState) -> Dict[str, Any]:
        return {
            "symbol": setup.symbol,
            "direction": setup.direction,
            "anchor_time": setup.anchor_time.isoformat(),
            "detected_time": setup.detected_time.isoformat(),
            "consecutive_count": setup.consecutive_count,
            "detected_avwap": self._avwap_to_dict(setup.detected_avwap)
            if setup.detected_avwap
            else None,
            "is_waiting_for_cross": setup.is_waiting_for_cross,
        }

    def _setup_from_dict(self, data: Mapping[str, Any]) -> _SetupState:
        return _SetupState(
            symbol=str(data["symbol"]),
            direction=data["direction"],
            anchor_time=self._parse_dt(str(data["anchor_time"])),
            detected_time=self._parse_dt(str(data["detected_time"])),
            consecutive_count=int(data["consecutive_count"]),
            detected_avwap=self._avwap_from_dict(data["detected_avwap"])
            if data.get("detected_avwap")
            else None,
            is_waiting_for_cross=bool(data.get("is_waiting_for_cross", False)),
        )

    def _pending_meta_to_dict(self, meta: _PendingEntryMeta) -> Dict[str, Any]:
        return {"candidate": self._candidate_to_dict(meta.candidate)}

    def _pending_meta_from_dict(self, data: Mapping[str, Any]) -> _PendingEntryMeta:
        return _PendingEntryMeta(candidate=self._candidate_from_dict(data["candidate"]))

    def _candidate_to_dict(self, candidate: _EntryCandidate) -> Dict[str, Any]:
        return {
            "symbol": candidate.symbol,
            "side": candidate.side.value,
            "direction": candidate.direction,
            "signal_time": candidate.signal_time.isoformat(),
            "anchor_time": candidate.anchor_time.isoformat(),
            "setup_detected_time": candidate.setup_detected_time.isoformat(),
            "candle_index": candidate.candle_index,
            "raw_entry_price": candidate.raw_entry_price,
            "order_price": candidate.order_price,
            "stop_for_risk": candidate.stop_for_risk,
            "dynamic_stop_at_entry": candidate.dynamic_stop_at_entry,
            "rigid_stop_at_entry": candidate.rigid_stop_at_entry,
            "trailing_activation_at_entry": candidate.trailing_activation_at_entry,
            "quantity": candidate.quantity,
            "risk_amount": candidate.risk_amount,
            "risk_amount_interpretation": candidate.risk_amount_interpretation,
            "entry_trigger_mode": candidate.entry_trigger_mode,
            "sizing": self._sizing_to_dict(candidate.sizing),
            "avwap": self._avwap_to_dict(candidate.avwap),
        }

    def _candidate_from_dict(self, data: Mapping[str, Any]) -> _EntryCandidate:
        return _EntryCandidate(
            symbol=str(data["symbol"]),
            side=PositionSide(str(data["side"])),
            direction=data["direction"],
            signal_time=self._parse_dt(str(data["signal_time"])),
            anchor_time=self._parse_dt(str(data["anchor_time"])),
            setup_detected_time=self._parse_dt(str(data["setup_detected_time"])),
            candle_index=int(data["candle_index"]),
            raw_entry_price=float(data["raw_entry_price"]),
            order_price=float(data["order_price"]),
            stop_for_risk=float(data["stop_for_risk"]),
            dynamic_stop_at_entry=float(data["dynamic_stop_at_entry"]),
            rigid_stop_at_entry=data.get("rigid_stop_at_entry"),
            trailing_activation_at_entry=float(data["trailing_activation_at_entry"]),
            quantity=float(data["quantity"]),
            risk_amount=float(data["risk_amount"]),
            risk_amount_interpretation=str(data["risk_amount_interpretation"]),
            entry_trigger_mode=str(data["entry_trigger_mode"]),
            sizing=self._sizing_from_dict(data["sizing"]),
            avwap=self._avwap_from_dict(data["avwap"]),
        )

    def _sizing_to_dict(self, sizing: _SizingDecision) -> Dict[str, Any]:
        return {
            "qty": sizing.qty,
            "distance": sizing.distance,
            "entry_price": sizing.entry_price,
            "estimated_exit_price": sizing.estimated_exit_price,
            "risk_amount_interpretation": sizing.risk_amount_interpretation,
            "base_qty_before_costs": sizing.base_qty_before_costs,
            "qty_reduction_from_costs": sizing.qty_reduction_from_costs,
            "sizing_reference_price": sizing.sizing_reference_price,
            "effective_price_for_sizing": sizing.effective_price_for_sizing,
            "entry_slippage_per_unit": sizing.entry_slippage_per_unit,
            "exit_slippage_per_unit": sizing.exit_slippage_per_unit,
            "entry_fee_per_unit": sizing.entry_fee_per_unit,
            "exit_fee_per_unit": sizing.exit_fee_per_unit,
            "total_cost_per_unit": sizing.total_cost_per_unit,
        }

    def _sizing_from_dict(self, data: Mapping[str, Any]) -> _SizingDecision:
        return _SizingDecision(
            qty=float(data["qty"]),
            distance=float(data["distance"]),
            entry_price=float(data["entry_price"]),
            estimated_exit_price=float(data["estimated_exit_price"]),
            risk_amount_interpretation=str(data["risk_amount_interpretation"]),
            base_qty_before_costs=float(data["base_qty_before_costs"]),
            qty_reduction_from_costs=float(data["qty_reduction_from_costs"]),
            sizing_reference_price=float(data["sizing_reference_price"]),
            effective_price_for_sizing=float(data["effective_price_for_sizing"]),
            entry_slippage_per_unit=float(data["entry_slippage_per_unit"]),
            exit_slippage_per_unit=float(data["exit_slippage_per_unit"]),
            entry_fee_per_unit=float(data["entry_fee_per_unit"]),
            exit_fee_per_unit=float(data["exit_fee_per_unit"]),
            total_cost_per_unit=float(data["total_cost_per_unit"]),
        )

    def _runtime_to_dict(self, runtime: _PositionRuntime) -> Dict[str, Any]:
        return {
            "direction": runtime.direction,
            "anchor_time": runtime.anchor_time.isoformat(),
            "setup_detected_time": runtime.setup_detected_time.isoformat(),
            "entry_signal_time": runtime.entry_signal_time.isoformat(),
            "raw_entry_price": runtime.raw_entry_price,
            "dynamic_stop_at_entry": runtime.dynamic_stop_at_entry,
            "rigid_stop_level": runtime.rigid_stop_level,
            "trailing_activation_at_entry": runtime.trailing_activation_at_entry,
            "entry_trigger_mode": runtime.entry_trigger_mode,
            "risk_amount_interpretation": runtime.risk_amount_interpretation,
            "position_sizing_mode": runtime.position_sizing_mode,
            "last_avwap": self._avwap_to_dict(runtime.last_avwap)
            if runtime.last_avwap
            else None,
            "trailing_active": runtime.trailing_active,
            "trailing_stop": runtime.trailing_stop,
            "extreme_price": runtime.extreme_price,
        }

    def _runtime_from_dict(self, data: Mapping[str, Any]) -> _PositionRuntime:
        return _PositionRuntime(
            direction=data["direction"],
            anchor_time=self._parse_dt(str(data["anchor_time"])),
            setup_detected_time=self._parse_dt(str(data["setup_detected_time"])),
            entry_signal_time=self._parse_dt(str(data["entry_signal_time"])),
            raw_entry_price=float(data["raw_entry_price"]),
            dynamic_stop_at_entry=float(data["dynamic_stop_at_entry"]),
            rigid_stop_level=data.get("rigid_stop_level"),
            trailing_activation_at_entry=float(data["trailing_activation_at_entry"]),
            entry_trigger_mode=str(data["entry_trigger_mode"]),
            risk_amount_interpretation=str(data["risk_amount_interpretation"]),
            position_sizing_mode=str(data["position_sizing_mode"]),
            last_avwap=self._avwap_from_dict(data["last_avwap"])
            if data.get("last_avwap")
            else None,
            trailing_active=bool(data.get("trailing_active", False)),
            trailing_stop=data.get("trailing_stop"),
            extreme_price=data.get("extreme_price"),
        )

    def _avwap_to_dict(self, avwap: _AvwapSnapshot) -> Dict[str, Any]:
        return {
            "anchor_index": avwap.anchor_index,
            "anchor_time": avwap.anchor_time.isoformat(),
            "candle_index": avwap.candle_index,
            "vwap": avwap.vwap,
            "stdev": avwap.stdev,
            "upper1": avwap.upper1,
            "lower1": avwap.lower1,
            "upper2": avwap.upper2,
            "lower2": avwap.lower2,
            "upper3": avwap.upper3,
            "lower3": avwap.lower3,
        }

    def _avwap_from_dict(self, data: Mapping[str, Any]) -> _AvwapSnapshot:
        return _AvwapSnapshot(
            anchor_index=int(data["anchor_index"]),
            anchor_time=self._parse_dt(str(data["anchor_time"])),
            candle_index=int(data["candle_index"]),
            vwap=float(data["vwap"]),
            stdev=float(data["stdev"]),
            upper1=float(data["upper1"]),
            lower1=float(data["lower1"]),
            upper2=float(data["upper2"]),
            lower2=float(data["lower2"]),
            upper3=float(data["upper3"]),
            lower3=float(data["lower3"]),
        )

    def _parse_dt_or_none(self, value: Any) -> datetime | None:
        if value is None:
            return None
        return self._parse_dt(str(value))

    def _parse_dt(self, value: str) -> datetime:
        return self._ensure_aware(datetime.fromisoformat(value))
