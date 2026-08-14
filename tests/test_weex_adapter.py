from __future__ import annotations

import unittest

from live_trading.exchange import ExchangeConfig, MarginMode, PositionSide
from live_trading.exchanges.weex import WeexExchange


class _FakeWeexClient:
    def __init__(self) -> None:
        self.markets = {
            "ZEC/USDT": {
                "id": "ZECUSDT",
                "symbol": "ZEC/USDT",
                "base": "ZEC",
                "quote": "USDT",
                "spot": True,
                "contract": False,
                "swap": False,
                "limits": {"amount": {"min": 0.1}},
            },
            "ZEC/USDT:USDT": {
                "id": "ZECUSDT",
                "symbol": "ZEC/USDT:USDT",
                "base": "ZEC",
                "quote": "USDT",
                "spot": False,
                "contract": True,
                "swap": True,
                "limits": {"amount": {"min": 0.1}},
            },
        }
        self.balance = {
            "USDT": {"free": 125.0, "total": 150.0},
            "ZEC": {"free": 2.5, "total": 2.5},
        }
        self.positions = []
        self.created_orders = []
        self.stop_orders = []
        self.open_orders = []
        self.closed_orders = []
        self.margin_calls = []
        self.leverage_calls = []
        self.cancel_all_calls = []

    def load_markets(self):
        return self.markets

    def amount_to_precision(self, symbol, quantity):
        del symbol
        return f"{float(quantity):.3f}"

    def price_to_precision(self, symbol, price):
        del symbol
        return f"{float(price):.2f}"

    def fetch_balance(self, params):
        self.balance_params = params
        return self.balance

    def fetch_ticker(self, symbol):
        self.ticker_symbol = symbol
        return {"last": 30.25}

    def fetch_tickers(self, symbols=None, params=None):
        self.ticker_symbols = symbols
        self.tickers_params = params
        return {
            "ZEC/USDT:USDT": {
                "last": 30.25,
                "change": 0.5,
                "percentage": 1.68,
                "baseVolume": 10.0,
                "quoteVolume": 302.5,
            }
        }

    def fetch_positions(self, params=None):
        self.fetch_positions_params = params
        return list(self.positions)

    def fetch_positions_for_symbol(self, symbol):
        self.position_symbol = symbol
        return list(self.positions)

    def set_margin_mode(self, mode, symbol):
        self.margin_calls.append((mode, symbol))

    def set_leverage(self, leverage, symbol, params=None):
        self.leverage_calls.append((leverage, symbol, params))

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        params = dict(params or {})
        order = {
            "id": f"order-{len(self.created_orders) + 1}",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "open",
            "timestamp": 1_700_000_000_000,
            "params": params,
        }
        self.created_orders.append(order)
        if "stopLossPrice" in params:
            self.stop_orders.append(
                {
                    "id": order["id"],
                    "side": side,
                    "reduceOnly": True,
                    "triggerPrice": params["stopLossPrice"],
                }
            )
        return order

    def close_position(self, symbol, side):
        self.close_call = (symbol, side)
        return {
            "id": "close-1",
            "symbol": symbol,
            "side": side,
            "amount": 2.0,
            "status": "closed",
        }

    def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
        self.open_orders_call = (symbol, since, limit, params)
        if params and params.get("trigger"):
            return list(self.stop_orders)
        return list(self.open_orders)

    def cancel_order(self, order_id, symbol, params=None):
        self.cancel_call = (order_id, symbol, params)
        self.stop_orders = [order for order in self.stop_orders if order["id"] != order_id]
        return {"id": order_id, "status": "canceled"}

    def cancel_all_orders(self, symbol, params=None):
        self.cancel_all_calls.append((symbol, params))
        return []

    def fetch_order(self, order_id, symbol, params=None):
        self.fetch_order_call = (order_id, symbol, params)
        return {"id": order_id, "status": "open"}

    def fetch_canceled_and_closed_orders(self, symbol, limit=None):
        self.closed_orders_call = (symbol, limit)
        return list(self.closed_orders)

    def fetch_ohlcv(self, symbol, interval, since, limit):
        self.ohlcv_call = (symbol, interval, since, limit)
        return [[1_700_000_000_000, 1, 2, 0.5, 1.5, 100]]

    def fetch_order_book(self, symbol, limit):
        return {"symbol": symbol, "limit": limit}

    def fetch_funding_rate(self, symbol):
        return {"symbol": symbol, "fundingRate": 0.0001}

    def fetch_position_mode(self, symbol):
        self.position_mode_symbol = symbol
        return {"hedged": False}

    def fetch_time(self):
        return 1_700_000_000_000


class WeexExchangeTests(unittest.TestCase):
    def _exchange(self, mode="futures"):
        self.client = _FakeWeexClient()
        return WeexExchange(
            ExchangeConfig(
                api_key="key",
                api_secret="secret",
                passphrase="passphrase",
                trading_mode=mode,
            ),
            client=self.client,
            symbols=["ZECUSDT"],
        )

    def test_futures_uses_ccxt_swap_symbol_and_configures_account(self) -> None:
        exchange = self._exchange()

        self.assertEqual(exchange.fetch_price("ZECUSDT"), 30.25)
        self.assertEqual(self.client.ticker_symbol, "ZEC/USDT:USDT")
        self.assertEqual(exchange.get_account_balance(), 125.0)
        self.assertEqual(self.client.balance_params, {"type": "swap"})

        exchange.set_margin_mode("ZECUSDT", MarginMode.ISOLATED)
        exchange.set_leverage("ZECUSDT", 5)

        self.assertEqual(self.client.margin_calls, [("isolated", "ZEC/USDT:USDT")])
        self.assertEqual(
            self.client.leverage_calls,
            [(5, "ZEC/USDT:USDT", {"marginMode": "isolated"})],
        )

    def test_tickers_pass_market_type_as_ccxt_params(self) -> None:
        exchange = self._exchange()

        tickers = exchange.get_24h_tickers()

        self.assertEqual(self.client.ticker_symbols, None)
        self.assertEqual(self.client.tickers_params, {"type": "swap"})
        self.assertEqual(tickers[0]["symbol"], "ZECUSDT")

    def test_futures_limit_entry_attaches_native_stop_and_client_id(self) -> None:
        exchange = self._exchange()

        order = exchange.open_limit_position(
            "ZECUSDT",
            PositionSide.LONG,
            quantity=1.23456,
            price=30.256,
            leverage=5,
            margin_mode=MarginMode.ISOLATED,
            stop_loss=29.0,
            client_id="ema-avwap-zec",
        )

        self.assertEqual(order.order_id, "order-1")
        created = self.client.created_orders[0]
        self.assertEqual(created["symbol"], "ZEC/USDT:USDT")
        self.assertEqual(created["amount"], 1.235)
        self.assertEqual(created["price"], 30.26)
        self.assertEqual(created["params"]["clientOrderId"], "ema-avwap-zec")
        self.assertEqual(
            created["params"]["stopLoss"],
            {"triggerPrice": 29.0, "triggerPriceType": "mark"},
        )

    def test_futures_positions_and_stops_are_normalized_and_verified(self) -> None:
        exchange = self._exchange()
        self.client.positions = [
            {
                "symbol": "ZEC/USDT:USDT",
                "side": "long",
                "contracts": 2.0,
                "entryPrice": 30.0,
                "leverage": 5,
                "marginMode": "isolated",
                "unrealizedPnl": 1.2,
                "liquidationPrice": 10.0,
                "id": "position-1",
            }
        ]

        position = exchange.get_position("ZECUSDT")

        self.assertEqual(position.symbol, "ZECUSDT")
        self.assertEqual(position.side, PositionSide.LONG)
        self.assertEqual(position.margin_mode, MarginMode.ISOLATED)
        self.assertEqual(exchange.ensure_position_stop_loss(position, 28.126), 28.13)
        created = self.client.created_orders[0]
        self.assertEqual(created["side"], "sell")
        self.assertEqual(created["params"]["stopLossPrice"], 28.13)
        self.assertEqual(self.client.open_orders_call[3], {"trigger": True})

    def test_stop_replacement_is_confirmed_before_old_stop_is_cancelled(self) -> None:
        exchange = self._exchange()
        self.client.positions = [
            {
                "symbol": "ZEC/USDT:USDT",
                "side": "long",
                "contracts": 2.0,
                "entryPrice": 30.0,
                "leverage": 5,
                "marginMode": "isolated",
            }
        ]
        self.client.stop_orders = [
            {
                "id": "old-stop",
                "side": "sell",
                "reduceOnly": True,
                "triggerPrice": 27.0,
            }
        ]

        actual = exchange.ensure_position_stop_loss(exchange.get_position("ZECUSDT"), 28.0)

        self.assertEqual(actual, 28.0)
        self.assertEqual(self.client.created_orders[0]["id"], "order-1")
        self.assertEqual(self.client.cancel_call, ("old-stop", "ZEC/USDT:USDT", {"trigger": True}))

    def test_futures_cancels_trigger_orders_and_recovers_by_client_id(self) -> None:
        exchange = self._exchange()
        self.client.open_orders = [
            {"id": "entry-1", "clientOrderId": "ema-zec", "status": "open"}
        ]

        exchange.cancel_all_orders("ZECUSDT")
        status = exchange.get_order_status(symbol="ZECUSDT", client_id="ema-zec")

        self.assertEqual(
            self.client.cancel_all_calls,
            [
                ("ZEC/USDT:USDT", {}),
                ("ZEC/USDT:USDT", {"trigger": True}),
            ],
        )
        self.assertEqual(status["id"], "entry-1")
        self.assertEqual(status["status"], "NEW")

    def test_futures_client_id_recovery_searches_recent_history(self) -> None:
        exchange = self._exchange()
        self.client.closed_orders = [
            {"id": "entry-2", "clientOrderId": "ema-zec", "status": "closed"}
        ]

        status = exchange.get_order_status(symbol="ZECUSDT", client_id="ema-zec")

        self.assertEqual(self.client.closed_orders_call, ("ZEC/USDT:USDT", 100))
        self.assertEqual(status["id"], "entry-2")
        self.assertEqual(status["status"], "FILLED")

    def test_futures_order_id_lookup_uses_server_id_and_normalizes_status(self) -> None:
        exchange = self._exchange()
        fetch_calls = []

        def fetch_order(order_id, symbol, params=None):
            fetch_calls.append((order_id, symbol, params))
            return {"id": order_id, "symbol": symbol, "status": "closed"}

        self.client.fetch_order = fetch_order

        status = exchange.get_order_status(
            symbol="ZECUSDT", order_id="entry-1", client_id="ema-zec"
        )

        self.assertEqual(fetch_calls, [("entry-1", "ZEC/USDT:USDT", None)])
        self.assertEqual(status["status"], "FILLED")

    def test_spot_uses_spot_market_and_only_allows_long_exposure(self) -> None:
        exchange = self._exchange("spot")

        self.assertEqual(exchange.get_account_balance(), 125.0)
        self.assertEqual(self.client.balance_params, {"type": "spot"})
        position = exchange.get_position("ZECUSDT")
        self.assertEqual(position.size, 2.5)
        self.assertEqual(position.leverage, 1.0)

        entry = exchange.open_limit_position(
            "ZECUSDT",
            PositionSide.LONG,
            quantity=1.0,
            price=30.0,
            leverage=1,
            margin_mode=MarginMode.CROSS,
            client_id="spot-entry",
        )
        self.assertEqual(entry.order_id, "order-1")
        self.assertEqual(self.client.created_orders[0]["symbol"], "ZEC/USDT")
        self.assertEqual(self.client.created_orders[0]["params"]["clientOrderId"], "spot-entry")

        with self.assertRaisesRegex(RuntimeError, "cannot open short"):
            exchange.open_market_position(
                "ZECUSDT",
                PositionSide.SHORT,
                quantity=1.0,
                leverage=1,
                margin_mode=MarginMode.CROSS,
            )
        with self.assertRaisesRegex(RuntimeError, "does not support leverage"):
            exchange.set_leverage("ZECUSDT", 2)

    def test_spot_testnet_is_rejected_before_ccxt_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "sandbox supports swap markets only"):
            WeexExchange(
                ExchangeConfig(
                    api_key="key",
                    api_secret="secret",
                    passphrase="passphrase",
                    testnet=True,
                    trading_mode="spot",
                )
            )
