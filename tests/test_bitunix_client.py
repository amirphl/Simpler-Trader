from __future__ import annotations

import logging
import unittest
from unittest.mock import Mock, patch

import requests

from live_trading.exchange import ExchangeConfig
from live_trading.exchanges.bitunix.client import BitunixClient


class BitunixClientTests(unittest.TestCase):
    def test_request_logs_early_retries_at_debug_and_later_retries_at_warning(
        self,
    ) -> None:
        logger = logging.getLogger("tests.bitunix_client.retry_logging")
        client = BitunixClient(
            ExchangeConfig(api_key="key", api_secret="secret", max_retries=4),
            logger,
        )
        client._session.request = Mock(  # noqa: SLF001
            side_effect=[
                requests.RequestException("first failure"),
                requests.RequestException("second failure"),
                requests.RequestException("third failure"),
                Mock(
                    raise_for_status=Mock(),
                    json=Mock(return_value={"code": 0, "data": {}}),
                ),
            ]
        )

        with (
            patch("live_trading.exchanges.bitunix.client.time.sleep"),
            self.assertLogs(logger, level="DEBUG") as logs,
        ):
            self.assertEqual(client._request("GET", "/retry-test"), {"code": 0, "data": {}})  # noqa: SLF001

        retry_records = [record for record in logs.records if "request retry" in record.msg]
        self.assertEqual(
            [(record.levelno, record.args[0]) for record in retry_records],
            [
                (logging.DEBUG, 1),
                (logging.DEBUG, 2),
                (logging.WARNING, 3),
            ],
        )

    def test_one_way_order_omits_hedge_trade_side(self) -> None:
        client = BitunixClient(ExchangeConfig(api_key="key", api_secret="secret"))
        client._request = Mock(  # noqa: SLF001
            return_value={"code": 0, "data": {"orderId": "order-1"}}
        )

        result = client.place_order(
            symbol="ETHUSDT",
            side="BUY",
            qty=1.0,
            order_type="LIMIT",
            price=100.0,
            effect="GTC",
            trade_side=None,
            client_id="emaavwap-test-id",
        )

        self.assertEqual(result, {"orderId": "order-1"})
        body = client._request.call_args.kwargs["body"]  # noqa: SLF001
        self.assertNotIn("tradeSide", body)
        self.assertEqual(body["clientId"], "emaavwap-test-id")

    def test_order_detail_treats_bitunix_order_not_found_as_absent(self) -> None:
        logger = logging.getLogger("tests.bitunix_client.order_not_found")
        client = BitunixClient(ExchangeConfig(api_key="key", api_secret="secret"), logger)
        client._request = Mock(  # noqa: SLF001
            side_effect=ValueError(
                "Bitunix error code 20007: Order not found. It may have been filled or canceled"
            )
        )

        with self.assertLogs(logger, level="DEBUG"):
            result = client.get_order_detail(client_id="emaavwap-never-sent")

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
