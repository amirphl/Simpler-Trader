from __future__ import annotations

import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from candle_downloader.binance import BinanceClient, BinanceClientConfig


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeOpener:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []

    def open(self, request, timeout=None):  # noqa: ANN001
        self.urls.append(request.full_url)
        if not self._responses:
            raise AssertionError("No fake response remaining")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(response)


class BinanceClientTests(unittest.TestCase):
    def test_config_rejects_zero_retries(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_retries"):
            BinanceClientConfig(max_retries=0)

    def test_fetch_klines_parses_valid_payload(self) -> None:
        opener = _FakeOpener(
            [
                [
                    [
                        1_700_000_000_000,
                        "100.0",
                        "110.0",
                        "95.0",
                        "105.0",
                        "12.5",
                        1_700_000_059_999,
                    ]
                ]
            ]
        )
        client = BinanceClient(BinanceClientConfig(), opener=opener)

        candles = client.fetch_klines(
            symbol="btcusdt",
            interval="1m",
            start_ms=1_700_000_000_000,
            end_ms=1_700_000_060_000,
            limit=1,
        )

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0].symbol, "BTCUSDT")
        self.assertIn("symbol=BTCUSDT", opener.urls[0])

    def test_config_strips_base_url(self) -> None:
        config = BinanceClientConfig(base_url=" https://api.binance.com/ ")
        client = BinanceClient(config, opener=_FakeOpener([[]]))

        self.assertEqual(client._base_url, "https://api.binance.com")  # noqa: SLF001

    def test_fetch_klines_raises_for_api_error_payload(self) -> None:
        opener = _FakeOpener([{"code": -1121, "msg": "Invalid symbol."}])
        client = BinanceClient(BinanceClientConfig(), opener=opener)

        with self.assertRaisesRegex(RuntimeError, "Invalid symbol"):
            client.fetch_klines(
                symbol="badpair",
                interval="1m",
                start_ms=1,
                end_ms=2,
                limit=1,
            )

    def test_fetch_klines_raises_for_malformed_row(self) -> None:
        opener = _FakeOpener([["not-a-kline-row"]])
        client = BinanceClient(BinanceClientConfig(), opener=opener)

        with self.assertRaisesRegex(RuntimeError, "invalid kline row"):
            client.fetch_klines(
                symbol="BTCUSDT",
                interval="1m",
                start_ms=1,
                end_ms=2,
                limit=1,
            )

    def test_fetch_top_symbols_filters_to_unique_usdt_pairs(self) -> None:
        opener = _FakeOpener(
            [
                [
                    {"symbol": "ethusdt", "quoteVolume": "10"},
                    {"symbol": "BTCUSDT", "quoteVolume": "20"},
                    {"symbol": "BTCUSDT", "quoteVolume": "19"},
                    {"symbol": "ETHBTC", "quoteVolume": "999"},
                    {"symbol": "SOLUSDT", "quoteVolume": "5"},
                ]
            ]
        )
        client = BinanceClient(BinanceClientConfig(), opener=opener)

        symbols = client.fetch_top_symbols(limit=3)

        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

    def test_non_retryable_http_error_raises_immediately(self) -> None:
        error = HTTPError(
            url="https://api.binance.com/api/v3/klines",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"code":-1121,"msg":"Invalid symbol."}'),
        )
        opener = _FakeOpener([error])
        client = BinanceClient(BinanceClientConfig(max_retries=3), opener=opener)

        with self.assertRaisesRegex(RuntimeError, "Invalid symbol"):
            client.fetch_top_symbols(limit=1)

        self.assertEqual(len(opener.urls), 1)

    def test_retryable_http_error_honors_retry_after_header(self) -> None:
        retryable_error = HTTPError(
            url="https://api.binance.com/api/v3/ticker/24hr",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "3"},
            fp=io.BytesIO(b'{"code":-1003,"msg":"Too many requests."}'),
        )
        opener = _FakeOpener([retryable_error, []])
        client = BinanceClient(BinanceClientConfig(max_retries=2), opener=opener)

        with patch("candle_downloader.binance.time.sleep") as sleep_mock:
            symbols = client.fetch_top_symbols(limit=5)

        self.assertEqual(symbols, [])
        sleep_mock.assert_called_once_with(3.0)


if __name__ == "__main__":
    unittest.main()
