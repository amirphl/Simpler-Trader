"""Live trading strategy implementation."""

from __future__ import annotations

import logging
from datetime import datetime
import os
from json import loads
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from candle_downloader.models import Candle

from .exchange import Exchange, PositionSide
from .heiken_ashi import calculate_heiken_ashi, detect_reversal_signal
from .models import LiveTradingConfig, SymbolInfo, TradingSignal, TradingState


class LiveTradingStrategy:
    """Live trading strategy based on Heiken Ashi reversals."""
    
    def __init__(
        self,
        config: LiveTradingConfig,
        exchange: Exchange,
        state: TradingState,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._exchange = exchange
        self._state = state
        self._log = logger or logging.getLogger(__name__)
        self._binance_opener = build_opener(*self._build_proxy_handlers())
    
    def analyze_symbol(
        self,
        symbol_info: SymbolInfo,
        current_time: datetime,
    ) -> Optional[TradingSignal]:
        """Analyze a symbol and generate trading signal if conditions are met.
        
        Args:
            symbol_info: Information about the symbol
            current_time: Current timestamp
            
        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        symbol = symbol_info.symbol
        
        # Check if symbol is disabled
        if self._state.is_symbol_disabled(symbol, current_time, self._config.disable_symbol_hours):
            self._log.debug(f"Symbol {symbol} is disabled")
            return None
        
        # Check if position already exists
        if symbol in self._state.active_positions:
            self._log.debug(f"Position already exists for {symbol}")
            return None
        
        try:
            # Fetch recent candles for Heiken Ashi calculation
            # We need at least W+1 candles for reversal detection
            required_candles = self._config.heiken_ashi_candles_before + 1
            # Pull candles from Binance directly to ensure consistent format/data
            klines = self._fetch_binance_klines(
                symbol=symbol,
                interval=self._config.timeframe,
                limit=min(required_candles * 2, 100) + 1,  # extra for dropping last
            )
            
            if len(klines) < required_candles:
                self._log.debug(f"Insufficient candles for {symbol}: {len(klines)} < {required_candles}")
                return None
            
            # Convert to Candle objects
            candles = [
                Candle.from_binance(symbol, self._config.timeframe, kline)
                for kline in klines
            ]
            
            # Calculate Heiken Ashi candles
            ha_candles = calculate_heiken_ashi(candles)
            
            if len(ha_candles) < required_candles:
                return None
            
            # Detect reversal signal
            signal_type = detect_reversal_signal(
                ha_candles,
                lookback_candles=self._config.heiken_ashi_candles_before,
            )
            
            if signal_type is None:
                return None
            
            # Determine position side based on price movement direction
            if signal_type == "LONG":
                # Price was falling (top losers), now reversing up
                # if symbol_info.price_change_pct >= 0: # TODO
                #     return None  # Symbol should be from losers list
                side = PositionSide.LONG
            else:  # SHORT
                # Price was rising (top gainers), now reversing down
                # if symbol_info.price_change_pct <= 0: # TODO
                #     return None  # Symbol should be from gainers list
                side = PositionSide.SHORT
            
            current_candle = candles[-1]
            current_ha_candle = ha_candles[-1]
            
            # Entry price is current market price
            # entry_price = symbol_info.current_price # TODO

            # Use last original close price
            entry_price = current_candle.close  
            
            # Stop loss is the open price of HA candle
            stop_loss = current_ha_candle.ha_open
            
            # Calculate take profit
            if side == PositionSide.LONG:
                take_profit = entry_price * (1 + self._config.take_profit_pct / 100)
            else:
                take_profit = entry_price * (1 - self._config.take_profit_pct / 100)
            
            # Validate signal
            if side == PositionSide.LONG:
                if stop_loss >= entry_price:
                    self._log.warning(
                        f"Invalid LONG signal for {symbol}: stop_loss ({stop_loss}) >= entry ({entry_price})"
                    )
                    return None
            else:
                if stop_loss <= entry_price:
                    self._log.warning(
                        f"Invalid SHORT signal for {symbol}: stop_loss ({stop_loss}) <= entry ({entry_price})"
                    )
                    return None
            
            # Create signal
            signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=self._config.leverage,
                margin_mode=self._config.margin_mode,
                reason=f"HA reversal: {signal_type} after {self._config.heiken_ashi_candles_before} candles",
                metadata={
                    "price_change_pct": symbol_info.price_change_pct,
                    "current_ha_candle": {
                        "open": current_ha_candle.ha_open,
                        "high": current_ha_candle.ha_high,
                        "low": current_ha_candle.ha_low,
                        "close": current_ha_candle.ha_close,
                    },
                },
            )
            
            self._log.info(
                f"Signal generated: {symbol} {side.value} @ {entry_price:.6f}, "
                f"SL: {stop_loss:.6f}, TP: {take_profit:.6f}"
            )
            
            return signal
            
        except Exception as e:
            self._log.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return None

    def _fetch_binance_klines(self, symbol: str, interval: str, limit: int) -> List[list]:
        """Fetch klines directly from Binance and drop the last (potentially open) candle."""
        params = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
        url = f"https://api.binance.com/api/v3/klines?{params}"
        try:
            with self._binance_opener.open(Request(url), timeout=10) as resp:
                klines = loads(resp.read())
                # Drop the last entry to avoid using an in-progress candle
                if klines:
                    klines = klines[:-1]
                return klines
        except (HTTPError, URLError) as exc:
            self._log.warning("Failed to fetch Binance klines for %s: %s", symbol, exc)
            return []
        except Exception as exc:
            self._log.warning("Error parsing Binance klines for %s: %s", symbol, exc)
            return []

    def _build_proxy_handlers(self):
        """Reuse environment proxy settings for Binance HTTP calls."""
        http_env = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_env = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        all_env = os.getenv("ALL_PROXY") or os.getenv("all_proxy")

        proxies = {}
        if all_env:
            proxies["http"] = all_env
            proxies["https"] = all_env
        else:
            if http_env:
                proxies["http"] = http_env
            if https_env:
                proxies["https"] = https_env

        return [ProxyHandler(proxies)] if proxies else []
    
    def generate_signals(
        self,
        top_gainers: List[SymbolInfo],
        top_losers: List[SymbolInfo],
        current_time: datetime,
    ) -> List[TradingSignal]:
        """Generate trading signals for top movers.
        
        Args:
            top_gainers: Symbols with highest positive price change
            top_losers: Symbols with highest negative price change
            current_time: Current timestamp
            
        Returns:
            List of trading signals
        """
        signals: List[TradingSignal] = []
        
        # Analyze top losers for LONG opportunities (reversal from decline)
        self._log.info(f"Analyzing {len(top_losers)} top losers for LONG signals")
        for symbol_info in top_losers:
            signal = self.analyze_symbol(symbol_info, current_time)
            if signal is not None: # and signal.side == PositionSide.LONG: # TODO
                signals.append(signal)
        
        # Analyze top gainers for SHORT opportunities (reversal from rally)
        self._log.info(f"Analyzing {len(top_gainers)} top gainers for SHORT signals")
        for symbol_info in top_gainers:
            signal = self.analyze_symbol(symbol_info, current_time)
            if signal is not None: # and signal.side == PositionSide.SHORT: # TODO
                signals.append(signal)
        
        self._log.info(f"Generated {len(signals)} trading signals")
        return signals
