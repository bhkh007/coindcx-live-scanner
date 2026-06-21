# market_data.py
import threading
print_lock = threading.Lock()
import requests
import json
import time
from typing import Dict, List, Optional, Any
import os
DEBUG_MODE = False

class CoinDCXMarketData:
    def __init__(self):
        """Initialize market data manager"""

        self.base_url = "https://api.coindcx.com"
        self.working_pairs = {}

        self.public_endpoints = {
            "ticker": "/exchange/ticker",
            "markets": "/exchange/v1/markets",
            "orderbook": "/exchange/v1/depth",
            "candles": "/exchange/v1/market_details",
        }
        self.session = requests.Session()
        self.market_details_cache = None
        self.market_details_last_fetch = 0
        self.candle_cache = {}
        self.candle_cache_expiry = 8
        self.candle_cache_lock = threading.Lock()

    def _make_public_request(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """ Make public API request """

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.get(url=url, params=params, timeout=15)

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            self.safe_print(f"❌ API Error: {e}")

            if hasattr(e, "response") and e.response:
                self.safe_print(f"Status Code: {e.response.status_code}")
                self.safe_print(f"Response: {e.response.text}")

            return None

    def get_all_tickers(self):
        """ Get all market prices """
        return self._make_public_request(self.public_endpoints["ticker"])

    def get_live_price(self, symbol: str):
        """ Get live market price """

        tickers = self.get_all_tickers()
        self.safe_print(type(tickers))
        self.safe_print(type(tickers))
        self.safe_print(len(tickers))

        self.safe_print("\nFIRST TICKER:\n")
        self.safe_print(json.dumps(tickers[0], indent=2))

        if not tickers:
            return None

        symbol = symbol.upper()

        for ticker in tickers:

            market = ticker.get("market", "").upper()

            if market == symbol:

                return {
                    "symbol": market,
                    "last_price": float(ticker.get("last_price", 0)),
                    "bid": float(ticker.get("bid", 0)),
                    "ask": float(ticker.get("ask", 0)),
                    "high": float(ticker.get("high", 0)),
                    "low": float(ticker.get("low", 0)),
                    "volume": float(ticker.get("volume", 0)),
                    "change_24h": float(ticker.get("change_24_hour", 0)),
                }

        self.safe_print(f"❌ Symbol not found: {symbol}")
        return None

    def get_multiple_prices(self, symbols: List[str]) -> Dict:

        results = {}

        for symbol in symbols:
            results[symbol] = self.get_live_price(symbol)

        return results

    def get_market_details(self):

        now = time.time()

        # cache for 10 minutes
        if (
            self.market_details_cache
            and now - self.market_details_last_fetch < 600
        ):
            return self.market_details_cache

        try:

            url = (
                "https://api.coindcx.com"
                "/exchange/v1/markets_details"
            )

            response = self.session.get(
                url,
                timeout=10
            )

            response.raise_for_status()

            data = response.json()

            self.market_details_cache = data
            self.market_details_last_fetch = now

            return data

        except Exception as e:

            self.safe_print(
                f"❌ Market Details Error: {e}"
            )

            return []

    def get_candles(self, symbol, interval="5m", limit=100):

        cache_key = f"{symbol}_{interval}_{limit}"
        now = time.time()

        with self.candle_cache_lock:
            cached = self.candle_cache.get(cache_key)
            if cached:
                cache_time, data = cached
                if now - cache_time < self.candle_cache_expiry:
                    return data
            
        try:
            url = "https://public.coindcx.com/market_data/candles"

            actual_pair = self.get_pair_symbol(symbol)
            base_symbol = symbol.replace("USDT", "")
            cached_pair = self.working_pairs.get(symbol)

            if cached_pair:
                pairs_to_try = [cached_pair]
            else:
                pairs_to_try = [
                    actual_pair,
                    f"B-{base_symbol}_USDT",
                    f"KC-{base_symbol}_USDT",
                    f"I-{base_symbol}_USDT",
                    f"G-{base_symbol}_USDT",
                ]

            if not actual_pair:
                self.safe_print(f"❌ No pair mapping found: {symbol}")
                return []

            # PAIR FALLBACK SYSTEM
            
            pairs_to_try = [f"B-{base_symbol}_USDT", actual_pair]

            # B works most often on CoinDCX
            candidate_pairs = [
                actual_pair,  # ALWAYS FIRST
                f"B-{base_symbol}_USDT",
                f"KC-{base_symbol}_USDT",
                f"I-{base_symbol}_USDT",
                f"G-{base_symbol}_USDT",
            ]

            SPECIAL_PAIRS = {
                "TONUSDT",
                "TAOUSDT",
                "ASTERUSDT",
            }
            # remove duplicates while preserving order
            seen = set()

            for pair in candidate_pairs:
                if pair and pair not in seen:
                    if symbol in SPECIAL_PAIRS:
                        pairs_to_try.append(pair)
                    seen.add(pair)

            # TRY ALL PAIRS
            for pair in pairs_to_try:
                if DEBUG_MODE:
                    self.safe_print(
                            f"TESTING PAIR | "
                            f"symbol={symbol} | "
                            f"pair={pair}",
                        )

                try:

                    response = self.session.get(
                        url,
                        params={
                            "pair": pair,
                            "interval": interval,
                            "limit": limit,
                        },
                        timeout=(2,4)
                    )

                    data = response.json()

                    if (
                        response.status_code == 200
                        and isinstance(data, list)
                        and len(data) > 0
                    ):

                        if DEBUG_MODE:
                            self.safe_print(
                                f"✅ Candle pair success | "
                                f"symbol={symbol} | "
                                f"pair={pair}"
                            )
                        self.working_pairs[symbol] = pair
                        with self.candle_cache_lock:
                            self.candle_cache[cache_key] = (
                                time.time(),
                                data
                            )
                        return data

                except Exception as e:
                    self.safe_print(
                        f"⚠️ Request failed | "
                        f"symbol={symbol} | "
                        f"pair={pair} | "
                        f"error={e}"
                    )

            self.safe_print(
                f"❌ No candle data available | "
                f"symbol={symbol}"
            )
            
            return []
        
        except Exception as e:

            self.safe_print(
                f"❌ get_candles failed | "
                f"symbol={symbol} | "
                f"error={e}"
            )
            
            return []
    
    def safe_print(self, *args, **kwargs):
        with print_lock:
            print(*args, **kwargs, flush=True)

    def get_orderbook(self, symbol: str):
        pair = self.get_pair_symbol(symbol)

        if not pair:
            return None

        self.safe_print(f"Using Pair: {pair}")

        endpoint = f"/exchange/v1/depth"

        params = {"pair": pair}

        data = self._make_public_request(endpoint, params=params)

        if not data:
            return None

        bids = data.get("bids", {})
        asks = data.get("asks", {})

        best_bid = None
        best_ask = None

        if bids:
            best_bid = max(float(price) for price in bids.keys())

        if asks:
            best_ask = min(float(price) for price in asks.keys())

        spread = None

        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid

        return {
            "symbol": symbol.upper(),
            "pair": pair,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "top_10_bids": list(bids.items())[:10],
            "top_10_asks": list(asks.items())[:10],
        }

    def print_orderbook(self, symbol: str):
        data = self.get_orderbook(symbol)

        if not data:
            self.safe_print("❌ Failed to fetch orderbook")
            return

        self.safe_print("\n" + "=" * 60)
        self.safe_print(f"📚 ORDERBOOK: {symbol.upper()}")
        self.safe_print("=" * 60)

        self.safe_print(f"🟢 Best Bid : {data['best_bid']}")
        self.safe_print(f"🔴 Best Ask : {data['best_ask']}")
        self.safe_print(f"📏 Spread   : {data['spread']}")

        self.safe_print(f"\n📈 Bid Levels: {data['bid_levels']}")

        self.safe_print(f"📉 Ask Levels: {data['ask_levels']}")

        self.safe_print("\n🟢 TOP 10 BIDS")
        self.safe_print("-" * 40)

        for price, qty in data["top_10_bids"]:
            self.safe_print(f"Price: {price} | Qty: {qty}")

        self.safe_print("\n🔴 TOP 10 ASKS")
        self.safe_print("-" * 40)

        for price, qty in data["top_10_asks"]:
            self.safe_print(f"Price: {price} | Qty: {qty}")

    def get_available_markets(self):

        return self._make_public_request(self.public_endpoints["markets"])

    def search_market(self, search_term: str):
        markets = self.get_available_markets()

        if not markets:
            return []

        search_term = search_term.upper()

        results = []

        for market in markets:

            symbol = str(market).upper()

            if search_term in symbol:
                results.append(symbol)

        return sorted(results)

    def print_live_price(self, symbol: str):

        data = self.get_live_price(symbol)

        if not data:
            return

        self.safe_print("\n" + "=" * 50)
        self.safe_print(f"📈 LIVE MARKET PRICE: {symbol}")
        self.safe_print("=" * 50)

        self.safe_print(f"💰 Last Price : {data['last_price']}")
        self.safe_print(f"🟢 Bid Price  : {data['bid']}")
        self.safe_print(f"🔴 Ask Price  : {data['ask']}")
        self.safe_print(f"📈 24H High   : {data['high']}")
        self.safe_print(f"📉 24H Low    : {data['low']}")
        self.safe_print(f"📊 Volume     : {data['volume']}")
        self.safe_print(f"⚡ 24H Change : {data['change_24h']}%")

    def stream_price(self, symbol: str, refresh_seconds: int = 2):

        try:
            while True:

                os.system("cls")

                data = self.get_live_price(symbol)

                if data:

                    self.safe_print(f"\n📈 {symbol.upper()} LIVE PRICE")

                    self.safe_print(f"💰 Price: {data['last_price']}")

                    self.safe_print(f"🟢 Bid: {data['bid']}")

                    self.safe_print(f"🔴 Ask: {data['ask']}")

                    self.safe_print(f"📊 Volume: {data['volume']}")

                time.sleep(refresh_seconds)

        except KeyboardInterrupt:
            self.safe_print("\n🛑 Stream stopped")

    def debug_market_details(self):

        data = self._make_public_request("/exchange/v1/markets_details")

        if not data:
            self.safe_print("No data")
            return

        self.safe_print(type(data))
        self.safe_print("TOTAL:", len(data))

        self.safe_print("\nFIRST ITEM:\n")
        self.safe_print(json.dumps(data[0], indent=2))

    def get_pair_symbol(self, symbol: str):
        markets = self.get_market_details()

        if not markets:
            return None

        symbol = symbol.upper()

        for market in markets:

            if market.get("symbol") == symbol:

                return market.get("pair")

        self.safe_print(f"❌ Pair not found for {symbol}")
        return None