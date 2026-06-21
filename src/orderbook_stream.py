import socketio
import json
import time
from market_data import CoinDCXMarketData


class CoinDCXOrderBook:
    def __init__(self):

        self.socket_endpoint = "wss://stream.coindcx.com"

        self.sio = socketio.Client()
        self.market = CoinDCXMarketData()

        self.orderbook = {"bids": {}, "asks": {}}

        self._register_events()

    # =====================================
    # EVENTS
    # =====================================

    def _register_events(self):
        @self.sio.event
        def connect():
            self.market.safe_print("🟢 Connected")

        @self.sio.event
        def disconnect():
            self.market.safe_print("🔴 Disconnected")

        @self.sio.on("depth-snapshot")
        def on_snapshot(data):

            payload = self._parse_payload(data)

            if payload:
                self._load_snapshot(payload)
                self.print_summary()

        @self.sio.on("depth-update")
        def on_update(data):

            payload = self._parse_payload(data)

            if payload:
                self._apply_update(payload)
                self.print_summary()

    # =====================================
    # PAYLOAD PARSER
    # =====================================

    def _parse_payload(self, raw_data):

        try:

            payload = raw_data.get("data")

            if isinstance(payload, str):
                payload = json.loads(payload)

            return payload

        except Exception as e:

            self.market.safe_print(f"❌ Parse Error: {e}")

            return None

    # =====================================
    # SNAPSHOT
    # =====================================

    def _load_snapshot(self, payload):

        bids = payload.get("bids", {})

        asks = payload.get("asks", {})

        self.orderbook["bids"] = {
            float(price): float(qty) for price, qty in bids.items() if float(qty) > 0
        }

        self.orderbook["asks"] = {
            float(price): float(qty) for price, qty in asks.items() if float(qty) > 0
        }

    # =====================================
    # DELTA UPDATES
    # =====================================

    def _apply_update(self, payload):

        bid_updates = payload.get("bids", {})

        ask_updates = payload.get("asks", {})

        # bids
        for price, qty in bid_updates.items():

            price = float(price)
            qty = float(qty)

            if qty == 0:

                self.orderbook["bids"].pop(price, None)

            else:

                self.orderbook["bids"][price] = qty

        # asks
        for price, qty in ask_updates.items():

            price = float(price)
            qty = float(qty)

            if qty == 0:

                self.orderbook["asks"].pop(price, None)

            else:

                self.orderbook["asks"][price] = qty

    # =====================================
    # STREAM
    # =====================================

    def stream_orderbook(self, pair, depth=20):

        channel = f"{pair}@orderbook@{depth}"

        self.market.safe_print(f"Joining channel: {channel}")

        try:

            self.sio.connect(self.socket_endpoint, transports=["websocket"])

            self.sio.emit("join", {"channelName": channel})

            while True:
                time.sleep(1)

        except KeyboardInterrupt:

            self.market.safe_print("\n🛑 Stopping stream...")

        except Exception as e:

            self.market.safe_print(f"\n❌ Stream Error: {e}")

        finally:

            try:
                self.sio.disconnect()
            except:
                pass

            self.market.safe_print("✅ Stream closed")

    # =====================================
    # DISPLAY
    # =====================================

    def print_summary(self):

        bids = self.orderbook["bids"]

        asks = self.orderbook["asks"]

        if not bids or not asks:
            return

        best_bid = max(bids.keys())

        best_ask = min(asks.keys())

        spread = round(best_ask - best_bid, 4)

        self.market.safe_print("\n" + "=" * 60)

        self.market.safe_print(f"🟢 Best Bid : " f"{best_bid}")

        self.market.safe_print(f"🔴 Best Ask : " f"{best_ask}")

        self.market.safe_print(f"📏 Spread   : " f"{spread}")

        self.market.safe_print(f"📈 Bid Levels: " f"{len(bids)}")

        self.market.safe_print(f"📉 Ask Levels: " f"{len(asks)}")
