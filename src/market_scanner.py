# market_scanner.py
import time
import numpy as np
import pandas as pd
from market_data import CoinDCXMarketData
from indicators import TechnicalIndicators
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback
from ta.momentum import RSIIndicator

DEAD_CANDLE_MARKETS = {
    "TONUSDT",
    "TAOUSDT",
    "ASTERUSDT",
    "LABUSDT",
    "HYPEUSDT",
    "ONDOUSDT",
}

class CoinDCXMarketScanner:
    def __init__(self):

        self.market = CoinDCXMarketData()
        self.print_lock = threading.Lock()

        self.dead_pairs = set()
        self.debug = False

    def get_rsi_score(self, rsi, trend):

        score = 0

        if trend == "BULLISH":  # BULLISH TREND → LOOK FOR DIP BUY

            if 25 <= rsi <= 35:
                score = 30
            elif 35 < rsi <= 45:
                score = 15
            elif rsi > 70:
                score = -10  # overextended

        elif trend == "BEARISH":  # BEARISH TREND → LOOK FOR SHORT

            if 60 <= rsi <= 75:
                score = 30
            elif 50 <= rsi < 60:
                score = 15
            elif rsi < 35:
                score = 20  # already dumped

        return score

    def get_rsi_reversal_score(self, candles):
        try:
            closes = [float(candle["close"]) for candle in candles]
            if len(closes) < 20:
                return 50, 0
            
            series = pd.Series(closes)
            rsi = RSIIndicator(close=series, window=14).rsi().iloc[-1]
            
            # More realistic reversal scoring
            if rsi <= 25:
                score = 35  # Extreme oversold - high conviction
            elif rsi <= 35:
                score = 25  # Good reversal zone
            elif rsi <= 45:
                score = 10  # Acceptable dip
            elif rsi <= 55:
                score = 0   # Neutral
            elif rsi <= 70:
                score = -5  # Slightly overbought
            else:
                score = -15  # Very overbought - avoid
                
            return rsi, score
        except Exception as e:
            self.market.safe_print(f"RSI Error: {e}")
            return 50, 0
    
    def get_momentum_score(self, candles):
        try:
            closes = [float(c["close"]) for c in candles]
            
            if len(closes) < 20:
                return 0
            
            change_1m = ((closes[-1] - closes[-2]) / closes[-2]) * 100
            change_5m = ((closes[-1] - closes[-6]) / closes[-6]) * 100
            change_15m = ((closes[-1] - closes[-16]) / closes[-16]) * 100
            
            score = 0
            
            # STRONG BULLISH MOMENTUM
            if change_1m > 0.2 and change_5m > 0.4 and change_15m > 0.7:
                score = 20
            # MODERATE BULLISH
            elif change_1m > 0.1 and change_5m > 0.2:
                score = 10
            # WEAK BULLISH
            elif change_1m > 0.05 and change_5m > 0.1:
                score = 5
            # STRONG BEARISH
            elif change_1m < -0.2 and change_5m < -0.4 and change_15m < -0.7:
                score = -15
            # MODERATE BEARISH
            elif change_1m < -0.1 and change_5m < -0.2:
                score = -8
            # WEAK BEARISH
            elif change_1m < -0.05 and change_5m < -0.1:
                score = -3
            
            self.market.safe_print(
                f"Momentum Debug | "
                f"1m={change_1m:.2f}% | "
                f"5m={change_5m:.2f}% | "
                f"15m={change_15m:.2f}% | "
                f"Score={score}"
            )
            
            return score
            
        except Exception as e:
            self.market.safe_print(f"Momentum Error: {e}")
            return 0

    def get_volume_spike(self, candles):
        try:
            if len(candles) < 2:
                return 0, 0
            
            volumes = pd.Series([float(c["volume"]) for c in candles])
            current_volume = volumes.iloc[-1]
            
            if len(volumes) > 20:
                avg_volume = volumes.iloc[-21:-1].mean()
            else:
                avg_volume = volumes.iloc[:-1].mean()
            
            if avg_volume <= 0:
                return 0, 0
            
            volume_ratio = current_volume / avg_volume
            
            # LESS STRICT VOLUME SCORING
            if volume_ratio >= 2.5:
                score = 20
            elif volume_ratio >= 1.8:
                score = 12
            elif volume_ratio >= 1.3:
                score = 8
            elif volume_ratio < 0.7:
                score = -5
            else:
                score = 0
            
            self.market.safe_print(f"Volume Debug | Ratio={volume_ratio:.2f}x | Score={score}")
            return round(volume_ratio, 2), score
            
        except Exception as e:
            self.market.safe_print(f"Volume Error: {e}")
            return 0, 0

    def get_multi_timeframe_score(self, candles_5m, candles_15m, candles_1h):
        try:
            trend_5m = self.get_ema_trend(candles_5m)
            trend_15m = self.get_ema_trend(candles_15m)
            trend_1h = self.get_ema_trend(candles_1h)
            
            score = 0
            
            # ALL BULLISH - STRONGEST SIGNAL
            if trend_5m == "BULLISH" and trend_15m == "BULLISH" and trend_1h == "BULLISH":
                score = 15
            # 5M + 15M BULLISH (good for scalping)
            elif trend_5m == "BULLISH" and trend_15m == "BULLISH":
                score = 10
            # 5M + 1H BULLISH (momentum with higher trend)
            elif trend_5m == "BULLISH" and trend_1h == "BULLISH":
                score = 8
            # 5M BULLISH only (minor positive)
            elif trend_5m == "BULLISH":
                score = 5
            # ALL BEARISH - STRONGEST NEGATIVE
            elif trend_5m == "BEARISH" and trend_15m == "BEARISH" and trend_1h == "BEARISH":
                score = -10
            # 5M + 15M BEARISH
            elif trend_5m == "BEARISH" and trend_15m == "BEARISH":
                score = -5
            # Mixed or sideways
            else:
                score = 0
            
            self.market.safe_print(
                f"MTF Debug | "
                f"5M={trend_5m} | "
                f"15M={trend_15m} | "
                f"1H={trend_1h} | "
                f"Score={score}"
            )
            
            return (score, trend_5m, trend_15m, trend_1h)
            
        except Exception as e:
            self.market.safe_print(f"MTF Error: {e}")
            return (0, "UNKNOWN", "UNKNOWN", "UNKNOWN")

    def get_market_structure_score(self, candles):
        try:
            highs = pd.Series([float(c["high"]) for c in candles])
            lows = pd.Series([float(c["low"]) for c in candles])
            
            recent_high = max(float(c["high"]) for c in candles[-20:])
            recent_low = min(lows[-20:])
            current_price = float(candles[-1]["close"])
            
            price_range = recent_high - recent_low
            
            if price_range <= 0:
                return 0
            
            range_position = (current_price - recent_low) / price_range
            
            # MORE GENEROUS STRUCTURE SCORING
            if range_position < 0.25:  # Strong support bounce
                score = 15
            elif range_position < 0.40:  # Support zone
                score = 10
            elif range_position > 0.75:  # Resistance breakout
                score = 8
            elif range_position > 0.60:  # Near resistance
                score = 5
            else:
                score = 0
            
            self.market.safe_print(f"Structure Debug | Position={range_position:.2f} | Score={score}")
            return score
            
        except Exception as e:
            self.market.safe_print(f"Structure Error: {e}")
            return 0

    def process_market(self, ticker):
        try:
            symbol = ticker["market"]
            
            self.market.safe_print(f"\nProcessing {symbol}")
            
            candles_5m = self.market.get_candles(symbol, interval="5m", limit=100)
            candles_15m = self.market.get_candles(symbol, interval="15m", limit=100)
            candles_1h = self.market.get_candles(symbol, interval="1h", limit=100)
            
            if not candles_5m:
                self.market.safe_print(f"❌ No candles: {symbol}")
                return None
            
            # TREND
            trend, trend_score = self.get_trend_score(candles_5m)
            
            # RSI
            rsi, rsi_score = self.get_rsi_reversal_score(candles_5m)
            
            # VOLUME
            volume_spike, volume_score = self.get_volume_spike(candles_5m)
            
            # MOMENTUM
            momentum_score = self.get_momentum_score(candles_5m)
            
            # STRUCTURE
            structure_score = self.get_market_structure_score(candles_5m)
            
            # RISK REWARD
            rr, entry, stop_loss, target, rr_score = self.get_risk_reward(candles_5m)
            
            # MTF
            mtf_score, trend_5m, trend_15m, trend_1h = self.get_multi_timeframe_score(
                candles_5m, candles_15m, candles_1h
            )
            
            change_percent = abs(float(ticker.get("change_24_hour", 0)))
            
            # Avoid pump coins
            if change_percent > 12:
                self.market.safe_print(f"❌ PUMP FILTER: {change_percent}%")
                return None
            
            # OVERSOLD BONUS (only for bullish trend)
            oversold_bonus = 0
            if trend == "BULLISH":
                if rsi < 30:
                    oversold_bonus = 10
                elif rsi < 40:
                    oversold_bonus = 5
            
            # FINAL SCORE CALCULATION
            total_score = (
                rsi_score +
                volume_score +
                momentum_score +
                structure_score +
                rr_score +
                mtf_score +
                oversold_bonus
            )
            
            # Detailed scoring breakdown
            self.market.safe_print(
                f"SCORE BREAKDOWN:\n"
                f"  RSI: {rsi_score}\n"
                f"  Momentum: {momentum_score}\n"
                f"  Volume: {volume_score}\n"
                f"  Structure: {structure_score}\n"
                f"  RR: {rr_score}\n"
                f"  MTF: {mtf_score}\n"
                f"  Oversold Bonus: {oversold_bonus}\n"
                f"  TOTAL: {total_score}"
            )
            
            self.market.safe_print(
                f"DEBUG | "
                f"Score={total_score} | "
                f"Trend={trend} | "
                f"RSI={round(rsi, 2)} | "
                f"RSIScore={rsi_score} | "
                f"Volume={volume_score} | "
                f"Momentum={momentum_score} | "
                f"Structure={structure_score} | "
                f"RR={round(rr,2)} | RRScore={rr_score} | "
                f"MTF={mtf_score} | "
                f"15M={trend_15m} | 1H={trend_1h}"
            )
            
            # SIGNAL LOGIC
            signal = self.calculate_signal(
                total_score=total_score,
                trend=trend,
                trend_score=trend_score,
                rsi=rsi,
                volume_score=volume_score,
                momentum_score=momentum_score,
                rr_score=rr_score,
                structure_score=structure_score,
                change_percent=change_percent,
            )
            
            return {
                "symbol": symbol,
                "signal": signal,
                "trend": trend,
                "score": total_score,
                "rsi": rsi,
                "volume_spike": volume_spike,
                "price": float(ticker["last_price"]),
                "change": float(ticker["change_24_hour"]),
                "volume": float(ticker["volume"]),
                "entry": round(entry, 8),
                "stop_loss": round(stop_loss, 8),
                "target": round(target, 8),
                "risk_reward": rr,
            }
            
        except Exception as e:
            self.market.safe_print(f"Process Error ({ticker.get('market')}): {e}")
            traceback.print_exc()
            return None

    def calculate_signal(self, total_score, trend, trend_score, rsi, volume_score, 
                        momentum_score, rr_score, structure_score, change_percent):
        
        # ========== GLOBAL HARD FILTERS ==========
        if change_percent > 12:
            signal = "IGNORE"
            self._log_signal_debug(trend, total_score, total_score, signal, rsi, structure_score, momentum_score, volume_score)
            return "IGNORE"
        
        if rr_score < -25:
            signal = "IGNORE"
            self._log_signal_debug(trend, total_score, total_score, signal, rsi, structure_score, momentum_score, volume_score)
            return "IGNORE"
        
        # ========== TREND-SPECIFIC ENGINES ==========
        effective_score = total_score  # Never mutate original
        
        # ENGINE A: BULLISH - Continuation Scalp
        if trend == "BULLISH":
            if rsi > 70:
                signal = "IGNORE"
            elif effective_score >= 35:
                signal = "STRONG BUY"
            elif effective_score >= 22:
                signal = "BUY"
            elif effective_score >= 15:
                signal = "WATCHLIST"
            else:
                signal = "IGNORE"
            
            self._log_signal_debug(trend, total_score, effective_score, signal, rsi, structure_score, momentum_score, volume_score)
            return signal
        
        # ENGINE B: BEARISH - Oversold Bounce / Reversal Scalp
        elif trend == "BEARISH":
            # Must be oversold enough for reversal
            if rsi > 52:
                signal = "IGNORE"
                self._log_signal_debug(trend, total_score, effective_score, signal, rsi, structure_score, momentum_score, volume_score)
                return "IGNORE"
            
            # Structure gate for weak setups
            if structure_score <= 0 and effective_score < 25:
                signal = "IGNORE"
                self._log_signal_debug(trend, total_score, effective_score, signal, rsi, structure_score, momentum_score, volume_score)
                return "IGNORE"
            
            # Small momentum bonus (max +3)
            if momentum_score > 0:
                effective_score = min(effective_score + 3, effective_score + momentum_score * 0.3)
            
            # Decision
            if effective_score >= 38:
                signal = "REVERSAL BUY"
            elif effective_score >= 22:
                signal = "WATCHLIST"
            else:
                signal = "IGNORE"
            
            self._log_signal_debug(trend, total_score, effective_score, signal, rsi, structure_score, momentum_score, volume_score)
            return signal
        
        # ENGINE C: SIDEWAYS - Range Bounce Scalp
        else:  # SIDEWAYS
            # Require at least one confirmation
            has_confirmation = (volume_score > 0 or momentum_score > 0 or structure_score > 0)
            
            if not has_confirmation and effective_score < 28:
                signal = "IGNORE"
            elif effective_score >= 28:
                signal = "BUY"
            elif effective_score >= 18:
                signal = "WATCHLIST"
            else:
                signal = "IGNORE"
            
            self._log_signal_debug(trend, total_score, effective_score, signal, rsi, structure_score, momentum_score, volume_score)
            return signal

    def _log_signal_debug(self, trend, original_score, effective_score, signal, rsi, structure_score, momentum_score, volume_score):
        """Helper method for consistent debug logging"""
        with self.print_lock:
            print(
                f"SIGNAL DEBUG | "
                f"Trend={trend} | "
                f"Score={original_score} | "
                f"Effective={effective_score:.1f} | "
                f"RSI={rsi:.1f} | "
                f"Structure={structure_score} | "
                f"Momentum={momentum_score} | "
                f"Volume={volume_score} | "
                f"Signal={signal}"
            )

    def calculate_macd(self,candles,fast_period=12,slow_period=26,signal_period=9,):

        try:
            closes = pd.Series([float(c["close"]) for c in candles])

            if len(closes) < slow_period:
                return None, None, None

            ema_fast = closes.ewm(span=fast_period, adjust=False).mean()

            ema_slow = closes.ewm(span=slow_period, adjust=False).mean()

            macd_line = ema_fast - ema_slow

            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            histogram = macd_line - signal_line

            return (
                macd_line,
                signal_line,
                histogram,
            )

        except Exception as e:
            self.market.safe_print(f"MACD ERROR: {e}")
            return None, None, None
    
    def calculate_rr_score(rr):

        if rr >= 3:
            return 15
        elif rr >= 2:
            return 10
        elif rr >= 1.2:
            return 5
        elif rr >= 0.8:
            return 0
        elif rr >= 0.5:
            return -3
        else:
            return -8

    def get_risk_reward(self, candles):

        try:
            if len(candles) < 20:
                return 0, 0, 0, 0, 0

            highs = [float(c["high"]) for c in candles]
            lows = [float(c["low"]) for c in candles]
            closes = [float(c["close"]) for c in candles]

            entry = closes[-1]

            recent_low = min(lows[-20:])
            recent_high = max(highs[-20:])

            stop_loss = recent_low
            target = recent_high

            risk = abs(entry - stop_loss)
            reward = abs(target - entry)

            if risk == 0:
                return 0, entry, stop_loss, target, 0

            rr = reward / risk
            rr = round(rr, 2)

            score = 0

            if rr < 0.7:
                return rr, entry, stop_loss, target, -20
            if rr >= 5:
                score = 6
            elif rr >= 3:
                score = 4
            elif rr >= 2:
                score = 2
            else:
                score = -5

            return (
                rr,
                round(entry, 6),
                round(stop_loss, 6),
                round(target, 6),
                score,
            )

        except Exception as e:
            self.market.safe_print(f"RR Error: {e}")
            return 0, 0, 0, 0, 0

    def get_usdt_markets(self):

        markets = self.market.get_available_markets()
        return [m for m in markets if (m.endswith("USDT") and "BTC" not in m)]

    # SCORE MARKET
    def score_market(self, ticker, candles, rsi, volume_spike):

        score = 0
        trend_score = self.get_trend_score(candles)

        score += trend_score

        # RSI

        if rsi is not None and rsi < 25:
            score += 40
        elif rsi is not None and rsi < 35:
            score += 25
        elif rsi is not None and rsi < 45:
            score += 10
        elif rsi is not None and rsi > 75:
            score -= 25

        # Volume Spike

        if volume_spike > 3:
            score += 40
        elif volume_spike > 2:
            score += 25
        elif volume_spike > 1.5:
            score += 10
        elif volume_spike < 0.8:
            score -= 15

        # 24h change
        change = abs(float(ticker.get("change_24_hour", 0)))

        if 2 < change < 8:
            score += 15
        elif 8 < change < 15:
            score += 5
        elif change > 20:
            score -= 20

        # Liquidity

        volume_24h = float(ticker.get("volume", 0))

        if volume_24h > 50_000_000:
            score += 20

        elif volume_24h > 10_000_000:
            score += 10

        score = max(0, min(score, 100))

        return round(score)

    def get_trend_score(self, candles):

        try:
            closes = [float(candle["close"]) for candle in candles]

            if len(closes) < 50:
                return "SIDEWAYS", 0

            series = pd.Series(closes)

            ema20 = series.ewm(span=20).mean().iloc[-1]
            ema50 = series.ewm(span=50).mean().iloc[-1]

            current_price = closes[-1]

            ema_gap = ((ema20 - ema50) / ema50) * 100

            trend = "SIDEWAYS"
            score = 0

            # Bullish trend
            if current_price > ema20 > ema50:

                trend = "BULLISH"

                if ema_gap > 3:
                    score = 20
                elif ema_gap > 1:
                    score = 10
                else:
                    score = 5

            # Bearish trend
            # Bearish trend
            elif current_price < ema20 < ema50:

                trend = "BEARISH"

                if ema_gap < -3:
                    score = 20
                elif ema_gap < -1:
                    score = 10
                else:
                    score = 5

            if trend == "SIDEWAYS":
                return "SIDEWAYS", 0

            return trend, score

        except Exception as e:
            self.market.safe_print(f"Trend Error: {e}")
            return "SIDEWAYS", 0

    # RUN SCAN
    def get_ema_trend(self, candles):

        try:
            df = pd.DataFrame(candles)
            closes = df["close"].astype(float)
            ema20 = closes.ewm(span=20).mean()
            ema50 = closes.ewm(span=50).mean()
            current_price = closes.iloc[-1]

            if current_price > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
                return "BULLISH"

            elif current_price < ema20.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
                return "BEARISH"

            return "SIDEWAYS"

        except:
            return "UNKNOWN"

    def scan_markets(self, top_n=10):
        try:
            self.market.safe_print("\n🔍 Scanning markets...")

            tickers = self.market.get_all_tickers()

            usdt_markets = []

            self.market.safe_print("\n===== MARKET CHECK =====")

            markets = self.market._make_public_request("/exchange/v1/markets_details")
            market_pair_map = {}

            for market in markets:

                symbol = market.get("symbol")
                pair = market.get("pair")

                if symbol in DEAD_CANDLE_MARKETS:
                    continue

                if symbol and pair:
                    market_pair_map[symbol] = pair

            if not markets:
                self.market.safe_print("❌ No markets found")
                return

            target_symbols = [
                "TAOUSDT",
                "TONUSDT",
                "HYPEUSDT",
                "ASTERUSDT",
                "UBUSDT",
            ]

            for market in markets:

                symbol = market.get("symbol", "")
                pair = market.get("pair", "")
                coindcx_name = market.get("coindcx_name", "")

                if symbol in target_symbols:

                    self.market.safe_print(
                        f"symbol={symbol} | "
                        f"pair={pair} | "
                        f"coindcx_name={coindcx_name}"
                    )

            self.market.safe_print("\n")

            for ticker in tickers:

                symbol = ticker.get("market", "")

                if symbol in DEAD_CANDLE_MARKETS:
                    continue

                stablecoin_pairs = {
                    "FDUSDUSDT",
                    "USDCUSDT",
                    "RLUSDUSDT",
                    "XUSDUSDT",
                    "UUSDT",
                }

                if symbol in stablecoin_pairs:
                    continue

                if not symbol.endswith("USDT"):
                    continue

                volume = float(ticker.get("volume", 0))

                change = abs(float(ticker.get("change_24_hour", 0)))
                if change > 12:
                    continue

                if volume < 8_000_000:
                    continue
                price = float(ticker.get("last_price", 0))

                if price < 0.01:
                    continue

                blacklist = ["1000CHEEMS", "BANANAS31", "GIGGLE", "MEGA", "PENGU"]

                if any(x in symbol for x in blacklist):
                    continue

                fast_score = 0

                if volume > 10_000_000:
                    fast_score += 30

                if change > 2:
                    fast_score += 20
                if change < 0.2:
                    continue
                if change > 12:
                    continue
                if change > 5:
                    fast_score += 10
                """ if change > 8:
                    continue for 5M scalping """
                if 6 < change < 15:
                    fast_score -= 20  # avoid pumps

                usdt_markets.append({"ticker": ticker, "score": fast_score})

            usdt_markets = sorted(usdt_markets, key=lambda x: x["score"], reverse=True)

            candidate_markets = [x["ticker"] for x in usdt_markets[:40]]
            self.market.safe_print(
                f"⚡ RSI scan on " f"{len(candidate_markets)} " f"markets..."
            )

            scored = []
            with ThreadPoolExecutor(max_workers=6) as executor:

                futures = {
                    executor.submit(self.process_market, ticker): ticker
                    for ticker in candidate_markets
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        scored.append(result)

            ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
            self.market.safe_print("\n🏆 TOP MARKETS")
            self.market.safe_print("=" * 100)
            for i, item in enumerate(ranked[:top_n], start=1):

                self.market.safe_print(
                    f"{i:02}. "
                    f"{item['symbol']:12}"
                    f" | Signal= "
                    f"{item['signal']:10}"
                    f" | Trend="
                    f"{item['trend']:10}"
                    f" | Score="
                    f"{item['score']:3}"
                    f" | RSI="
                    f"{item['rsi']}"
                    f" | VolSpike="
                    f"{item['volume_spike']}x"
                    f" | Price="
                    f"{item['price']}"
                    f" | Change="
                    f"{item['change']}%"
                    f" | Volume="
                    f"{float(item['volume']):,.0f}"
                )
        except KeyboardInterrupt:
            self.market.safe_print("\n👋 Bot stopped safely.")