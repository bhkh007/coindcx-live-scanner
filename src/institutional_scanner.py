# institutional_scanner.py
from datetime import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from market_data import CoinDCXMarketData
import config

_GLOBAL_SIGNAL_HISTORY = {}
_GLOBAL_CONFLUENCE_HISTORY = {}
_HISTORY_FILE = Path(__file__).parent.parent / "signal_history.json"

class MarketRegime(Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGE = "RANGE"
    REVERSAL_POTENTIAL = "REVERSAL_POTENTIAL"

class Signal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY_WATCHLIST = "BUY_WATCHLIST"
    STRONG_SELL = "STRONG_SELL"
    SELL_WATCHLIST = "SELL_WATCHLIST"
    REVERSAL_BUY = "REVERSAL_BUY"
    IGNORE = "IGNORE"
    ELITE_SETUP = "ELITE_SETUP"
    HIGH_CONVICTION = "HIGH_CONVICTION"
    GOOD = "GOOD"
    WATCHLIST = "WATCHLIST"
    STRONG = "STRONG"

@dataclass
class MarketAnalysis:
    symbol: str
    regime: MarketRegime
    trend_alignment_score: float
    structure_score: float
    liquidity_score: float
    momentum_score: float
    rr_score: float
    entry_score: float
    rsi_context_score: float
    confidence: float
    signal: Signal
    details: Dict

class InstitutionalScanner:
    def __init__(self):
        self.market = CoinDCXMarketData()
        self.print_lock = threading.Lock()
        self._signal_streak = {}  # Track consecutive signals
        self._min_streak_required = 2  # Need 2 scans for promotion
        self._history_file = "signal_history.json"
        self._load_history()
        self._history_expiry_seconds = 3600  # 1 hour expiry
        
        # Use global histories for persistence across scans
        self._signal_history = _GLOBAL_SIGNAL_HISTORY
        self._confluence_history = _GLOBAL_CONFLUENCE_HISTORY
        
    def _log_historical_signal(self, symbol: str, signal: Signal, confidence: float, 
                                regime: MarketRegime, base_score: float = None, 
                                confluence_pct: float = None, multiplier: float = None):
            """Log signal to historical file for later analysis"""
            
            log_entry = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "symbol": symbol,
                "signal": signal.value,
                "confidence": round(confidence, 2),
                "regime": regime.value,
                "base_score": base_score,
                "confluence_pct": confluence_pct,
                "multiplier": multiplier
            }
            
            try:
                with open(self.historical_log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                pass  # Don't let logging break scanning
        
    def _load_history(self):
        """Load persistent signal history from disk"""
        try:
            if _HISTORY_FILE.exists():
                with open(_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    # Update global dictionaries
                    _GLOBAL_SIGNAL_HISTORY.update(data.get('signals', {}))
                    _GLOBAL_CONFLUENCE_HISTORY.update(data.get('confluence', {}))
                    # Also update instance references
                    self._signal_history = _GLOBAL_SIGNAL_HISTORY
                    self._confluence_history = _GLOBAL_CONFLUENCE_HISTORY
                print(f"✅ Loaded {len(self._signal_history)} signal records from history")
        except Exception as e:
            print(f"⚠️ Could not load history: {e}")
    
    def _save_history(self):
        """Save persistent signal history to disk"""
        try:
            data = {
                'signals': self._signal_history,
                'confluence': self._confluence_history,
                'last_updated': time.time()
            }
            with open(_HISTORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save history: {e}")    
        
    def analyze_market_regime(self, candles_4h: List, candles_1h: List) -> Tuple[MarketRegime, Dict]:
        """Weighted institutional regime detection"""
        
        if len(candles_4h) < 50:
            return MarketRegime.RANGE, {"error": "insufficient data"}
        
        closes_4h = [float(c["close"]) for c in candles_4h]
        highs_4h = [float(c["high"]) for c in candles_4h]
        lows_4h = [float(c["low"]) for c in candles_4h]
        
        # Calculate indicators
        df = pd.DataFrame({"close": closes_4h, "high": highs_4h, "low": lows_4h})
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        
        current_price = closes_4h[-1]
        ema20 = df["ema20"].iloc[-1]
        ema50 = df["ema50"].iloc[-1]
        
        # Detect swing points
        swing_highs = []
        swing_lows = []
        for i in range(2, len(highs_4h)-2):
            if highs_4h[i] > highs_4h[i-1] and highs_4h[i] > highs_4h[i-2] and \
            highs_4h[i] > highs_4h[i+1] and highs_4h[i] > highs_4h[i+2]:
                swing_highs.append(highs_4h[i])
            if lows_4h[i] < lows_4h[i-1] and lows_4h[i] < lows_4h[i-2] and \
            lows_4h[i] < lows_4h[i+1] and lows_4h[i] < lows_4h[i+2]:
                swing_lows.append(lows_4h[i])
        
        # Calculate trend scores
        bullish_score = 0
        bearish_score = 0
        
        # EMA alignment
        if ema20 > ema50:
            bullish_score += 20
        else:
            bearish_score += 20
        
        # Price vs EMA
        if current_price > ema20:
            bullish_score += 20
        else:
            bearish_score += 20
        
        # Structure (HH/HL vs LH/LL)
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            hh_hl = swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]
            lh_ll = swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]
            
            if hh_hl:
                bullish_score += 20
            elif lh_ll:
                bearish_score += 20
        
        # Momentum (recent candles)
        recent_closes = closes_4h[-5:]
        momentum_up = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i-1])
        if momentum_up >= 4:
            bullish_score += 20
        elif momentum_up <= 1:
            bearish_score += 20
        
        # Volume trend
        volumes = [float(c["volume"]) for c in candles_4h[-10:]]
        volume_trend = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
        if volume_trend >= 7:
            if bullish_score > bearish_score:
                bullish_score += 20
            else:
                bearish_score += 20
        
        # Calculate ADX for trend strength
        adx = self._calculate_adx(candles_4h)
        
        # Determine regime
        total_score = max(bullish_score, bearish_score)
        is_bullish = bullish_score > bearish_score
        strength = min(100, total_score)
        
        regime_details = {
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "adx": round(adx, 1),
            "strength": strength
        }
        
        # Regime classification
        if is_bullish and strength >= 60 and adx > 25:
            regime = MarketRegime.TRENDING_BULLISH
            regime_details["reason"] = f"EMA bullish, HH/HL, strength={strength}"
        elif not is_bullish and strength >= 60 and adx > 25:
            regime = MarketRegime.TRENDING_BEARISH
            regime_details["reason"] = f"EMA bearish, LH/LL, strength={strength}"
        elif strength >= 40 and adx < 25:
            # Check for reversal conditions
            rsi = self._calculate_rsi(closes_4h)
            if (rsi < 30 and not is_bullish) or (rsi > 70 and is_bullish):
                regime = MarketRegime.REVERSAL_POTENTIAL
                regime_details["reason"] = f"Trend exhaustion, RSI={round(rsi,1)}"
            else:
                regime = MarketRegime.RANGE
                regime_details["reason"] = f"Weak trend, ADX={round(adx,1)}"
        else:
            regime = MarketRegime.RANGE
            regime_details["reason"] = f"No clear trend, strength={strength}"
            
        # Calculate regime confidence
        regime_confidence = 0
        if regime == MarketRegime.TRENDING_BEARISH:
            if adx > 40 and strength > 80:
                regime_confidence = 90
            elif adx > 30 and strength > 60:
                regime_confidence = 75
            elif adx > 25:
                regime_confidence = 60
            else:
                regime_confidence = 45
        elif regime == MarketRegime.TRENDING_BULLISH:
            # Same logic
            if adx > 40 and strength > 80:
                regime_confidence = 90
            elif adx > 30 and strength > 60:
                regime_confidence = 75
            else:
                regime_confidence = 45
        else:
            regime_confidence = 40

        regime_details["regime_confidence"] = regime_confidence
        
        self._log_debug("REGIME", regime_details)
        return regime, regime_details

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Simple RSI calculation for regime detection"""
        if len(closes) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_adx(self, candles: List, period: int = 14) -> float:
        """Calculate ADX for trend strength"""
        if len(candles) < period + 1:
            return 0
        
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        closes = [float(c["close"]) for c in candles]
        
        plus_dm = []
        minus_dm = []
        tr = []
        
        for i in range(1, len(candles)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
            else:
                plus_dm.append(0)
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm.append(low_diff)
            else:
                minus_dm.append(0)
            
            tr_value = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr.append(tr_value)
        
        if len(tr) < period:
            return 0
        
        atr = sum(tr[-period:]) / period
        
        plus_di = 100 * (sum(plus_dm[-period:]) / period) / atr if atr > 0 else 0
        minus_di = 100 * (sum(minus_dm[-period:]) / period) / atr if atr > 0 else 0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        return dx

    def _calculate_atr_expansion(self, candles: List) -> float:
        """Check if ATR is expanding (increased volatility)"""
        if len(candles) < 20:
            return 1.0
        
        current_atr = self._calculate_atr(candles[-14:])
        prev_atr = self._calculate_atr(candles[-28:-14])
        
        if prev_atr == 0:
            return 1.0
        
        return current_atr / prev_atr
    
    def analyze_mtf_alignment(self, analysis_5m: Dict, analysis_15m: Dict, 
                          analysis_1h: Dict, analysis_4h: Dict) -> Tuple[float, Dict]:
        """Multi-timeframe alignment - trending markets should score HIGH when aligned"""
        
        # Convert to numeric (1 = bullish, 0 = neutral, -1 = bearish)
        def get_direction(analysis):
            if analysis.get("bullish"):
                return 1
            elif analysis.get("bearish"):
                return -1
            return 0
        
        dir_5m = get_direction(analysis_5m)
        dir_15m = get_direction(analysis_15m)
        dir_1h = get_direction(analysis_1h)
        dir_4h = get_direction(analysis_4h)
        
        # Check if all timeframes agree on direction
        all_bullish = all(d == 1 for d in [dir_5m, dir_15m, dir_1h, dir_4h])
        all_bearish = all(d == -1 for d in [dir_5m, dir_15m, dir_1h, dir_4h])
        
        # Count aligned timeframes
        if all_bullish or all_bearish:
            mtf_score = 100  # Perfect alignment
            alignment = "FULL_ALIGNMENT"
        elif (dir_4h == dir_1h == dir_15m) and dir_5m != 0:
            mtf_score = 85  # HTF aligned, LTF may vary
            alignment = "STRONG_ALIGNMENT"
        elif dir_4h == dir_1h and dir_15m != 0:
            mtf_score = 70  # Higher timeframes aligned
            alignment = "GOOD_ALIGNMENT"
        elif dir_4h != 0 and (dir_1h == dir_4h or dir_15m == dir_4h):
            mtf_score = 50  # Mixed but HTF has direction
            alignment = "MIXED"
        elif dir_4h == 0:
            mtf_score = 30  # No clear HTF direction
            alignment = "WEAK"
        else:
            # Conflicting directions (4H up, 1H down)
            mtf_score = 15
            alignment = "CONFLICTING"
        
        mtf_details = {
            "directions": {"5M": dir_5m, "15M": dir_15m, "1H": dir_1h, "4H": dir_4h},
            "alignment": alignment,
            "final_score": mtf_score
        }
        
        self._log_debug("MTF", mtf_details)
        return mtf_score, mtf_details
    
    def analyze_structure_quality(self, candles: List, current_price: float, regime: MarketRegime = None) -> Tuple[float, Dict]:
        """Pure structure scoring - fixed bearish detection"""
        
        if len(candles) < 30:
            return 30, {"error": "insufficient data"}  # Neutral baseline, not 0
        
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        closes = [float(c["close"]) for c in candles]
        
        # Calculate EMAs
        df = pd.DataFrame({"close": closes})
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
            highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
            lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        # Start with neutral baseline
        structure_score = 30
        details = {"trend_structure": "NEUTRAL"}
        
        # Universal structure detection
        # In analyze_structure_quality method, for TRENDING_BEARISH:
        if regime == MarketRegime.TRENDING_BEARISH:
            # Calculate TREND STRENGTH first
            closes = [float(c["close"]) for c in candles]
            df = pd.DataFrame({"close": closes})
            df["ema20"] = df["close"].ewm(span=20).mean()
            df["ema50"] = df["close"].ewm(span=50).mean()
            
            # Calculate EMA slope (momentum of trend)
            ema_slope = (df["ema20"].iloc[-1] - df["ema20"].iloc[-5]) / df["ema20"].iloc[-5] * 100
            
            # Calculate price distance below EMA
            price_distance = ((df["ema20"].iloc[-1] - current_price) / current_price) * 100
            
            # DYNAMIC BASE SCORE based on trend strength
            if ema_slope < -1.0 and price_distance > 1.0:
                structure_score = 75  # Strong bearish trend
                trend_strength = "STRONG"
            elif ema_slope < -0.5 or price_distance > 0.5:
                structure_score = 70  # Moderate bearish trend
                trend_strength = "MODERATE"
            else:
                structure_score = 65  # Weak but confirmed bearish
                trend_strength = "WEAK"
            
            # Add confirmation bonuses (same as before)
            confirmations = []
            if len(swing_highs) >= 2 and swing_highs[-1] < swing_highs[-2]:
                structure_score += 5  # Reduced from 10 to prevent over-scoring
                confirmations.append("lower_highs")
            
            if len(swing_lows) >= 2 and swing_lows[-1] < swing_lows[-2]:
                structure_score += 5
                confirmations.append("lower_lows")
            
            if closes[-1] < df["ema20"].iloc[-1] < df["ema50"].iloc[-1]:
                structure_score += 5
                confirmations.append("ema_alignment")
            
            # Cap at 95 (leave room for exceptional cases)
            structure_score = min(95, structure_score)
            
            details = {
                "trend_strength": trend_strength,
                "ema_slope": round(ema_slope, 2),
                "price_distance": round(price_distance, 2),
                "confirmations": confirmations,
                "score": structure_score
            }

        # For TRENDING_BULLISH:
        elif regime == MarketRegime.TRENDING_BULLISH:
            # Calculate TREND STRENGTH first
            closes = [float(c["close"]) for c in candles]
            df = pd.DataFrame({"close": closes})
            df["ema20"] = df["close"].ewm(span=20).mean()
            df["ema50"] = df["close"].ewm(span=50).mean()
            
            # Calculate EMA slope (momentum of trend)
            ema_slope = (df["ema20"].iloc[-1] - df["ema20"].iloc[-5]) / df["ema20"].iloc[-5] * 100
            
            # Calculate price distance ABOVE EMA (FIXED)
            price_distance = ((current_price - df["ema20"].iloc[-1]) / current_price) * 100
            
            # DYNAMIC BASE SCORE based on trend strength
            if ema_slope > 1.0 and price_distance > 1.0:  # Strong bullish
                structure_score = 75
                trend_strength = "STRONG"
            elif ema_slope > 0.5 or price_distance > 0.5:  # Moderate bullish
                structure_score = 70
                trend_strength = "MODERATE"
            else:
                structure_score = 65  # Weak but confirmed bullish
                trend_strength = "WEAK"
            
            # Add confirmation bonuses for BULLISH
            confirmations = []
            if len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]:  # Higher highs
                structure_score += 5
                confirmations.append("higher_highs")
            
            if len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]:  # Higher lows
                structure_score += 5
                confirmations.append("higher_lows")
            
            if closes[-1] > df["ema20"].iloc[-1] > df["ema50"].iloc[-1]:  # EMA bullish alignment
                structure_score += 5
                confirmations.append("ema_alignment")
            
            if closes[-1] > df["ema20"].iloc[-1]:  # Price above EMA
                structure_score += 5
                confirmations.append("price_above_ema")
            
            structure_score = min(95, structure_score)
            
            details = {
                "trend_strength": trend_strength,
                "ema_slope": round(ema_slope, 2),
                "price_distance": round(price_distance, 2),
                "confirmations": confirmations,
                "score": structure_score
            }
        
        elif regime == MarketRegime.REVERSAL_POTENTIAL:
            # Reversal structure detection
            recent_low = min(lows[-10:])
            recent_high = max(highs[-10:])
            
            # Support bounce
            if closes[-1] > recent_low * 1.01:
                structure_score += 25
                details["support_bounce"] = True
            
            # Displacement candle
            avg_body = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes))) / (len(closes)-1)
            current_body = abs(closes[-1] - closes[-2])
            if current_body > avg_body * 1.5:
                structure_score += 25
                details["displacement"] = True
            
            # Volume spike
            volumes = [float(c["volume"]) for c in candles[-5:]]
            if len(volumes) >= 2 and volumes[-1] > volumes[-2] * 1.5:
                structure_score += 25
                details["volume_spike"] = True
            
            details["trend_structure"] = "REVERSAL"
        
        # Strict clamping
        structure_score = min(100, max(0, structure_score))
        details["score"] = structure_score
        
        return structure_score, details
    
    def calculate_regime_confidence(self, regime: MarketRegime, regime_details: Dict) -> float:
        """Calculate confidence in regime classification (0-100)"""
        
        adx = regime_details.get("adx", 0)
        strength = regime_details.get("strength", 0)
        
        if regime == MarketRegime.TRENDING_BULLISH:
            if adx > 40 and strength > 80:
                return 90
            elif adx > 30 and strength > 60:
                return 75
            elif adx > 25:
                return 60
            else:
                return 45
        
        elif regime == MarketRegime.TRENDING_BEARISH:
            if adx > 40 and strength > 80:
                return 90
            elif adx > 30 and strength > 60:
                return 75
            elif adx > 25:
                return 60
            else:
                return 45
        
        elif regime == MarketRegime.REVERSAL_POTENTIAL:
            return 55
        
        else:  # RANGE
            return 40
    
    def analyze_liquidity(self, candles: List, ticker: Dict) -> Tuple[float, Dict]:
        """STEP 4: Liquidity filter - volume spike, 24h volume"""
        
        volumes = [float(c["volume"]) for c in candles]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:-1]) if len(volumes) > 20 else np.mean(volumes[:-1])
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Volume spike score
        if volume_ratio >= config.LIQUIDITY["excellent_volume_ratio"]:
            volume_score = 100
        elif volume_ratio >= config.LIQUIDITY["good_volume_ratio"]:
            volume_score = 70
        elif volume_ratio >= config.LIQUIDITY["neutral_volume_ratio"]:
            volume_score = 50
        else:
            volume_score = 20
        
        # 24h volume minimum
        volume_24h = float(ticker.get("volume", 0))
        if volume_24h < config.LIQUIDITY["min_24h_volume_usdt"]:
            volume_score *= 0.5
        
        details = {
            "volume_ratio": round(volume_ratio, 2),
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "volume_24h": volume_24h,
            "score": volume_score
        }
        
        self._log_debug("LIQUIDITY", details)
        return volume_score, details
    
    def analyze_momentum_displacement(self, candles: List, volume_score: float) -> Tuple[float, Dict]:
        """Institutional momentum engine - never exceeds 100"""
        
        if len(candles) < 20:
            return 0, {"error": "insufficient data"}
        
        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        volumes = [float(c["volume"]) for c in candles]
        
        # Candle body expansion
        bodies = [abs(closes[i] - candles[i]["open"]) for i in range(len(candles))]
        avg_body = sum(bodies[-20:-1]) / 19 if len(bodies) > 20 else sum(bodies) / len(bodies)
        current_body = bodies[-1]
        body_ratio = current_body / avg_body if avg_body > 0 else 0
        is_bullish = float(candles[-1]["close"]) > float(candles[-1]["open"])
        
        # ATR expansion
        atr = self._calculate_atr(candles[-20:])
        prev_atr = self._calculate_atr(candles[-40:-20]) if len(candles) >= 40 else atr
        atr_expansion = atr / prev_atr if prev_atr > 0 else 1
        
        # Volume confirmation
        avg_volume = sum(volumes[-20:-1]) / 19
        volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 0
        volume_confirmed = volume_score >= 50 or volume_spike >= 1.3
        
        # Impulse strength
        impulse = False
        if body_ratio >= 1.5 and volume_confirmed:
            impulse = True
        
        # Multi-candle continuation
        continuation = False
        if len(closes) >= 5:
            direction = 1 if closes[-1] > closes[-2] else -1
            consecutive = sum(1 for i in range(-4, 0) if (closes[i] - closes[i-1]) * direction > 0)
            if consecutive >= 3:
                continuation = True
        
        # Replace the score calculation section:
        score = 0

        # Body ratio (max 40)
        if body_ratio >= 2.0:
            score += 35
        elif body_ratio >= 1.5:
            score += 25
        elif body_ratio >= 1.2:
            score += 15
        elif body_ratio >= 1.0:
            score += 8

        # Volume confirmation (max 30)
        if volume_confirmed:
            score += 25
        elif volume_spike >= 1.3:
            score += 15

        # Follow-through (max 20)
        if continuation:
            score += 15

        # Close strength (max 10) - where in candle did it close?
        candle_range = highs[-1] - lows[-1]
        if candle_range > 0:
            if is_bullish:
                close_strength = (closes[-1] - lows[-1]) / candle_range
            else:
                close_strength = (highs[-1] - closes[-1]) / candle_range
            
            if close_strength > 0.7:
                score += 10
            elif close_strength > 0.6:
                score += 5

        # ATR expansion (max 5)
        if atr_expansion >= 1.2:
            score += 5

        # Cap and categorize
        score = min(100, max(0, score))

        if score >= 80:
            strength = "EXCEPTIONAL"
        elif score >= 60:
            strength = "STRONG"
        elif score >= 40:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        details = {
            "score": score,
            "strength": strength,
            "volume_confirmed": volume_confirmed,
            "body_ratio": round(body_ratio, 2),
            "atr_expansion": round(atr_expansion, 2)
        }
        
        return score, details
    
    def analyze_risk_reward(self, candles: List, entry: float, current_price: float, 
                        regime: MarketRegime,symbol: str = "", timeframe: str = "15m") -> Tuple[float, Dict]:
        """Structure-based entry calculation with ATR fallback"""
        
        if len(candles) < 20:
            return 0, {"error": "insufficient data", "rr_ratio": 0}
        
        if current_price <= 0:
            return 0, {"error": "invalid price", "rr_ratio": 0}
        
        highs = [float(c["high"]) for c in candles[-30:]]
        lows = [float(c["low"]) for c in candles[-30:]]
        closes = [float(c["close"]) for c in candles[-30:]]
        
        # Calculate ATR
        atr = self._calculate_atr(candles[-20:])
        if atr <= 0 or atr > current_price * 0.05:
            atr = current_price * 0.01
            
        raw_atr = self._calculate_atr(candles[-20:])
        if raw_atr <= 0 or raw_atr > current_price * 0.05:
            raw_atr = current_price * 0.01
        
        atr_pct = (raw_atr / current_price) * 100
        
        # HONEST: Reject low volatility assets instead of faking
        if atr_pct < 0.25:  # Less than 0.25% volatility
            self.market.safe_print(f"❌ {symbol}: Low volatility rejection (ATR {atr_pct:.2f}%)")
            return 0, {"error": "low volatility asset", "rr_ratio": 0}
        
        # Use raw ATR, no floor
        atr = raw_atr
        
        stop_loss = 0
        target = 0
        
        is_long = regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.REVERSAL_POTENTIAL]
        is_short = regime == MarketRegime.TRENDING_BEARISH
        
        if is_short:
            # Stop loss: 1.5x ATR (minimum 1%, maximum 4%)
            atr_stop = current_price + (atr * 1.5)
            min_stop = current_price * 1.01
            max_stop = current_price * 1.04
            stop_loss = max(min_stop, min(atr_stop, max_stop))
            
            # Calculate risk using the PASSED entry (not current_price)
            risk = stop_loss - entry
            
            # Get natural target
            target, natural_rr = self.calculate_natural_target(candles, entry, stop_loss, regime, is_short=True)
    
            # NOW validate target direction
            if target >= entry:
                self.market.safe_print(f"⚠️ WARNING: Short target {target:.4f} not below entry {entry:.4f}")
                risk = stop_loss - entry
                target = entry - (risk * 1.5)
                natural_rr = 2.0
            
            # Ensure target is reasonable
            recent_lows = [l for l in lows[-20:] if l < current_price]
            if recent_lows:
                nearest_support = max(recent_lows)
                if target < nearest_support:
                    target = nearest_support * 0.998
            
            reward = entry - target
            rr_ratio = natural_rr
            
            print(f"SHORT DEBUG: entry={entry:.2f}, current={current_price:.2f}, atr={atr:.2f}, sl={stop_loss:.2f}, tp={target:.2f}, risk={risk:.2f}, reward={reward:.2f}, rr={rr_ratio:.2f}")
        
        elif is_long:
            # Stop loss: 1.5x ATR (minimum 1%, maximum 4%)
            atr_stop = current_price - (atr * 1.5)
            max_stop = current_price * 0.99
            min_stop = current_price * 0.96
            stop_loss = max(min_stop, min(atr_stop, max_stop))
            
            # Calculate risk using the PASSED entry
            risk = entry - stop_loss
            
            # Get natural target
            target, natural_rr = self.calculate_natural_target(candles, entry, stop_loss, regime, is_short=False)
            
            if target <= entry:
                self.market.safe_print(f"⚠️ WARNING: Long target {target:.4f} not above entry {entry:.4f}")
                target = entry + (risk * 1.5)
                reward = target - entry
                rr_ratio = reward / risk
            
            # Ensure target is reasonable
            recent_highs = [h for h in highs[-20:] if h > current_price]
            if recent_highs:
                nearest_resistance = min(recent_highs)
                if target > nearest_resistance:
                    target = nearest_resistance * 1.002
            
            reward = target - entry
            rr_ratio = natural_rr
            
            print(f"LONG DEBUG: entry={entry:.2f}, current={current_price:.2f}, atr={atr:.2f}, sl={stop_loss:.2f}, tp={target:.2f}, risk={risk:.2f}, reward={reward:.2f}, rr={rr_ratio:.2f}")
        
        else:
            return 0, {"error": "range market", "rr_ratio": 0}
        
        # Safety checks
        if risk <= 0:
            risk = current_price * 0.01
            if is_long:
                stop_loss = entry - risk
            else:
                stop_loss = entry + risk
        
        if reward <= 0:
            if is_long:
                target = entry + risk
            else:
                target = entry - risk
            reward = risk
        
        # Validate target distance
        target_distance_pct = abs(target - entry) / entry * 100
        if target_distance_pct > 15:
            if is_long:
                target = entry + (atr * 3)
            else:
                target = entry - (atr * 3)
            reward = abs(target - entry)
            rr_ratio = reward / risk
        
        # Reject if RR below 1.5
        if rr_ratio < 1.5:
            return 0, {"rr_ratio": rr_ratio, "reason": f"Poor RR: {rr_ratio:.2f}"}
        
        # Cap at 5x
        if rr_ratio > 5.0:
            rr_ratio = 5.0
            if is_long:
                target = entry + (risk * 5.0)
            else:
                target = entry - (risk * 5.0)
            reward = risk * 5.0
        
        # Score based on RR
        if rr_ratio >= 3.0:
            rr_score = 100
        elif rr_ratio >= 2.0:
            rr_score = 70
        else:
            rr_score = 50
        
        details = {
            "entry": round(entry, 6),
            "stop_loss": round(stop_loss, 6),
            "target": round(target, 6),
            "risk": round(risk, 6),
            "reward": round(reward, 6),
            "rr_ratio": round(rr_ratio, 2),
            "score": rr_score
        }
        if entry <= 0 or entry > current_price * 2:
            entry = current_price
            print(f"⚠️ Invalid entry {entry}, using current price {current_price}")
            
        if is_short:
            actual_risk = stop_loss - entry
            actual_reward = entry - target
        else:
            actual_risk = entry - stop_loss
            actual_reward = target - entry

        # Calculate actual RR
        if actual_risk > 0:
            actual_rr = actual_reward / actual_risk
        else:
            actual_rr = 0

        # Check for mismatch
        if abs(actual_rr - rr_ratio) > 0.15:
            self.market.safe_print(f"⚠️ RR MISMATCH for {symbol}: displayed={rr_ratio:.2f}, actual={actual_rr:.2f}")
            rr_ratio = actual_rr  # Use actual RR

        # Update reward/risk with actual values
        reward = actual_reward
        risk = actual_risk

        # Reject if actual RR below 1.5
        if natural_rr < 1.5:
            return 0, {"rr_ratio": natural_rr, "reason": f"Poor natural RR: {natural_rr:.2f}"}
        
        return rr_score, details
    
    def get_market_bias_penalty(self, symbol: str, regime: MarketRegime, 
                            structure_score: float) -> float:
        """Calculate BTC bias penalty based on context"""
        
        btc_bias, btc_strength = self.get_market_bias()
        
        # Only penalize if BTC trend is STRONG
        if btc_bias == "BEARISH" and btc_strength > 0.7:
            # For alt longs in strong BTC downtrend
            if regime == MarketRegime.TRENDING_BULLISH:
                # Less penalty for high-quality setups
                if structure_score >= 80:
                    return -5  # Reduced penalty for exceptional setups
                elif structure_score >= 60:
                    return -10
                else:
                    return -15  # Full penalty for weak setups
        
        # Neutral or weak BTC bias - no penalty
        return 0
    
    def calculate_entry_price(self, candles: List, current_price: float, regime: MarketRegime, atr: float) -> Tuple[float, str, float]:
        """
        Fixed: Low volatility = LARGER pullback (need deeper retracement)
        High volatility = SMALLER pullback (volatile, might not retrace far)
        """
        
        if atr <= 0 or current_price <= 0:
            return current_price, "market price (invalid atr)", 0
        
        # Calculate ATR as percentage of price
        atr_pct = (atr / current_price) * 100
        
        # REVERSED: Low volatility needs larger pullback to get filled
        if atr_pct >= 2.0:      # High volatility (meme coins, small caps)
            pullback_multiplier = 0.25  # Smaller pullback (0.25× ATR)
            vol_note = "high vol"
        elif atr_pct >= 1.0:    # Normal volatility (ETH, SOL)
            pullback_multiplier = 0.35  # Medium pullback (0.35× ATR)
            vol_note = "normal vol"
        elif atr_pct >= 0.5:    # Low volatility (BTC)
            pullback_multiplier = 0.50  # Larger pullback (0.50× ATR)
            vol_note = "low vol"
        else:                   # Very low volatility (stablecoins, large caps in calm market)
            pullback_multiplier = 0.75  # Largest pullback (0.75× ATR)
            vol_note = "very low vol"
        
        pullback_amount = atr * pullback_multiplier
        pullback_pct = (pullback_amount / current_price) * 100
        
        if regime == MarketRegime.TRENDING_BEARISH:
            entry = current_price + pullback_amount
            direction = "+"
        elif regime == MarketRegime.TRENDING_BULLISH:
            entry = current_price - pullback_amount
            direction = "-"
        elif regime == MarketRegime.REVERSAL_POTENTIAL:
            entry = current_price
            direction = ""
            pullback_pct = 0
        else:
            entry = current_price
            direction = ""
            pullback_pct = 0
        
        # Cap at 2% max to prevent unrealistic entries
        distance_pct = abs(entry - current_price) / current_price * 100
        if distance_pct > 2.0:
            if entry > current_price:
                entry = current_price * 1.02
            else:
                entry = current_price * 0.98
            entry_reason = f"Capped at 2.0% (was {distance_pct:.2f}%)"
            distance_pct = 2.0
        else:
            entry_reason = f"{direction}{pullback_pct:.2f}% ({vol_note}, {pullback_multiplier:.2f}×ATR)"
        
        return entry, entry_reason, distance_pct
    
    def get_structure_score_tier(self, structure_score: float) -> Tuple[float, str]:
        """Convert structure score to contribution points"""
        
        if structure_score >= 85:
            return 2.5, "Exceptional structure (+2.5)"
        elif structure_score >= 70:
            return 2.0, "Strong structure (+2)"
        elif structure_score >= 60:
            return 1.5, "Good structure (+1.5)"
        elif structure_score >= 50:
            return 1.0, "Moderate structure (+1)"
        elif structure_score >= 40:
            return 0.5, "Basic structure (+0.5)"
        else:
            return 0.0, "Weak structure (+0)"
    
    def calculate_natural_target(self, candles: List, entry: float, stop_loss: float, regime: MarketRegime, is_short: bool) -> Tuple[float, float]:
        """Calculate NATURAL target from market structure"""
        
        if entry <= 0 or stop_loss <= 0:
            print(f"⚠️ Invalid inputs: entry={entry}, stop_loss={stop_loss}")
            atr = self._calculate_atr(candles[-14:]) if len(candles) >= 14 else entry * 0.01
            if is_short:
                target = entry - (atr * 1.5)
            else:
                target = entry + (atr * 1.5)
            return target, 2.0
        
        highs = [float(c["high"]) for c in candles[-20:]]
        lows = [float(c["low"]) for c in candles[-20:]]
        
        # Calculate risk first
        if is_short:
            risk = stop_loss - entry
        else:
            risk = entry - stop_loss
        
        # Minimum reward = 2x risk
        min_reward = risk * 2.0
        
        if is_short:
            # Find support below entry
            supports_below = [l for l in lows if l < entry and l > entry * 0.95]
            if supports_below:
                target = max(supports_below)
            else:
                # Use ATR-based target
                atr = self._calculate_atr(candles[-14:])
                if atr <= 0:
                    atr = entry * 0.01
                target = entry - (atr * 1.5)
            
            # Ensure minimum reward
            reward = entry - target
            if reward < min_reward:
                target = entry - min_reward
                reward = min_reward
            
            # Cap max reward at 10%
            max_reward = entry * 0.10
            if reward > max_reward:
                target = entry - max_reward
                reward = max_reward
        
        else:  # LONG
            # Find resistance above entry
            resistances_above = [h for h in highs if h > entry and h < entry * 1.05]
            if resistances_above:
                target = min(resistances_above)
            else:
                atr = self._calculate_atr(candles[-14:])
                if atr <= 0:
                    atr = entry * 0.01
                target = entry + (atr * 1.5)
            
            # Ensure minimum reward
            reward = target - entry
            if reward < min_reward:
                target = entry + min_reward
                reward = min_reward
            
            max_reward = entry * 0.10
            if reward > max_reward:
                target = entry + max_reward
                reward = max_reward
        
        # Calculate natural RR
        if is_short:
            risk = stop_loss - entry
            reward = entry - target
        else:
            risk = entry - stop_loss
            reward = target - entry
        
        # Ensure positive values
        risk = abs(risk)
        reward = abs(reward)
        
        natural_rr = reward / risk if risk > 0 else 0
        
        print(f"🎯 Target calc: entry={entry:.2f}, sl={stop_loss:.2f}, tp={target:.2f}, risk={risk:.4f}, reward={reward:.4f}, RR={natural_rr:.2f}")
        
        return target, natural_rr

    def _calculate_atr(self, candles: List, period: int = 14, debug_symbol: str = "") -> float:
        """Calculate ATR with sanity checks"""
        
        if len(candles) < period:
            return 0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = float(candles[i]["high"])
            low = float(candles[i]["low"])
            prev_close = float(candles[i-1]["close"])
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        
        if not true_ranges:
            return 0
        
        atr = sum(true_ranges[-period:]) / period
        
        # Sanity check - ATR should be reasonable relative to price
        current_price = float(candles[-1]["close"])
        if current_price > 0:
            atr_pct = (atr / current_price) * 100
            if atr_pct < 0.1 and debug_symbol:
                print(f"⚠️ WARNING: {debug_symbol} ATR {atr:.4f} ({atr_pct:.2f}% of price) - unusually low")
            if atr_pct > 10 and debug_symbol:
                print(f"⚠️ WARNING: {debug_symbol} ATR {atr:.4f} ({atr_pct:.2f}% of price) - unusually high")
        
        return atr
    
    def analyze_entry_quality(self, candles: List, current_price: float, entry_price: float, regime: MarketRegime) -> Tuple[float, Dict]:
        """Score entry quality based on distance from ideal zone"""
        
        if entry_price <= 0:
            return 50, {"error": "no entry price"}
        
        # Calculate distance from current price
        distance_pct = abs(entry_price - current_price) / current_price * 100
        
        # Score based on proximity to ideal entry
        if distance_pct < 0.3:
            entry_score = 100  # Perfect - already at entry
        elif distance_pct < 0.5:
            entry_score = 90   # Very close
        elif distance_pct < 1.0:
            entry_score = 70   # Acceptable
        elif distance_pct < 2.0:
            entry_score = 50   # Extended
        else:
            entry_score = 30   # Chasing
        
        details = {
            "distance_pct": round(distance_pct, 2),
            "score": entry_score,
            "entry_vs_price": "above" if entry_price > current_price else "below"
        }
        
        return entry_score, details
    
    def analyze_rsi_context(self, candles: List, regime: MarketRegime) -> Tuple[float, Dict]:
        """STEP 7: RSI as context only - minimal importance"""
        
        closes = [float(c["close"]) for c in candles]
        series = pd.Series(closes)
        
        # Calculate RSI
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Context-based scoring (max ±5 impact)
        if regime == MarketRegime.TRENDING_BULLISH:
            if 40 <= current_rsi <= 65:
                rsi_score = 5  # Bonus for healthy trend
            elif current_rsi < 30:
                rsi_score = 3  # Slight bonus for oversold in uptrend
            elif current_rsi > 75:
                rsi_score = -3  # Slight penalty for overbought
            else:
                rsi_score = 0
        elif regime == MarketRegime.TRENDING_BEARISH:
            if 35 <= current_rsi <= 60:
                rsi_score = 5
            elif current_rsi > 70:
                rsi_score = 3
            elif current_rsi < 25:
                rsi_score = -3
            else:
                rsi_score = 0
        else:  # Range or reversal
            if current_rsi < 30 or current_rsi > 70:
                rsi_score = 3
            else:
                rsi_score = 0
        
        details = {
            "rsi": round(current_rsi, 1),
            "regime": regime.value,
            "score": rsi_score
        }
        
        self._log_debug("RSI_CONTEXT", details)
        return rsi_score, details
    
    def calculate_confidence(self, scores: Dict, details: Dict = None) -> Tuple[float, Dict]:
        """Transparent confidence calculation - no missing attributes"""
        
        # Normalize all scores to 0-100
        normalized = {
            "structure": max(0, min(100, scores.get("structure", 0))),
            "mtf_alignment": max(0, min(100, scores.get("mtf_alignment", 0))),
            "momentum": max(0, min(100, scores.get("momentum", 0))),
            "liquidity": max(0, min(100, scores.get("liquidity", 0))),
            "risk_reward": max(0, min(100, scores.get("risk_reward", 0))),
            "entry": max(0, min(100, scores.get("entry", 0))),
            "rsi": max(0, min(100, scores.get("rsi", 0)))
        }
        
        # Calculate weighted contributions
        structure_contrib = normalized["structure"] * 0.25
        mtf_contrib = normalized["mtf_alignment"] * 0.20
        momentum_contrib = normalized["momentum"] * 0.20
        liquidity_contrib = normalized["liquidity"] * 0.10
        rr_contrib = normalized["risk_reward"] * 0.10
        entry_contrib = normalized["entry"] * 0.10
        rsi_contrib = normalized["rsi"] * 0.05
        
        confidence = structure_contrib + mtf_contrib + momentum_contrib + \
                    liquidity_contrib + rr_contrib + entry_contrib + rsi_contrib
                    
        breakdown = {
            "structure": round(structure_contrib, 1),
            "mtf": round(mtf_contrib, 1),
            "momentum": round(momentum_contrib, 1),
            "liquidity": round(liquidity_contrib, 1),
            "rr": round(rr_contrib, 1),
            "entry": round(entry_contrib, 1),
            "rsi": round(rsi_contrib, 1),
            "base_total": round(confidence, 1)
        }
        
        print(f"\n🔍 CONFIDENCE TRACE for {details.get('symbol', 'unknown')}:")
        print(f"   STEP 1 - Base Score: {confidence:.4f}")
        
        # Get confluence multiplier
        confluence_pct = details.get("confluence_pct", 50) if details else 50
        
        # Determine multiplier
        if confluence_pct >= 80:
            multiplier = 1.00
        elif confluence_pct >= 70:
            multiplier = 0.98
        elif confluence_pct >= 60:
            multiplier = 0.95
        elif confluence_pct >= 50:
            multiplier = 0.90
        else:
            multiplier = 0.80
        
        print(f"   STEP 2 - Confluence: {confluence_pct:.1f}% → Multiplier: {multiplier}")
        
        confidence_before_mult = confidence
        confidence *= multiplier
        print(f"   STEP 3 - After Multiplier: {confidence:.4f} ({confidence_before_mult:.4f} × {multiplier})")
        
        # Apply penalties
        penalties = 0
        penalty_details = []
        
        if details:
            if details.get("overextended", False):
                penalties += 5
                penalty_details.append("overextended -5")
            if details.get("low_volatility", False):
                penalties += 3
                penalty_details.append("low volatility -3")
            if not details.get("volume_confirmed", True):
                penalties += 3
                penalty_details.append("weak volume -3")
            if details.get("recent_exhaustion", False):
                penalties += 2
                penalty_details.append("exhaustion -2")
        
        if penalties > 0:
            confidence_before_penalties = confidence
            confidence -= penalties
            print(f"   STEP 4 - Penalties: {penalties} ({penalty_details})")
            print(f"   STEP 5 - After Penalties: {confidence:.4f} (was {confidence_before_penalties:.4f})")
        else:
            print(f"   STEP 4 - Penalties: 0")
            print(f"   STEP 5 - After Penalties: {confidence:.4f} (unchanged)")
        
        # Apply floor for trending markets
        regime = details.get("regime", "") if details else ""
        floor_applied = None
        
        if regime in ["TRENDING_BULLISH", "TRENDING_BEARISH"]:
            old_conf = confidence
            if normalized["structure"] >= 70:
                confidence = max(55, confidence)
                if old_conf < 55:
                    floor_applied = f"55 (structure>=70, was {old_conf:.1f})"
            elif normalized["structure"] >= 60:
                confidence = max(50, confidence)
                if old_conf < 50:
                    floor_applied = f"50 (structure>=60, was {old_conf:.1f})"
            
            if floor_applied:
                print(f"   STEP 6 - Floor Applied: {floor_applied}")
                print(f"   STEP 7 - After Floor: {confidence:.4f}")
            else:
                print(f"   STEP 6 - No floor needed (confidence {old_conf:.1f} >= floor threshold)")
        
        # Final clamp
        confidence = max(0, min(100, confidence))
        print(f"   STEP 8 - Final Score: {confidence:.4f}\n")
        
        return max(0, min(100, confidence)), breakdown
    
    def determine_signal(self, confidence: float, regime: MarketRegime, 
                structure_score: float, mtf_score: float, rr_score: float, 
                confluence_pct: float, details: Dict, volume_24h: float = 0,
                reversal_confirmed: bool = True) -> Signal:
        """ Fixed: GOOD requires confidence >= 65 """
        
        confidence = round(confidence, 1)

        # HARD FILTERS (absolute rejects)
        if rr_score <= 0:
            return Signal.IGNORE
        
        if regime == MarketRegime.RANGE:
            return Signal.IGNORE
        
        if volume_24h < 15_000_000:
            return Signal.IGNORE
        
        if structure_score < 45:
            return Signal.IGNORE
        
        # HARD CONFLUENCE RULE - Force IGNORE below 45%
        if confluence_pct < 45:
            return Signal.IGNORE
        
        # Extract symbol for state tracking
        symbol = details.get('symbol', 'unknown')
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            print(f"\n📊 SIGNAL PERSISTENCE for {symbol}:")
        
        # Extract major conditions
        has_htf_aligned = details.get("mtf_aligned", False)
        has_momentum_confirmed = details.get("momentum_confirmed", False)
        has_volume_confirmed = details.get("volume_confirmed", False)

        if symbol == 'ETHUSDT':
            print(f"\n🔍 ENTER_GOOD DEBUG for {symbol}:")
            print(f"   confidence raw = {confidence!r} (type: {type(confidence).__name__})")
            print(f"   confidence >= 65 = {confidence >= 65}")
            print(f"   confluence_pct raw = {confluence_pct!r}")
            print(f"   confluence_pct >= 60 = {confluence_pct >= 60}")
            print(f"   structure_score raw = {structure_score!r}")
            print(f"   structure_score >= 55 = {structure_score >= 55}")
            print(f"   has_htf_aligned = {has_htf_aligned}")
            print(f"   has_momentum_confirmed = {has_momentum_confirmed}")
            print(f"   has_volume_confirmed = {has_volume_confirmed}")
            print(f"\n   ENTER_GOOD expression:")
            print(f"   = ({confidence >= 65}) and ({confluence_pct >= 60}) and ({structure_score >= 55}) and ({has_htf_aligned}) and ({has_momentum_confirmed}) and ({has_volume_confirmed})")
            
            # Calculate each term individually
            term1 = confidence >= 65
            term2 = confluence_pct >= 60
            term3 = structure_score >= 55
            term4 = has_htf_aligned
            term5 = has_momentum_confirmed
            term6 = has_volume_confirmed
            
            print(f"\n   Individual terms:")
            print(f"   term1 (conf>=65): {term1}")
            print(f"   term2 (confl>=60): {term2}")
            print(f"   term3 (struct>=55): {term3}")
            print(f"   term4 (htf_aligned): {term4}")
            print(f"   term5 (momentum_confirmed): {term5}")
            print(f"   term6 (volume_confirmed): {term6}")
            print(f"   \n   FINAL: {term1 and term2 and term3 and term4 and term5 and term6}")
        
        # ========== END DEBUG BLOCK ==========

        # REVERSAL special rule
        if regime == MarketRegime.REVERSAL_POTENTIAL:
            if not reversal_confirmed:
                if confluence_pct >= 50 and confidence >= 48:
                    return Signal.WATCHLIST
                return Signal.IGNORE
        
        # Get previous signal for hysteresis
        prev_signal, prev_confidence, prev_regime, signal_age = self._get_prev_signal(symbol)
        prev_data = self._signal_history.get(symbol)
        if prev_data and details.get('symbol') in ['BTCUSDT', 'ETHUSDT']:
            print(f"\n📊 SIGNAL PERSISTENCE for {symbol}:")
            print(f"   Previous Signal: {prev_signal.value if prev_signal else 'NONE'}")
            print(f"   Previous Confidence: {prev_confidence if prev_confidence else 'N/A'}")
            print(f"   Previous Regime: {prev_regime if prev_regime else 'N/A'}")
            print(f"   Signal Age: {signal_age:.1f}s" if signal_age else "   Signal Age: N/A")
            print(f"   Current Regime: {regime.value}")
            print(f"   Current Confidence: {confidence:.1f}")
            print(f"   Current Confluence: {confluence_pct:.1f}%")
            print(f"\n🔍 SIGNAL DEBUG for {details.get('symbol')}:")
            print(f"   confidence={confidence:.1f}, confluence={confluence_pct:.1f}%, structure={structure_score}")
            print(f"   prev_signal={self._get_prev_signal(details.get('symbol', ''))}")
            
        regime_matches = (prev_regime == regime.value) if prev_regime else False
        
        # TRENDING MARKETS
        if regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH]:
            
            # === GOOD SIGNAL CONDITIONS ===
            enter_good = (
                confidence >= 65.0 and
                confluence_pct >= 60.0 and
                structure_score >= 55 and
                has_htf_aligned and
                has_momentum_confirmed and
                has_volume_confirmed
            )
            if symbol == 'ETHUSDT':
                print(f"\n   enter_good evaluated to: {enter_good}")
            
            # === MAINTAIN GOOD with REGIME CHECK and TIGHTER HYSTERESIS ===
            if prev_signal == Signal.GOOD and regime_matches and signal_age is not None:
                # Tighter hysteresis: 3-point buffer instead of 7
                maintain_good = (
                    confidence >= 62.0 and  # Was 58 - tighter!
                    confluence_pct >= 55.0 and
                    structure_score >= 50
                )
                
                if maintain_good:
                    if symbol in ['BTCUSDT', 'ETHUSDT']:
                        print(f"   → MAINTAIN_GOOD (prev GOOD, regime matches, conf>=62, age={signal_age:.0f}s)")
                    self._store_signal_state(symbol, Signal.GOOD, confidence, regime)
                    self._save_history()
                    return Signal.GOOD
                else:
                    if symbol in ['BTCUSDT', 'ETHUSDT']:
                        print(f"   → DROPPING GOOD (conf={confidence:.1f} < 62)")
            
            if enter_good:
                if symbol in ['BTCUSDT', 'ETHUSDT']:
                    print(f"   → ENTER_GOOD (conf>=65, confluence>=60, structure>=55)")
                self._store_signal_state(symbol, Signal.GOOD, confidence, regime)
                self._save_history()
                return Signal.GOOD
            
            # === WATCHLIST with tighter hysteresis ===
            enter_watchlist = (
                confidence >= 50.0 and
                confluence_pct >= 50.0 and
                structure_score >= 50
            )
            
            if prev_signal == Signal.WATCHLIST and regime_matches:
                maintain_watchlist = (
                    confidence >= 47.0 and  # 3-point buffer (was 46, now tighter)
                    confluence_pct >= 45.0
                )
                if maintain_watchlist:
                    if symbol in ['BTCUSDT', 'ETHUSDT']:
                        print(f"   → MAINTAIN_WATCHLIST (prev WATCHLIST, conf>=47)")
                    self._store_signal_state(symbol, Signal.WATCHLIST, confidence, regime)
                    self._save_history()
                    return Signal.WATCHLIST
            
            if enter_watchlist:
                if symbol in ['BTCUSDT', 'ETHUSDT']:
                    print(f"   → ENTER_WATCHLIST (conf>=50, confluence>=50)")
                self._store_signal_state(symbol, Signal.WATCHLIST, confidence, regime)
                self._save_history()
                return Signal.WATCHLIST
        
        # Fallback
        self._store_signal_state(symbol, Signal.IGNORE, confidence, regime)
        self._save_history()
        return Signal.IGNORE

    def _store_signal_state(self, symbol: str, signal: Signal, confidence: float, regime: MarketRegime):
        """Store previous signal with metadata"""
        self._signal_history[symbol] = {
            "signal": signal.value,
            "confidence": confidence,
            "regime": regime.value if regime else "UNKNOWN",
            "timestamp": time.time()
        }
        
        # Update streak
        prev = self._get_prev_signal(symbol)
        if prev == signal:
            self._signal_streak[symbol] = self._signal_streak.get(symbol, 0) + 1
        else:
            self._signal_streak[symbol] = 1

    def _get_prev_signal(self, symbol: str) -> tuple:
        """Get previous signal with metadata"""
        data = self._signal_history.get(symbol)
        if not data:
            return Signal.IGNORE, None, None, None
        
        # Check expiry
        age = time.time() - data.get("timestamp", 0)
        if age > self._history_expiry_seconds:
            # Expired - treat as IGNORE
            return Signal.IGNORE, None, None, age
        
        # Convert string back to Signal enum
        signal_map = {
            "GOOD": Signal.GOOD,
            "WATCHLIST": Signal.WATCHLIST,
            "IGNORE": Signal.IGNORE
        }
        signal = signal_map.get(data.get("signal", "IGNORE"), Signal.IGNORE)
        
        return signal, data.get("confidence"), data.get("regime"), age

    def _get_trade_confirmations(self, candles: List) -> Dict:
        """Check for trade confirmation signals"""
        if len(candles) < 20:
            return {}
        
        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        
        confirmations = {}
        
        # Liquidity sweep above recent highs
        recent_high = max(highs[-10:-1])
        if highs[-1] > recent_high and closes[-1] < recent_high:
            confirmations["liquidity_sweep_above"] = True
        
        # Bearish CHOCH (Change of Character)
        swing_highs = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
            highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
        
        if len(swing_highs) >= 3:
            if swing_highs[-1] < swing_highs[-2] and closes[-1] < lows[-2]:
                confirmations["bearish_choch"] = True
        
        # Rejection candle (long wick)
        candle_range = highs[-1] - lows[-1]
        if candle_range > 0:
            upper_wick = highs[-1] - max(closes[-1], candles[-1]["open"])
            if upper_wick / candle_range > 0.6:  # 60% wick
                confirmations["rejection_candle"] = True
        
        # Displacement down
        avg_body = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes))) / (len(closes)-1)
        current_body = abs(closes[-1] - closes[-2])
        if current_body > avg_body * 1.5 and closes[-1] < closes[-2]:
            confirmations["displacement_down"] = True
        
        # Failed breakout above resistance
        resistance = max(highs[-15:-5])
        if highs[-2] > resistance and closes[-1] < resistance:
            confirmations["failed_breakout"] = True
        
        # Bearish retest of broken support
        support = min(lows[-15:-5])
        if closes[-2] < support and closes[-1] > support and closes[-1] < support * 1.01:
            confirmations["bearish_retest"] = True
        
        return confirmations

    def _get_reversal_confirmations(self, candles: List) -> Dict:
        """Check for reversal confirmations"""
        if len(candles) < 20:
            return {"reversal_confirmed": False}
        
        closes = [float(c["close"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        
        # Bullish reversal confirmation
        recent_low = min(lows[-10:-1])
        reversal_confirmed = False
        
        # Check for bullish engulfing or hammer
        if closes[-1] > candles[-1]["open"] and closes[-2] < candles[-2]["open"]:
            if closes[-1] > candles[-2]["open"]:  # Engulfing
                reversal_confirmed = True
        
        # Check for higher low after sweep
        if lows[-1] > recent_low and closes[-1] > closes[-2]:
            reversal_confirmed = True
        
        return {"reversal_confirmed": reversal_confirmed}
    
    def process_market(self, ticker: Dict) -> Optional[Dict]:
        """ Main processing pipeline - institutional order flow """
        
        try:
            symbol = ticker["market"]
            current_price = float(ticker["last_price"])
            
            if symbol in config.DEADCANDLEMARKETS:
                return None
            
            # Stablecoin check
            if any(stable in symbol for stable in ["USDCUSDT", "USDTUSDT", "BUSDUSDT", "DAIUSDT"]):
                return None
            
            # === STAGE 2: CANDLE FETCH ===
            candles_5m = self.market.get_candles(symbol, interval="5m", limit=100)
            candles_15m = self.market.get_candles(symbol, interval="15m", limit=100)
            candles_1h = self.market.get_candles(symbol, interval="1h", limit=100)
            candles_4h = self.market.get_candles(symbol, interval="4h", limit=100)
            candles_5m_filter = self.market.get_candles(symbol, interval="5m", limit=100)
            
            if len(candles_5m) < 50:
                return None
            if not all([candles_15m, candles_1h, candles_4h]):
                return None
            
            # === STAGE 3: VOLATILITY CHECK (Early exit) ===
            atr = self._calculate_atr(candles_5m[-20:])
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
            
            if atr_pct < 0.25:  # LOW VOLATILITY - REJECT IMMEDIATELY
                self.market.safe_print(f"❌ {symbol}: Low volatility rejection (ATR {atr_pct:.2f}%)")
                return None
            
            if len(candles_15m) < 30:
                self.market.safe_print(f"❌ Skipping {symbol} - insufficient candles")
                return None
            
            # Quick check for price action
            highs_15m = [float(c["high"]) for c in candles_15m[-20:]]
            lows_15m = [float(c["low"]) for c in candles_15m[-20:]]
            if max(highs_15m) == min(lows_15m):
                self.market.safe_print(f"❌ Skipping {symbol} - no price movement")
                return None
            
            filter_result = self.filter_market(symbol, ticker, candles_5m_filter)
            if not filter_result["valid"]:
                self.market.safe_print(f"❌ Skipping {symbol} - {filter_result['reason']}")
                return None
            
            change_24h = float(ticker.get("change_24_hour", 0))
            if abs(change_24h) > 10:
                # Check if exceptional case (pullback after breakout)
                candles_1h_check = self.market.get_candles(symbol, interval="1h", limit=24)
                if candles_1h_check:
                    closes_1h = [float(c["close"]) for c in candles_1h_check]
                    # Check if consolidating after move
                    recent_range = (max(closes_1h[-6:]) - min(closes_1h[-6:])) / min(closes_1h[-6:]) * 100
                    if recent_range > 5:  # Still volatile
                        self.market.safe_print(f"❌ Skipping {symbol} - {abs(change_24h):.1f}% move, no consolidation")
                        return None
                else:
                    self.market.safe_print(f"❌ Skipping {symbol} - {abs(change_24h):.1f}% extreme move")
                    return None
            if change_24h > 12:  # Avoid extreme moves
                self.market.safe_print(f"❌ Skipping {symbol} - {change_24h:.1f}% move too extreme")
                return None
            
            if not all([candles_5m, candles_15m, candles_1h, candles_4h]):
                return None
            
            regime, regime_details = self.analyze_market_regime(candles_4h, candles_1h)
        
            if regime == MarketRegime.RANGE:
                return None
            
            print(f"DEBUG {symbol}: Regime = {regime.value}, Current Price = {current_price}")
            
            is_long = regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.REVERSAL_POTENTIAL]
            is_short = regime == MarketRegime.TRENDING_BEARISH
            
            is_exhausting, last_3_move, avg_candle = self.check_recent_exhaustion(candles_15m, is_short)
            
            # STEP 2: Multi-timeframe alignment
            tf_analysis = {}
            for tf_name, tf_candles in [("5M", candles_5m), ("15M", candles_15m), 
                                        ("1H", candles_1h), ("4H", candles_4h)]:
                closes = [float(c["close"]) for c in tf_candles]
                if len(closes) >= 20:
                    ema20 = pd.Series(closes).ewm(span=20).mean().iloc[-1]
                    tf_analysis[tf_name] = {
                        "bullish": closes[-1] > ema20,
                        "bearish": closes[-1] < ema20
                    }
            
            mtf_score, mtf_details = self.analyze_mtf_alignment(
                tf_analysis.get("5M", {}), tf_analysis.get("15M", {}),
                tf_analysis.get("1H", {}), tf_analysis.get("4H", {})
            )
            
            # STEP 3-7: Component analysis
            structure_score, structure_details = self.analyze_structure_quality(candles_15m, current_price, regime)

            liquidity_score, liquidity_details = self.analyze_liquidity(candles_5m, ticker)
            momentum_score, momentum_details = self.analyze_momentum_displacement(candles_5m, liquidity_score)

            # Calculate ATR for entry calculation
            atr = self._calculate_atr(candles_15m, period=14)

            # Calculate entry price based on institutional logic (NOT current price)
            entry_price, entry_reason, entry_distance = self.calculate_entry_price(candles_15m, current_price, regime, atr)
            
            # Entry quality filter - reject if price moved too far
            distance_pct = abs(entry_price - current_price) / current_price * 100

            if distance_pct > 1.5:
                self.market.safe_print(f"❌ Skipping {symbol} - entry {entry_price} too far from current {current_price} ({distance_pct:.2f}%)")
                return None
            elif distance_pct > 0.5:
                self.market.safe_print(f"⚠️ {symbol} - Entry distance: +{distance_pct:.2f}% (acceptable but not ideal)")

            # Store for display
            entry_distance_pct = distance_pct
            rr_score, rr_details = self.analyze_risk_reward(candles_15m, entry_price, current_price, regime, symbol, timeframe="15m")
            
            if rr_score <= 0:
                self.market.safe_print(f"❌ Skipping {symbol} - invalid RR")
                return None
            
            entry_score, entry_details = self.analyze_entry_quality(candles_5m, current_price, entry_price, regime)
            rsi_score, rsi_details = self.analyze_rsi_context(candles_5m, regime)
            
            momentum_score = min(100, max(0, momentum_score))

            if entry_price > 0 and current_price > 0:
                entry_distance_pct = abs(entry_price - current_price) / current_price * 100
                if entry_distance_pct > 3.0:
                    self.market.safe_print(f"❌ Skipping {symbol} - entry {entry_price} too far from market {current_price}")
                    return None
            
            # STEP 8: Component scores
            component_scores = {
                "structure": structure_score,
                "mtf_alignment": mtf_score,
                "liquidity": liquidity_score,
                "momentum": momentum_score,
                "risk_reward": rr_score,
                "entry": entry_score,
                "rsi": rsi_score
            }

            # ========== FIXED: Build result WITHOUT confidence_details placeholder ==========
            result = {
                "symbol": symbol,
                "regime": regime.value,
                "price": float(ticker["last_price"]),
                "change_24h": float(ticker["change_24_hour"]),
                "volume_24h": float(ticker["volume"]),
                "details": {
                    "regime": regime_details,
                    "mtf": mtf_details,
                    "structure": structure_details,
                    "liquidity": liquidity_details,
                    "momentum": momentum_details,
                    "rr": rr_details,
                    "entry_quality": entry_details,
                    "rsi": rsi_details,
                    # No confidence_details here yet - will add after confluence
                }
            }
            
            # Calculate confluence FIRST (before confidence)
            confluence_pct, confluence_score, confluence_max, confluence_reasons = self.calculate_confluence(result)
            
            # NOW build confidence_details with correct confluence
            confidence_details = {
                "symbol": symbol,
                "regime": regime.value,
                "confluence_pct": confluence_pct,
                "volume_confirmed": momentum_details.get("volume_confirmed", False),
                "momentum_score": momentum_score,
                "structure_score": structure_score,
                "mtf_score": mtf_score,
                "rr_score": rr_score,
                "atr_pct": (atr / current_price) * 100 if current_price > 0 else 100,
                "entry_distance_pct": entry_distance_pct,
                "counter_trend": False,
                "overextended": abs(change_24h) > 8,
                "recent_exhaustion": is_exhausting,
                "last_3_move_pct": last_3_move
            }
            
            # Calculate confidence with correct confluence
            confidence, confidence_breakdown = self.calculate_confidence(component_scores, confidence_details)
            
            # Add confidence_details to result AFTER it's defined
            result["details"]["confidence"] = confidence_details
            
            # Debug print for confluence
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                self.market.safe_print(f"🔧 {symbol}: Confluence {confluence_pct:.1f}% → Multiplier {confidence_breakdown.get('confluence_multiplier', '?')}")
                self.market.safe_print(f"🔍 Confidence breakdown for {symbol}: {confidence_breakdown}")
            
            if is_exhausting:
                self.market.safe_print(f"⚠️ {symbol} - Recent exhaustion detected: {last_3_move:.2f}% move in 3 candles")

            # STEP 9: Determine signal
            signal_details = {
                "symbol": symbol,  # ← MUST include this!
                "mtf_aligned": mtf_score >= 65,
                "momentum_confirmed": momentum_score >= 70,
                "volume_confirmed": momentum_details.get("volume_confirmed", False)
            }

            # Initialize reversal_confirmed variable
            reversal_confirmed = True
            reversal_reasons = []
            
            if symbol == 'ETHUSDT':
                print(f"\n🔍 SIGNAL DETAILS for {symbol}:")
                print(f"   mtf_aligned = {mtf_score >= 65}")
                print(f"   momentum_confirmed = {momentum_score >= 70}")
                print(f"   volume_confirmed = {momentum_details.get('volume_confirmed', False)}")
                print(f"   momentum_score raw = {momentum_score}")
            
            # Determine signal
            signal = self.determine_signal(
                confidence, regime, structure_score, mtf_score, rr_score, 
                confluence_pct, signal_details,
                volume_24h=float(ticker.get("volume", 0)),
                reversal_confirmed=reversal_confirmed if regime == MarketRegime.REVERSAL_POTENTIAL else True
            )
            
            if regime == MarketRegime.REVERSAL_POTENTIAL:
                is_long = regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.REVERSAL_POTENTIAL]
                is_short = regime == MarketRegime.TRENDING_BEARISH
                
                # For reversal, we need to check direction
                reversal_long = is_long
                reversal_confirmed, reversal_reasons = self.check_reversal_confirmation(candles_15m, regime, reversal_long)
                
                if not reversal_confirmed:
                    # Cap confidence at 65
                    original_confidence = confidence
                    confidence = min(65, confidence)
                    self.market.safe_print(f"⚠️ Reversal waiting for confirmation - confidence capped: {original_confidence:.1f} → {confidence:.1f}")
                    
                    # Store reversal reasons for display
                    result["reversal_waiting"] = True
                    result["reversal_reasons"] = reversal_reasons
                    
                    # Never allow GOOD for unconfirmed reversals
                    if signal == Signal.GOOD:
                        signal = Signal.WATCHLIST
                        self.market.safe_print(f"⚠️ Reversal waiting for confirmation - downgraded from GOOD to WATCHLIST")
                else:
                    result["reversal_waiting"] = False
                    result["reversal_reasons"] = reversal_reasons
                    self.market.safe_print(f"✅ Reversal confirmed: {len(reversal_reasons)}/4 confirmations")

            # Complete result
            result.update({
                "signal": signal.value,
                "confidence": round(confidence, 1),
                "rr_ratio": rr_details.get("rr_ratio", 0),
                "entry": rr_details.get("entry", 0),
                "stop_loss": rr_details.get("stop_loss", 0),
                "target": rr_details.get("target", 0),
                "confluence": {
                    "percentage": confluence_pct,
                    "score": confluence_score,
                    "max_score": confluence_max,
                    "reasons": confluence_reasons
                },
                "entry_distance_pct": entry_distance,
                "entry_reason": entry_reason
            })
            
            self.check_atr_sanity(symbol, candles_15m, atr, current_price)
            
            # Print final analysis
            self._print_analysis(result)
            self.market.safe_print(f"🔍 {symbol}: {regime.value} | Structure: {structure_score} | MTF: {mtf_score} | RR: {rr_score}")
            
            return result
            
        except Exception as e:
            self.market.safe_print(f"Process Error ({ticker.get('market')}): {e}")
            traceback.print_exc()
            return None
        
    def check_reversal_confirmation(self, candles: List, regime: MarketRegime, is_long: bool) -> Tuple[bool, List[str]]:
        """Check if reversal setup has proper confirmation"""
        
        if len(candles) < 30:
            return False, ["Insufficient data"]
        
        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        volumes = [float(c["volume"]) for c in candles]
        
        confirmations = []
        required_count = 3  # Need at least 3 confirmations
        
        # Calculate RSI for divergence detection
        rsi = self._calculate_rsi(closes, period=14)
        prev_rsi = self._calculate_rsi(closes[:-5], period=14) if len(closes) > 20 else rsi
        
        if is_long:
            # LONG REVERSAL CONFIRMATIONS
            
            # 1. RSI bullish divergence OR oversold recovery
            price_made_lower_low = lows[-1] < lows[-5] if len(lows) >= 5 else False
            rsi_made_higher_low = rsi > prev_rsi
            if price_made_lower_low and rsi_made_higher_low:
                confirmations.append("✓ RSI bullish divergence")
            elif rsi < 35 and closes[-1] > closes[-2]:
                confirmations.append("✓ Oversold recovery")
            
            # 2. Break above previous swing high OR EMA reclaim
            recent_high = max(highs[-10:-1])
            if closes[-1] > recent_high:
                confirmations.append("✓ Broke above swing high")
            else:
                # Check EMA20 reclaim
                df = pd.DataFrame({"close": closes})
                df["ema20"] = df["close"].ewm(span=20).mean()
                if closes[-1] > df["ema20"].iloc[-1] and closes[-2] < df["ema20"].iloc[-2]:
                    confirmations.append("✓ EMA20 reclaim")
            
            # 3. Volume expansion
            avg_volume = sum(volumes[-20:-5]) / 15 if len(volumes) >= 20 else volumes[-1]
            if volumes[-1] > avg_volume * 1.2:
                confirmations.append("✓ Volume expansion")
            
            # 4. Confirmation candle close
            candle_range = highs[-1] - lows[-1]
            if candle_range > 0:
                close_strength = (closes[-1] - lows[-1]) / candle_range
                if close_strength > 0.6:  # Closed in upper 40% of candle
                    confirmations.append("✓ Strong confirmation candle")
        
        else:
            # SHORT REVERSAL CONFIRMATIONS
            
            # 1. RSI bearish divergence OR overbought rejection
            price_made_higher_high = highs[-1] > highs[-5] if len(highs) >= 5 else False
            rsi_made_lower_high = rsi < prev_rsi
            if price_made_higher_high and rsi_made_lower_high:
                confirmations.append("✓ RSI bearish divergence")
            elif rsi > 65 and closes[-1] < closes[-2]:
                confirmations.append("✓ Overbought rejection")
            
            # 2. Break below swing low OR EMA loss
            recent_low = min(lows[-10:-1])
            if closes[-1] < recent_low:
                confirmations.append("✓ Broke below swing low")
            else:
                # Check EMA20 loss
                df = pd.DataFrame({"close": closes})
                df["ema20"] = df["close"].ewm(span=20).mean()
                if closes[-1] < df["ema20"].iloc[-1] and closes[-2] > df["ema20"].iloc[-2]:
                    confirmations.append("✓ EMA20 loss")
            
            # 3. Volume expansion
            avg_volume = sum(volumes[-20:-5]) / 15 if len(volumes) >= 20 else volumes[-1]
            if volumes[-1] > avg_volume * 1.2:
                confirmations.append("✓ Volume expansion")
            
            # 4. Confirmation candle close
            candle_range = highs[-1] - lows[-1]
            if candle_range > 0:
                close_strength = (highs[-1] - closes[-1]) / candle_range
                if close_strength > 0.6:  # Closed in lower 40% of candle
                    confirmations.append("✓ Strong confirmation candle")
        
        is_confirmed = len(confirmations) >= required_count
        return is_confirmed, confirmations
        
    def check_atr_sanity(self, symbol: str, candles: List, atr: float, current_price: float):
        """Debug ATR calculation issues"""
        
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
        
        if atr_pct < 0.2:
            self.market.safe_print(f"⚠️ WARNING: {symbol} ATR unusually low: {atr:.4f} ({atr_pct:.2f}% of price)")
            
            # Debug info
            if len(candles) >= 14:
                highs = [float(c["high"]) for c in candles[-14:]]
                lows = [float(c["low"]) for c in candles[-14:]]
                avg_range = sum(highs[i] - lows[i] for i in range(len(highs))) / len(highs)
                avg_range_pct = (avg_range / current_price) * 100
                self.market.safe_print(f"   Avg candle range: {avg_range:.4f} ({avg_range_pct:.2f}%)")
        
        return atr_pct
    
    def _update_signal_streak(self, symbol: str, new_signal: Signal) -> int:
        """Update consecutive signal count"""
        prev_signal = self._get_prev_signal(symbol)
        
        if new_signal == prev_signal:
            self._signal_streak[symbol] = self._signal_streak.get(symbol, 0) + 1
        else:
            self._signal_streak[symbol] = 1
        
        return self._signal_streak[symbol]
    
    def _can_promote(self, symbol: str, target_signal: Signal) -> bool:
        """Check if signal can be promoted (needs streak)"""
        current_streak = self._signal_streak.get(symbol, 0)
        
        # Promotion requires minimum streak
        if target_signal == Signal.GOOD:
            return current_streak >= self._min_streak_required
        return True
        
    def calculate_confluence(self, analysis: Dict) -> Tuple[float, float, int, List[str]]:
        """
        Fixed confluence calculation - no double counting
        Returns: (percentage, score, max_score, reasons)
        """
        
        reasons = []
        score = 0.0
        max_score = 10.0
        
        # 1. MTF ALIGNMENT (2.5 max)
        mtf_score = analysis['details']['mtf'].get('final_score', 0)
        if mtf_score >= 85:
            score += 2.5
            reasons.append("✓ Perfect HTF alignment (+2.5)")
        elif mtf_score >= 70:
            score += 2.0
            reasons.append("✓ Strong HTF alignment (+2)")
        elif mtf_score >= 60:
            score += 1.5
            reasons.append("✓ Good HTF alignment (+1.5)")
        
        # 2. MOMENTUM (2.5 max)
        momentum = analysis['details']['momentum'].get('score', 0)
        if momentum >= 80:
            score += 2.5
            reasons.append("✓ Exceptional momentum (+2.5)")
        elif momentum >= 65:
            score += 1.5
            reasons.append("✓ Strong momentum (+1.5)")
        elif momentum >= 50:
            score += 1.0
            reasons.append("✓ Moderate momentum (+1)")
        elif momentum >= 35:
            score += 0.5
            reasons.append("✓ Weak momentum (+0.5)")
        
        # 3. RISK REWARD (2.0 max)
        rr = analysis.get('rr_ratio', 0)
        if rr >= 3.0:
            score += 2.0
            reasons.append(f"✓ Excellent RR {rr:.1f}:1 (+2)")
        elif rr >= 2.5:
            score += 1.5
            reasons.append(f"✓ Great RR {rr:.1f}:1 (+1.5)")
        elif rr >= 2.0:
            score += 1.0
            reasons.append(f"✓ Good RR {rr:.1f}:1 (+1)")
        
        # 4. VOLUME (1.0 max)
        if analysis['details']['momentum'].get('volume_confirmed'):
            score += 1.0
            reasons.append("✓ Volume confirmed (+1)")
        
        # 5. LIQUIDITY (1.0 max)
        volume_24h = analysis.get('volume_24h', 0)
        if volume_24h > 100_000_000:
            score += 1.0
            reasons.append("✓ Very high liquidity (+1)")
        elif volume_24h > 50_000_000:
            score += 0.75
            reasons.append("✓ High liquidity (+0.75)")
        
        # 6. ENTRY (1.0 max)
        entry_dist = analysis.get('entry_distance_pct', 100)
        if abs(entry_dist) < 0.3:
            score += 1.0
            reasons.append("✓ Perfect entry (+1)")
        elif abs(entry_dist) < 0.5:
            score += 0.75
            reasons.append("✓ Good entry (+0.75)")
        
        # Calculate percentage from ACTUAL score (not rounded)
        raw_percentage = (score / max_score) * 100
        
        # SMOOTHING to prevent flipping
        symbol = analysis.get('symbol', '')
        if not hasattr(self, '_confluence_history'):
            self._confluence_history = {}
        
        prev = self._confluence_history.get(symbol, raw_percentage)
        # EMA with alpha=0.4 (60% weight to previous, 40% to new)
        smoothed = (prev * 0.6) + (raw_percentage * 0.4)
        self._confluence_history[symbol] = smoothed
        
        return smoothed, score, max_score, reasons
    
    def get_market_bias(self) -> Tuple[str, float]:
        """Get BTC market regime as macro bias"""
        try:
            # Fetch BTC 4H candles
            btc_candles = self.market.get_candles("BTCUSDT", interval="4h", limit=50)
            if not btc_candles or len(btc_candles) < 30:
                return "NEUTRAL", 0
            
            closes = [float(c["close"]) for c in btc_candles]
            highs = [float(c["high"]) for c in btc_candles]
            lows = [float(c["low"]) for c in btc_candles]
            
            # Calculate EMAs
            df = pd.DataFrame({"close": closes})
            df["ema20"] = df["close"].ewm(span=20).mean()
            df["ema50"] = df["close"].ewm(span=50).mean()
            
            current_price = closes[-1]
            ema20 = df["ema20"].iloc[-1]
            ema50 = df["ema50"].iloc[-1]
            
            # Detect trend
            if current_price > ema20 > ema50:
                return "BULLISH", 0.8
            elif current_price < ema20 < ema50:
                return "BEARISH", 0.8
            else:
                return "NEUTRAL", 0.4
                
        except Exception as e:
            return "NEUTRAL", 0
    
    def _print_analysis(self, analysis: Dict):
        """Fixed display without double printing"""
        with self.print_lock:
            print("\n" + "="*80)
            print(f"📊 {analysis['symbol']} | {analysis['signal']} | Confidence: {analysis['confidence']:.1f}/100")
            print("="*80)
            print(f"🎯 Setup: {analysis['regime']} | RR: {analysis['rr_ratio']:.2f}")
            
            # Show entry distance
            entry_dist = analysis.get('entry_distance_pct', 0)
            dist_symbol = "+" if entry_dist > 0 else ""
            print(f"💰 Current: {analysis['price']:.4f} | Entry: {analysis['entry']:.4f} ({dist_symbol}{entry_dist:.2f}%)")
            print(f"🛑 Stop: {analysis['stop_loss']:.4f} | 🎯 Target: {analysis['target']:.4f}")
            
            # Show confluence with proper formatting
            if "confluence" in analysis:
                conf = analysis['confluence']
                # Handle different return formats
                if isinstance(conf, dict):
                    score = conf.get('score', 0)
                    max_score = conf.get('max_score', 12)
                    percentage = conf.get('percentage', 0)
                    reasons = conf.get('reasons', [])
                else:
                    # If it's a tuple from older code
                    score = conf[1] if len(conf) > 1 else 0
                    max_score = conf[2] if len(conf) > 2 else 12
                    percentage = conf[0] if len(conf) > 0 else 0
                    reasons = conf[3] if len(conf) > 3 else []
                
                print(f"\n📋 CONFLUENCE: {score:.1f}/{max_score:.0f} ({percentage:.0f}%)")
                # Show unique reasons only
                seen_reasons = set()
                for reason in reasons[:6]:
                    if reason not in seen_reasons:
                        seen_reasons.add(reason)
                        print(f"  {reason}")
                        
            if analysis.get('reversal_waiting', False):
                print(f"\n⚠️ REVERSAL AWAITING CONFIRMATION")
                for reason in analysis.get('reversal_reasons', [])[:3]:
                    print(f"  {reason}")
            
            print("="*80 + "\n")
            
    def check_recent_exhaustion(self, candles: List, is_short: bool) -> Tuple[bool, float, float]:
        """Check if recent move is exhausted (too aggressive)"""
        
        if len(candles) < 4:
            return False, 0, 0
        
        closes = [float(c["close"]) for c in candles[-4:]]
        
        # Calculate last 3 candles move percentage
        last_3_move_pct = abs((closes[-1] - closes[-4]) / closes[-4]) * 100
        
        # Calculate individual candle momentum
        candle_1 = abs((closes[-1] - closes[-2]) / closes[-2]) * 100
        candle_2 = abs((closes[-2] - closes[-3]) / closes[-3]) * 100
        candle_3 = abs((closes[-3] - closes[-4]) / closes[-4]) * 100
        avg_candle_move = (candle_1 + candle_2 + candle_3) / 3 if candle_1 + candle_2 + candle_3 > 0 else 0
        
        # Check if move is exhausting (last candle smaller than average)
        is_exhausting = candle_1 < avg_candle_move * 0.7 and last_3_move_pct > 3 if avg_candle_move > 0 else False
        
        return is_exhausting, last_3_move_pct, avg_candle_move
            
    def filter_market(self, symbol: str, ticker: Dict, candles_5m: List = None) -> Dict:
        """Robust market filtering before expensive analysis"""
        
        # 1. Dead candle filter
        if symbol in config.DEADCANDLEMARKETS:
            return {"valid": False, "reason": "dead candle market"}
        
        # 2. Stablecoin/pegged asset filter
        stablecoins = ["USDCUSDT", "USDTUSDT", "BUSDUSDT", "DAIUSDT", "XUSDUSDT", "FDUSDUSDT", 
                    "TUSDUSDT", "USDPUSDT", "PYUSDUSDT", "EURCUSDT"]
        if symbol in stablecoins:
            return {"valid": False, "reason": "permanent stablecoin blacklist"}
        for stable in stablecoins:
            if stable in symbol:
                self.market.safe_print(f"❌ Skipping stable/pegged asset: {symbol}")
                return {"valid": False, "reason": f"stablecoin: {stable}"}
        
        # 3. Minimum candle data
        if not candles_5m or len(candles_5m) < 100:
            return {"valid": False, "reason": f"insufficient candles: {len(candles_5m) if candles_5m else 0}"}
        
        # 4. Check ATR % for low volatility assets
        if candles_5m and len(candles_5m) >= 20:
            atr = self._calculate_atr(candles_5m[-20:])
            current_price = float(ticker.get("last_price", 0))
            if current_price > 0:
                atr_pct = (atr / current_price) * 100
                if atr_pct < 0.1:  # Less than 0.1% volatility
                    self.market.safe_print(f"❌ Skipping low volatility asset: {symbol} (ATR {atr_pct:.2f}%)")
                    return {"valid": False, "reason": f"low volatility: ATR {atr_pct:.2f}%"}
        
        # 5. Extreme move filter (simple version without regime)
        change_24h = abs(float(ticker.get("change_24_hour", 0)))
        volume_24h = float(ticker.get("volume", 0))
        
        if change_24h > 15:  # Extreme move threshold
            if candles_5m and len(candles_5m) >= 24:
                closes = [float(c["close"]) for c in candles_5m[-24:]]
                recent_range = (max(closes[-6:]) - min(closes[-6:])) / min(closes[-6:]) * 100
                if recent_range > 5 or volume_24h < 50_000_000:
                    return {"valid": False, "reason": f"{change_24h:.1f}% extreme move, no consolidation"}
            else:
                return {"valid": False, "reason": f"{change_24h:.1f}% extreme move"}
        
        # 6. Liquidity filter
        if volume_24h < 10_000_000:
            return {"valid": False, "reason": f"low liquidity: ${volume_24h:,.0f}"}
        
        # 7. Price sanity check
        price = float(ticker.get("last_price", 0))
        if price < 0.1:
            return {"valid": False, "reason": f"price too low: ${price}"}
        
        return {"valid": True, "reason": "passed all filters"}
    
    def _log_debug(self, component: str, details: Dict):
        """Internal debug logging"""
        # Uncomment for verbose debugging
        # with self.print_lock:
        #     print(f"🔍 {component}: {details}")
        pass
    
    def scan_markets(self, top_n: int = 10):
        """ Main scan function with duplicate prevention """
        
        try:
            self.market.safe_print("\n🔍 Institutional Market Scan Started...")
            
            # Get all markets
            tickers = self.market.get_all_tickers()
            if not tickers:
                self.market.safe_print("❌ Failed to fetch markets")
                return []
            
            # === STAGE 1: HARD FILTERS ===
            candidates = []
            seen_symbols = set()  # PREVENT DUPLICATES
            processed_symbols = set()  # ← Must be BEFORE the loop
            
            for ticker in tickers:
                symbol = ticker.get("market", "")
                
                # Skip duplicates
                if symbol in seen_symbols:
                    continue
                seen_symbols.add(symbol)
                
                # Skip stablecoins
                if any(pattern in symbol for pattern in config.STABLECOINS):
                    continue
                
                # Skip dead candle markets
                if symbol in config.DEADCANDLEMARKETS:
                    continue
                
                # Must be USDT pair
                if not symbol.endswith("USDT"):
                    continue
                
                # Volume filter
                volume_24h = float(ticker.get("volume", 0))
                if volume_24h < config.LIQUIDITY["min_24h_volume_usdt"]:
                    continue
                
                # Price sanity
                price = float(ticker.get("last_price", 0))
                if price < 0.5:
                    continue
                
                # Extreme move filter
                change = abs(float(ticker.get("change_24_hour", 0)))
                if change > 15:
                    continue
                
                candidates.append(ticker)
            
            self.market.safe_print(f"✅ Stage 1 passed: {len(candidates)} markets")
            
            # === STAGE 2: QUICK CANDLE CHECK ===
            viable = []
            for ticker in candidates[:50]:
                symbol = ticker.get("market", "")
                # Quick candle fetch to verify data exists
                candles = self.market.get_candles(symbol, interval="5m", limit=50)
                if candles and len(candles) >= 30:
                    viable.append(ticker)
            
            self.market.safe_print(f"✅ Stage 2 passed: {len(viable)} markets (has candle data)")
            
            # === STAGE 3: PARALLEL PROCESSING ===
            results = []
            processed_symbols = set()  # PREVENT DUPLICATES IN RESULTS
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.process_market, ticker): ticker for ticker in viable[:30]}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result and result['signal'] != 'IGNORE':
                        symbol = result['symbol']
                        if symbol not in processed_symbols:  # ← Check BEFORE adding
                            processed_symbols.add(symbol)
                            results.append(result)
            
            # === STAGE 4: SORT AND DISPLAY ===
            # Sort by confidence and confluence
            results.sort(key=lambda x: (x['confidence'], x.get('confluence', {}).get('percentage', 0)), reverse=True)
            
            # Clear screen and display
            self.market.safe_print("\n" + "🏆 TOP INSTITUTIONAL SETUPS".center(80))
            self.market.safe_print("="*80)
            
            if not results:
                self.market.safe_print("  No setups found meeting criteria")
            else:
                for i, result in enumerate(results[:top_n], 1):
                    signal_color = "🟢" if result['signal'] == "GOOD" else "🟡" if result['signal'] == "WATCHLIST" else "🔴"
                    self.market.safe_print(
                        f"{i:02}. {result['symbol']:12} | "
                        f"{signal_color} {result['signal']:12} | "
                        f"Conf: {result['confidence']:5.1f} | "
                        f"RR: {result['rr_ratio']:4.2f} | "
                        f"Entry: {result.get('entry', 0):8.2f} | "
                        f"24h: {result['change_24h']:5.2f}%"
                    )
            
            self.market.safe_print("="*80 + "\n")
            
            self._save_history()
            
            return results
            
        except KeyboardInterrupt:
            self.market.safe_print("\n👋 Scan stopped.")
            return []
        except Exception as e:
            self.market.safe_print(f"❌ Scan error: {e}")
            traceback.print_exc()
            return []