# src/backtest_framework/signal_generator.py
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGE = "RANGE"
    REVERSAL_POTENTIAL = "REVERSAL_POTENTIAL"

class Signal(Enum):
    GOOD = "GOOD"
    WATCHLIST = "WATCHLIST"
    IGNORE = "IGNORE"

@dataclass
class SignalResult:
    timestamp: datetime
    symbol: str
    signal: Signal
    confidence: float
    confluence_pct: float
    entry_price: float
    stop_loss: float
    target_price: float
    regime: str
    base_score: float
    multiplier_applied: float

class BacktestSignalGenerator:
    """Replicates your institutional scanner logic for backtesting"""
    
    def __init__(self, config: Dict, multiplier_version: str):
        self.config = config
        self.multiplier_version = multiplier_version
        self.MIN_MTF_SCORE = config.get('min_mtf_score', 75)  # gate for partial MTF alignment
        
    def calculate_confluence(self, analysis: Dict) -> Tuple[float, float]:
        """Calculate confluence score (replicates your calculate_confluence)"""
        
        score = 0.0
        max_score = 10.0
        
        # MTF Alignment (max 2.5)
        mtf_score = analysis.get('mtf_score', 0)
        if mtf_score >= 85:
            score += 2.5
        elif mtf_score >= 70:
            score += 2.0
        elif mtf_score >= 60:
            score += 1.5
        elif mtf_score >= 50:
            score += 1.0
        
        # Momentum (max 2.5)
        momentum = analysis.get('momentum_score', 0)
        if momentum >= 80:
            score += 2.5
        elif momentum >= 70:
            score += 2.0
        elif momentum >= 60:
            score += 1.5
        elif momentum >= 50:
            score += 1.0
        elif momentum >= 35:
            score += 0.5
        
        # Risk/Reward (max 2.0)
        rr = analysis.get('rr_ratio', 0)
        if rr >= 3.5:
            score += 2.0
        elif rr >= 3.0:
            score += 1.5
        elif rr >= 2.5:
            score += 1.0
        elif rr >= 2.0:
            score += 0.5
        
        # Volume confirmed (1.0)
        if analysis.get('volume_confirmed', False):
            score += 1.0
        
        # Liquidity (1.0)
        volume_24h = analysis.get('volume_24h', 0)
        if volume_24h > 100_000_000:
            score += 1.0
        elif volume_24h > 50_000_000:
            score += 0.75
        
        # Entry quality (1.0)
        entry_dist = analysis.get('entry_distance_pct', 100)
        if abs(entry_dist) < 0.3:
            score += 1.0
        elif abs(entry_dist) < 0.5:
            score += 0.75
        elif abs(entry_dist) < 0.8:
            score += 0.5
        
        percentage = (score / max_score) * 100
        return percentage, score
    
    def get_multiplier(self, confluence_pct: float) -> float:
        """Get multiplier based on selected version"""
        
        if self.multiplier_version == "extreme":
            # Extreme: Cut confidence in half
            return 0.50
        
        if self.multiplier_version == "current":
            if confluence_pct >= 80: return 1.00
            elif confluence_pct >= 70: return 0.98
            elif confluence_pct >= 60: return 0.95
            elif confluence_pct >= 50: return 0.90
            else: return 0.80
                
        elif self.multiplier_version == "soft":
            if confluence_pct >= 80: return 1.00
            elif confluence_pct >= 70: return 0.99
            elif confluence_pct >= 60: return 0.98
            elif confluence_pct >= 50: return 0.95
            else: return 0.90
                
        elif self.multiplier_version == "linear":
            return 0.8 + (confluence_pct / 100) * 0.2
                
        else:  # "none"
            return 1.00
    
    def calculate_base_confidence(self, scores: Dict) -> float:
        """Calculate base confidence without multiplier"""
        
        normalized = {
            "structure": min(100, max(0, scores.get("structure", 0))),
            "mtf": min(100, max(0, scores.get("mtf", 0))),
            "momentum": min(100, max(0, scores.get("momentum", 0))),
            "liquidity": min(100, max(0, scores.get("liquidity", 0))),
            "rr": min(100, max(0, scores.get("rr", 0))),
            "entry": min(100, max(0, scores.get("entry", 0))),
            "rsi": min(100, max(0, scores.get("rsi", 0)))
        }
        
        confidence = (
            normalized["structure"] * 0.25 +
            normalized["mtf"] * 0.20 +
            normalized["momentum"] * 0.20 +
            normalized["liquidity"] * 0.10 +
            normalized["rr"] * 0.10 +
            normalized["entry"] * 0.10 +
            normalized["rsi"] * 0.05
        )
        
        return confidence
    
    def generate_signal(self, timestamp: datetime, symbol: str, 
                    market_data: Dict) -> Optional[SignalResult]:
    
        df_1h = market_data['1h']
        if df_1h.empty or len(df_1h) < 50:
            return None
        
        current_price = df_1h['close'].iloc[-1]
        
        # Calculate REAL indicators
        ema20 = df_1h['close'].ewm(span=20).mean().iloc[-1]
        ema50 = df_1h['close'].ewm(span=50).mean().iloc[-1]
        ema20_slope = (ema20 - df_1h['close'].ewm(span=20).mean().iloc[-5]) / df_1h['close'].ewm(span=20).mean().iloc[-5] * 100
        
        # Calculate REAL structure score based on actual price action
        if current_price > ema20 > ema50 and ema20_slope > 0:
            structure_score = 75 + min(20, ema20_slope * 10)  # 75-95
            regime = MarketRegime.TRENDING_BULLISH
        elif current_price < ema20 < ema50 and ema20_slope < 0:
            structure_score = 75 + min(20, abs(ema20_slope) * 10)  # 75-95
            regime = MarketRegime.TRENDING_BEARISH
        else:
            return None  # Skip non-trending
        
        # Calculate REAL MTF alignment
        df_4h = market_data['4h']
        df_15m = market_data['15m']
        
        ema20_4h = df_4h['close'].ewm(span=20).mean().iloc[-1] if len(df_4h) >= 20 else ema20
        ema20_15m = df_15m['close'].ewm(span=20).mean().iloc[-1] if len(df_15m) >= 20 else ema20
        
        # Calculate alignment (0-100)
        mtf_score = 0
        price_above_ema_1h = current_price > ema20
        price_above_ema_4h = df_4h['close'].iloc[-1] > ema20_4h
        price_above_ema_15m = df_15m['close'].iloc[-1] > ema20_15m

        aligned_4h  = price_above_ema_1h == price_above_ema_4h
        aligned_15m = price_above_ema_1h == price_above_ema_15m

        if aligned_4h and aligned_15m:
            # Full alignment across all timeframes - strong signal
            mtf_score = 85 + min(15, abs(ema20_slope) * 5)  # 85-100
        elif aligned_4h and not aligned_15m:
            # 4h agrees but 15m disagrees - medium confidence
            # 4h is more important than 15m for trend direction
            mtf_score = 60
        elif not aligned_4h and aligned_15m:
            # 15m agrees but 4h disagrees - weak, higher timeframe conflict
            mtf_score = 45
        else:
            # Neither agrees - should not happen if regime filter passed,
            # but handle defensively
            return None  # Skip - no MTF confirmation at all
        
        if not aligned_4h:
            return None  # No 4h confirmation - skip regardless of other scores
        
        # Calculate REAL momentum
        returns = df_1h['close'].pct_change().iloc[-20:].dropna()
        momentum = (returns.mean() / returns.std()) * 10 if returns.std() > 0 else 0
        momentum_score = min(100, max(0, 50 + momentum * 5))  # 0-100
        
        # Calculate REAL ATR and RR
        atr = self._calculate_atr(df_1h)

        # Use recent swing high/low for stops instead of fixed ATR multiples
        lookback = 20
        recent_high = df_1h['high'].iloc[-lookback:].max()
        recent_low = df_1h['low'].iloc[-lookback:].min()

        if regime == MarketRegime.TRENDING_BULLISH:
            entry = current_price - (atr * 0.25)
            # Stop below recent swing low, with ATR buffer
            stop = recent_low - (atr * 0.5)
            # Ensure stop is not too far (cap at 3x ATR from entry)
            stop = max(stop, entry - (atr * 3.0))
            # Target: 2x the actual risk distance
            risk_distance = entry - stop
            target = entry + (risk_distance * 2.5)
        else:
            entry = current_price + (atr * 0.25)
            # Stop above recent swing high, with ATR buffer
            stop = recent_high + (atr * 0.5)
            # Ensure stop is not too far (cap at 3x ATR from entry)
            stop = min(stop, entry + (atr * 3.0))
            # Target: 2x the actual risk distance
            risk_distance = stop - entry
            target = entry - (risk_distance * 2.5)

        rr_ratio = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
        rr_score = min(100, 50 + (rr_ratio - 1.5) * 20)
        
        # Volume score
        avg_volume = df_1h['volume'].iloc[-20:-1].mean()
        current_volume = df_1h['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        liquidity_score = min(100, 50 + volume_ratio * 25)  # 50-100
        
        # Entry quality (how close to ideal)
        entry_distance_pct = abs(entry - current_price) / current_price * 100
        if entry_distance_pct < 0.3:
            entry_score = 100
        elif entry_distance_pct < 0.5:
            entry_score = 90
        elif entry_distance_pct < 1.0:
            entry_score = 70
        else:
            entry_score = 50
        
        # RSI score (minimal impact)
        rsi = self._calculate_rsi(df_1h['close'])
        if (regime == MarketRegime.TRENDING_BULLISH and 40 <= rsi <= 70) or \
        (regime == MarketRegime.TRENDING_BEARISH and 30 <= rsi <= 60):
            rsi_score = 60
        else:
            rsi_score = 50
            
        if mtf_score < self.MIN_MTF_SCORE:
            return None  # Partial alignment only - skip signal
        
        # Now proceed with confidence calculation...
        component_scores = {
            "structure": structure_score,
            "mtf": mtf_score,
            "momentum": momentum_score,
            "liquidity": liquidity_score,
            "rr": rr_score,
            "entry": entry_score,
            "rsi": rsi_score
        }
        
        # Calculate base confidence
        base_confidence = self.calculate_base_confidence(component_scores)
        
        # Calculate confluence
        analysis = {
            'mtf_score': mtf_score,
            'momentum_score': momentum_score,
            'rr_ratio': rr_ratio,
            'volume_confirmed': volume_ratio > 1.2,
            'volume_24h': avg_volume * current_price * 24,
            'entry_distance_pct': entry_distance_pct
        }
        
        confluence_pct, _ = self.calculate_confluence(analysis)
        
        # Apply multiplier
        multiplier = self.get_multiplier(confluence_pct)
        final_confidence = base_confidence * multiplier
        
        # Debug logging - disabled by default to keep logs clean
        # Set DEBUG_SIGNALS = True in config to re-enable
        if self.config.get('debug_signals', False) and self.multiplier_version != "none":
            confidence_without_multiplier = base_confidence
            if abs(confidence_without_multiplier - final_confidence) > 0.01:
                print(f"DEBUG: {self.multiplier_version} | {symbol} | "
                    f"Base: {confidence_without_multiplier:.1f} → "
                    f"Final: {final_confidence:.1f} | "
                    f"Multiplier: {multiplier} | "
                    f"Confluence: {confluence_pct:.1f}%")
        
        # Determine signal with multiplier
        good_threshold = self.config['signal_thresholds']['GOOD']
        watchlist_threshold = self.config['signal_thresholds']['WATCHLIST']
        
        if final_confidence >= good_threshold:
            signal = Signal.GOOD
        elif final_confidence >= watchlist_threshold:
            signal = Signal.WATCHLIST
        else:
            signal = Signal.IGNORE
        
        # Signal change logging - disabled by default
        if self.config.get('debug_signals', False) and self.multiplier_version != "none":
            confidence_without_multiplier = base_confidence
            signal_without = None
            if confidence_without_multiplier >= good_threshold:
                signal_without = "GOOD"
            elif confidence_without_multiplier >= watchlist_threshold:
                signal_without = "WATCHLIST"
            else:
                signal_without = "IGNORE"

            if signal.value != signal_without:
                print(f"SIGNAL CHANGE: {self.multiplier_version} | {symbol} | "
                    f"{signal_without} → {signal.value} | "
                    f"Conf: {confidence_without_multiplier:.1f} → {final_confidence:.1f}")
        
        if signal == Signal.IGNORE:
            return None
        # Short signal confluence gate
        # Diagnostic shows TRENDING_BEARISH trades in 75-79 confidence band
        # have 19% win rate - catastrophic. Require stronger confluence
        # confirmation for short signals to filter low-quality bearish setups.
        min_short_confluence = self.config.get('min_short_confluence', 70)
        if regime == MarketRegime.TRENDING_BEARISH and confluence_pct < min_short_confluence:
            return None
        
        entry_price = entry
        stop_loss = stop
        target_price = target
        
        return SignalResult(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal,
            confidence=final_confidence,
            confluence_pct=confluence_pct,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            regime=regime.value,
            base_score=base_confidence,
            multiplier_applied=multiplier
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else close.iloc[-1] * 0.01
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50