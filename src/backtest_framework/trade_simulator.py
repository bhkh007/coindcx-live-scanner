# src/backtest_framework/trade_simulator.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from backtest_framework.signal_generator import SignalResult

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    stop_loss: float
    target_price: float
    risk_r: float           # Risk as fraction of entry price
    result: str             # 'WIN', 'LOSS', 'BREAKEVEN'
    r_multiple: float
    fees: float
    slippage: float
    # --- audit fields added ---
    confluence_pct: float   # confluence score at signal time
    base_confidence: float  # confidence before multiplier
    final_confidence: float # confidence after multiplier
    multiplier_applied: float
    regime: str             # 'TRENDING_BULLISH' or 'TRENDING_BEARISH'
    scan_time: datetime     # when the signal was generated

class TradeSimulator:
    """Simulate trades based on signals and subsequent price movement"""
    
    def __init__(self, config: Dict, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.trades = []
        self.open_positions = {}
        self.daily_trade_count = 0
        self.last_trade_day = None
        
    def simulate_trade(self, signal: SignalResult, market_data: pd.DataFrame) -> Optional[Trade]:
        """Simulate trade from entry to exit"""
        
        # Get price data after signal
        future_data = market_data[market_data.index > signal.timestamp]
        
        if len(future_data) < 10:
            return None
        
        # Determine direction
        if signal.regime == "TRENDING_BULLISH":
            direction = "LONG"
            # For long: price goes up to target, down to stop
            hits_target = future_data['high'] >= signal.target_price
            hits_stop = future_data['low'] <= signal.stop_loss
        else:
            direction = "SHORT"
            # For short: price goes down to target, up to stop
            hits_target = future_data['low'] <= signal.target_price
            hits_stop = future_data['high'] >= signal.stop_loss
            
        if direction == "LONG":
            if signal.stop_loss >= signal.entry_price or signal.target_price <= signal.entry_price:
                return None
        
        # Find first hit
        target_idx = hits_target.idxmax() if hits_target.any() else None
        stop_idx = hits_stop.idxmax() if hits_stop.any() else None
        
        # Calculate trade outcome
        if target_idx is None and stop_idx is None:
            # No exit within lookback period
            return None
        
        if target_idx is None:
            # Stop hit first
            exit_time = stop_idx
            exit_price = signal.stop_loss
            result = "LOSS"
        elif stop_idx is None:
            # Target hit first
            exit_time = target_idx
            exit_price = signal.target_price
            result = "WIN"
        else:
            # Both hit, take first
            if target_idx < stop_idx:
                exit_time = target_idx
                exit_price = signal.target_price
                result = "WIN"
            else:
                exit_time = stop_idx
                exit_price = signal.stop_loss
                result = "LOSS"
        
        # Calculate R-multiple
        if direction == "LONG":
            risk = signal.entry_price - signal.stop_loss
            if result == "WIN":
                reward = signal.target_price - signal.entry_price
            else:
                reward = exit_price - signal.entry_price
        else:  # SHORT
            risk = signal.stop_loss - signal.entry_price
            if result == "WIN":
                reward = signal.entry_price - signal.target_price
            else:
                reward = signal.entry_price - exit_price
        
        r_multiple = reward / risk if risk > 0 else 0
        
        # Apply slippage and fees
        slippage = signal.entry_price * self.config['slippage_pct']
        fee_rate = self.config['taker_fee']
        fees = signal.entry_price * fee_rate + exit_price * fee_rate
        
        # Update daily trade count
        trade_day = signal.timestamp.date()
        if self.last_trade_day is None or self.last_trade_day != trade_day:
            self.daily_trade_count = 0
            self.last_trade_day = trade_day
        
        if self.daily_trade_count >= self.config['max_daily_trades']:
            print(f"   Skipping trade for {signal.symbol} at {signal.timestamp}: Max daily trades reached.")
            return None
        
        self.daily_trade_count += 1

        return Trade(
            entry_time=signal.timestamp,
            exit_time=exit_time,
            symbol=signal.symbol,
            direction=direction,
            entry_price=signal.entry_price,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            risk_r=risk / signal.entry_price,
            result=result,
            r_multiple=r_multiple,
            fees=fees,
            slippage=slippage,
            # audit fields
            confluence_pct=signal.confluence_pct,
            base_confidence=signal.base_score,
            final_confidence=signal.confidence,
            multiplier_applied=signal.multiplier_applied,
            regime=signal.regime,
            scan_time=signal.timestamp,
        )
    
    def run_backtest(self, signals: List[SignalResult], market_data: pd.DataFrame) -> List[Trade]:
        """Run backtest on all signals"""
        
        self.trades = []
        
        # Sort signals by timestamp
        signals_sorted = sorted(signals, key=lambda x: x.timestamp)
        
        for signal in signals_sorted:
            # Clean up positions that have already closed
            self.open_positions = {
                k: v for k, v in self.open_positions.items()
                if v.exit_time > signal.timestamp
            }
            
            if len(self.open_positions) >= self.config['max_concurrent_trades']:
                continue

            # Simulate trade
            trade = self.simulate_trade(signal, market_data)
            if trade:
                self.trades.append(trade)
                # Use unique key so multiple positions on same symbol are tracked separately
                position_key = f"{signal.symbol}_{signal.timestamp}"
                self.open_positions[position_key] = trade

            # Clean expired positions BEFORE the concurrent check (move cleanup to top of loop)
        
        return self.trades