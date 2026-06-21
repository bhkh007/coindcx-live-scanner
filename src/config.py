# config.py - Institutional Scanner Configuration
MARKET_REGIME = {
    "TRENDING_BULLISH": {
        "min_ema_gap": 0.5,  # minimum % gap between EMAs
        "min_price_above_ema": 0.2,  # % above EMA20
        "structure_required": ["HH", "HL"]  # Higher High + Higher Low
    },
    "TRENDING_BEARISH": {
        "min_ema_gap": 0.5,
        "min_price_below_ema": 0.2,
        "structure_required": ["LH", "LL"]
    },
    "RANGE": {
        "max_ema_slope": 0.3,  # maximum slope % for sideways
        "max_price_deviation": 3.0  # max % from range mean
    }
}

TIMEFRAME_WEIGHTS = {
    "4H": 0.40,  # 40% weight - highest timeframe dominates
    "1H": 0.30,  # 30% weight
    "15M": 0.20,  # 20% weight
    "5M": 0.10   # 10% weight - lowest influence
}

SCORING_WEIGHTS = {
    "market_structure": 0.30,
    "trend_alignment": 0.20,
    "momentum_displacement": 0.20,
    "liquidity": 0.10,
    "risk_reward": 0.10,
    "entry_quality": 0.07,
    "rsi_context": 0.03
}

LIQUIDITY = {
    "excellent_volume_ratio": 2.0,
    "good_volume_ratio": 1.3,
    "neutral_volume_ratio": 0.8,
    "min_24h_volume_usdt": 10_000_000,  # $10M minimum
    "min_avg_volume_5m": 100_000        # $100k per 5m candle
}

MOMENTUM = {
    "strong_displacement": 1.5,    # candle body > 1.5x average
    "moderate_displacement": 1.2,
    "min_volume_confirmation": 1.3  # volume spike required
}

RISK_REWARD = {
    "scalp_min": 1.5,
    "intraday_min": 2.0,
    "swing_min": 2.5,
    "excellent_rr": 3.0
}

ENTRY_QUALITY = {
    "max_pump_percent": 5.0,  # don't chase >5% without pullback
    "ideal_pullback_depth": [0.3, 0.7],  # 30-70% retracement ideal
    "min_retracement_candles": 3
}

SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 75,
    "BUY_WATCHLIST": 60,
    "STRONG_SELL": 75,
    "SELL_WATCHLIST": 60,
    "REVERSAL_BUY": 65,
    "REVERSAL_CONFIRMATION": ["support_zone", "bullish_divergence", "reversal_candle"]
}

# Add these to existing config.py
STABLECOINS = ["FDUSDUSDT", "USDCUSDT", "USDTUSDT", "BUSDUSDT", "DAIUSDT", "RLUSDUSDT",
               "FDUSD", "USDC", "TUSD", "DAI", "USDP", "UUSDT"]

MIN_RR_REQUIRED = 1.5  # Hard reject below this
MAX_ENTRY_DISTANCE_PCT = 3.0  # Entry must be within 3% of current price
ADX_TRENDING_THRESHOLD = 25
ATR_EXPANSION_THRESHOLD = 1.5

DEADCANDLEMARKETS = {
    "TONUSDT", "TAOUSDT", "ASTERUSDT", "LABUSDT", "HYPEUSDT", "ONDOUSDT","PIEVERSEUSDT"
}

DEBUG_MODE = False  # Set to True for verbose debugging

# Signal thresholds
MIN_CONFIDENCE_GOOD = 70
MIN_CONFIDENCE_WATCHLIST = 50
MIN_STRUCTURE_GOOD = 70
MIN_RR_SCORE_GOOD = 70
MIN_VOLUME_24H = 15_000_000
MIN_REGIME_CONFIDENCE_TRENDING = 70