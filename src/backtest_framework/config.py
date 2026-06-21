# src/backtest_framework/config.py
"""
Backtest Configuration
"""

BACKTEST_CONFIG = {
    # Date range for backtest
    "start_date": "2025-01-05",
    "end_date": "2025-07-05",

    # Symbols to test
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT"],
    
    # Timeframes
    "timeframes": ["5m", "15m", "1h", "4h"],
    
    # Multiplier versions to test
    "multiplier_versions": {
        "extreme": {
            "name": "Extreme Multiplier",
            "multiplier_func": "extreme_multiplier"
        },
        "current": {
            "name": "Current Multiplier",
            "multiplier_func": "current_multiplier"
        },
        "soft": {
            "name": "Soft Multiplier",
            "multiplier_func": "soft_multiplier"
        },
        "linear": {
            "name": "Linear Multiplier",
            "multiplier_func": "linear_multiplier"
        },
        "none": {
            "name": "No Multiplier",
            "multiplier_func": "none_multiplier"
        }
    },
    
    # Signal thresholds
    "signal_thresholds": {
        # Raised from 65 to 73 based on threshold sweep evidence:
        # Threshold_73: Sharpe 2.199, 270 trades (best reliable result)
        # Threshold_70: Sharpe 1.641, 290 trades (previous baseline)
        # Improvement: +0.558 Sharpe with only 20 fewer trades
        "GOOD": 73,
        "WATCHLIST": 50
    },

    # Risk parameters
    "risk_per_trade": 0.005,
    "max_concurrent_trades": 50,
    "max_daily_trades": 100,
    "effective_max_positions": 5,
    
    # Commission
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    
    # Slippage
    "slippage_pct": 0.0005,
    
    # Backtest options
    "reinvest_profits": True,
    "compound_returns": True,
    "initial_capital": 10000,

    # Debug options
    "debug_signals": False,

    # Risk investigation
    "risk_per_trade_live": 0.005,

    # Threshold experiments
    "threshold_experiments": [73, 74, 75, 76, 77, 78, 79, 80],

    # Portfolio heat limit
    "max_portfolio_heat": 0.10,
    
    "min_mtf_score": 75,
    "min_short_confluence": 70,

    # ------------------------------------------------------------------
    # FORWARD VALIDATION SETTINGS
    # Added for run_forward_validation.py
    # ------------------------------------------------------------------

    # The in-sample period end date. Forward validation starts the day
    # after this. Do not change this once set - it is the fixed boundary
    # between in-sample and out-of-sample data.
    "in_sample_end_date": "2025-07-05",

    # Forward validation end date. Set to "today" to always run to the
    # current date, or pin to a specific date for reproducibility.
    # "today" is resolved at runtime in run_forward_validation.py.
    "forward_end_date": "today",

    # Minimum candles required in the forward window before running.
    # At 4h scan frequency, 45 days = ~270 scan points per symbol.
    # Below this the trade count will be too low for reliable statistics.
    "forward_min_days": 45,

    # Window 3 conservative baseline - the benchmark all forward results
    # are compared against. These values come from the walk-forward audit
    # (Jan-Jul 2025, Window 3: May-Jul 2025 test period).
    # Do not update these values based on forward results - they are the
    # fixed reference point established before any forward data was seen.
    "baseline": {
        "sharpe":      2.070,
        "expectancy":  0.217,
        "win_rate":    34.8,
        "source":      "Walk-forward Window 3 (May-Jul 2025 test period)",
    },

    # 75-79 confidence band gate.
    # True  = hard exclusion (reject all signals with confidence 75-79).
    #         Proven in-sample: +0.798 Sharpe improvement.
    # False = no exclusion (baseline behaviour, same as current backtest).
    # Both modes are run in parallel in run_forward_validation.py so you
    # can see the gate's effect on live data without committing to it.
    "exclude_75_79_band": True,

    # Minimum out-of-sample trades required before making a live
    # deployment decision. Below this threshold the statistics are
    # unreliable regardless of how good the numbers look.
    "min_oos_trades_for_deployment": 200,
    # Controls whether run_oos_validation() is called at the end of main().
    # Set to False when you want an in-sample-only run without waiting for
    # the data fetch. True = always run OOS after the backtest.
    "run_forward_validation": True,
    
    "band_70_74_short_assessment": {
        "min_trades_for_evaluation":  20,
        "holding_exp_threshold":       0.30,   # E >= 0.30R = holding
        "degrading_exp_threshold":     0.10,   # E >= 0.10R = degrading
        # below 0.10R expectancy = collapsed (edge is gone)
        "note": "WR-based assessment removed - in-sample 59% WR was overfitted. "
                "OOS baseline established at 41.7% WR / 0.458R expectancy (96 trades)."
    },
}
