# src/run_forward_validation.py
"""
Forward Validation Runner
-------------------------
Extends the backtest into new (out-of-sample) data beyond the in-sample
end date. Run this periodically (weekly or fortnightly) to accumulate
out-of-sample evidence before making a live deployment decision.

What this does:
  1. Resolves the forward window: in_sample_end_date → today (or
     forward_end_date if pinned in config).
  2. Forces the fetcher to re-download data for all symbols so the CSV
     cache is extended to cover the new period.
  3. Runs signal generation and trade simulation on the forward window
     ONLY - the in-sample period is never re-evaluated here.
  4. Runs two parallel versions:
       - Baseline: no 75-79 band gate (same as current backtest)
       - Gated:    75-79 band hard exclusion applied
  5. Compares both versions against the Window 3 conservative baseline.
  6. Tracks the 70-74 SHORT edge specifically (highest overfitting risk).
  7. Appends results to forward_validation_log.json for trend tracking
     across multiple runs.
  8. Prints a deployment readiness assessment.

Usage:
  python src/run_forward_validation.py

  Run this every 1-2 weeks. Do not change config parameters between runs.
  Changing parameters between runs invalidates the out-of-sample claim.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from backtest_framework.config import BACKTEST_CONFIG
from backtest_framework.data_loader import HistoricalDataLoader
from backtest_framework.signal_generator import BacktestSignalGenerator, Signal, SignalResult
from backtest_framework.trade_simulator import TradeSimulator
from backtest_framework.metrics_calculator import MetricsCalculator

# ---------------------------------------------------------------------------
# CONSTANTS - do not change between runs
# ---------------------------------------------------------------------------

LOG_FILE = "forward_validation_log.json"
RESULTS_DIR = "forward_validation_results"

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def resolve_forward_window(config: Dict) -> Tuple[str, str]:
    """
    Returns (forward_start, forward_end) as 'YYYY-MM-DD' strings.
    forward_start is the day after in_sample_end_date.
    forward_end is today if config says 'today', otherwise the pinned date.
    """
    in_sample_end = pd.Timestamp(config["in_sample_end_date"], tz="UTC")
    forward_start = (in_sample_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw_end = config.get("forward_end_date", "today")
    if raw_end == "today":
        forward_end = date.today().strftime("%Y-%m-%d")
    else:
        forward_end = raw_end

    return forward_start, forward_end


def check_minimum_window(forward_start: str, forward_end: str, min_days: int) -> bool:
    """Returns True if the forward window is long enough to be meaningful."""
    start = pd.Timestamp(forward_start, tz="UTC")
    end   = pd.Timestamp(forward_end,   tz="UTC")
    days  = (end - start).days
    if days < min_days:
        print(f"\n  WARNING: Forward window is only {days} days "
              f"(minimum {min_days} days required).")
        print(f"  Statistics will be unreliable at this sample size.")
        print(f"  Come back in {min_days - days} days for a meaningful read.")
        return False
    return True


def build_forward_config(base_config: Dict, forward_start: str, forward_end: str) -> Dict:
    """
    Returns a config dict scoped to the forward window only.
    Preserves all signal generation parameters unchanged.
    """
    cfg = {**base_config}
    cfg["start_date"] = forward_start
    cfg["end_date"]   = forward_end
    return cfg


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_forward_signals(
    config: Dict,
    data_loader: HistoricalDataLoader,
) -> List[SignalResult]:
    """
    Generates all GOOD signals in the forward window using the fixed config.
    Identical logic to run_backtest.py generate_signals() - no changes.
    """
    signal_generator = BacktestSignalGenerator(config, "none")
    all_signals = []

    start_date = pd.Timestamp(config["start_date"], tz="UTC")
    end_date   = pd.Timestamp(config["end_date"],   tz="UTC")
    scan_times = pd.date_range(start=start_date, end=end_date, freq="4h", tz="UTC")

    print(f"  Scan points : {len(scan_times)}")

    agg = {
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }

    for symbol in config["symbols"]:
        print(f"  Processing  : {symbol}...")

        market_data = data_loader.load_historical_data(symbol)
        if market_data is None or market_data.empty:
            print(f"  WARNING     : No data for {symbol}, skipping")
            continue

        for scan_time in scan_times:
            if market_data.index.tz is not None and scan_time.tzinfo is None:
                scan_time_aligned = scan_time.tz_localize("UTC")
            else:
                scan_time_aligned = scan_time

            data_up_to = market_data[market_data.index <= scan_time_aligned]
            if len(data_up_to) < 50:
                continue

            mtf_data = {
                "5m":  data_up_to.resample("5min").agg(agg).dropna(),
                "15m": data_up_to.resample("15min").agg(agg).dropna(),
                "1h":  data_up_to.copy(),
                "4h":  data_up_to.resample("4h").agg(agg).dropna(),
            }

            signal = signal_generator.generate_signal(scan_time, symbol, mtf_data)
            if signal and signal.signal == Signal.GOOD:
                all_signals.append(signal)

    good_count  = len(all_signals)
    long_count  = sum(1 for s in all_signals if s.regime == "TRENDING_BULLISH")
    short_count = sum(1 for s in all_signals if s.regime == "TRENDING_BEARISH")
    print(f"  GOOD signals: {good_count} "
          f"({long_count} LONG, {short_count} SHORT)")

    return all_signals


# ---------------------------------------------------------------------------
# TRADE SIMULATION
# ---------------------------------------------------------------------------

def simulate_forward_trades(
    signals: List[SignalResult],
    config: Dict,
    data_loader: HistoricalDataLoader,
) -> List:
    """
    Simulates trades from forward signals.
    Identical position management logic to run_backtest.py simulate_trades().
    """
    if not signals:
        return []

    market_data_cache = {}
    for symbol in set(s.symbol for s in signals):
        data = data_loader.load_historical_data(symbol)
        if data is not None and not data.empty:
            market_data_cache[symbol] = data

    all_signals_sorted = sorted(signals, key=lambda x: x.timestamp)
    simulator = TradeSimulator(config, data_loader)

    effective_max_positions = config.get("effective_max_positions", 5)
    risk_per_trade          = config.get("risk_per_trade", 0.005)
    max_heat                = config.get("max_portfolio_heat", 0.10)

    all_trades = []

    for signal in all_signals_sorted:
        market_data = market_data_cache.get(signal.symbol)
        if market_data is None or market_data.empty:
            continue

        signal_ts = pd.Timestamp(signal.timestamp)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")

        # Clean up expired positions
        simulator.open_positions = {
            k: v for k, v in simulator.open_positions.items()
            if _to_utc(pd.Timestamp(v.exit_time)) > signal_ts
        }

        if len(simulator.open_positions) >= effective_max_positions:
            continue

        current_heat = len(simulator.open_positions) * risk_per_trade
        if current_heat >= max_heat:
            continue

        trade = simulator.simulate_trade(signal, market_data)
        if trade:
            all_trades.append(trade)
            position_key = f"{signal.symbol}_{signal.timestamp}"
            simulator.open_positions[position_key] = trade

    return all_trades


# ---------------------------------------------------------------------------
# BAND GATE FILTER
# ---------------------------------------------------------------------------

def apply_75_79_gate(signals: List[SignalResult]) -> Tuple[List[SignalResult], int]:
    """
    Removes signals with confidence in [75, 80).
    Returns (filtered_signals, n_removed).
    """
    filtered = [s for s in signals if not (75 <= s.confidence < 80)]
    removed  = len(signals) - len(filtered)
    return filtered, removed


# ---------------------------------------------------------------------------
# METRICS EXTRACTION
# ---------------------------------------------------------------------------

def extract_direction_metrics(trades: List) -> Dict:
    """
    Extracts long/short split and the specific 70-74 SHORT edge metrics
    that are the highest overfitting risk.
    """
    long_trades  = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]

    def wr(subset):
        if not subset:
            return 0.0
        return sum(1 for t in subset if t.result == "WIN") / len(subset) * 100

    def exp(subset):
        if not subset:
            return 0.0
        wins   = [t for t in subset if t.result == "WIN"]
        losses = [t for t in subset if t.result == "LOSS"]
        avg_w  = np.mean([t.r_multiple for t in wins])   if wins   else 0.0
        avg_l  = np.mean([abs(t.r_multiple) for t in losses]) if losses else 0.0
        w_rate = len(wins) / len(subset)
        return round((w_rate * avg_w) - ((1 - w_rate) * avg_l), 3)

    # 70-74 SHORT edge - the specific metric we are tracking
    band_70_74_short = [
        t for t in trades
        if 70 <= t.final_confidence < 75 and t.direction == "SHORT"
    ]
    band_70_74_long = [
        t for t in trades
        if 70 <= t.final_confidence < 75 and t.direction == "LONG"
    ]

    # 75-79 band metrics (to measure gate effectiveness)
    band_75_79 = [
        t for t in trades
        if 75 <= t.final_confidence < 80
    ]

    return {
        "long_count":  len(long_trades),
        "short_count": len(short_trades),
        "long_wr":     round(wr(long_trades),  1),
        "short_wr":    round(wr(short_trades), 1),
        "long_exp":    exp(long_trades),
        "short_exp":   exp(short_trades),
        # 70-74 SHORT edge
        "band_70_74_short_count": len(band_70_74_short),
        "band_70_74_short_wr":    round(wr(band_70_74_short), 1),
        "band_70_74_short_exp":   exp(band_70_74_short),
        "band_70_74_long_count":  len(band_70_74_long),
        "band_70_74_long_wr":     round(wr(band_70_74_long), 1),
        "band_70_74_long_exp":    exp(band_70_74_long),
        # 75-79 band
        "band_75_79_count": len(band_75_79),
        "band_75_79_wr":    round(wr(band_75_79), 1),
        "band_75_79_exp":   exp(band_75_79),
    }


# ---------------------------------------------------------------------------
# BASELINE COMPARISON
# ---------------------------------------------------------------------------

def compare_to_baseline(metrics: Dict, baseline: Dict) -> Dict:
    sharpe_delta  = metrics.get("sharpe_ratio",  0) - baseline["sharpe"]
    exp_delta     = metrics.get("expectancy",    0) - baseline["expectancy"]
    wr_delta      = metrics.get("win_rate",      0) - baseline["win_rate"]

    sharpe_floor  = baseline["sharpe"]     * 0.60
    exp_floor     = baseline["expectancy"] * 0.60
    wr_floor      = baseline["win_rate"]   - 10.0

    return {
        "sharpe_delta":   round(float(sharpe_delta), 3),
        "exp_delta":      round(float(exp_delta),    3),
        "wr_delta":       round(float(wr_delta),     1),
        "sharpe_pass":    bool(metrics.get("sharpe_ratio", 0) >= sharpe_floor),
        "exp_pass":       bool(metrics.get("expectancy",   0) >= exp_floor),
        "wr_pass":        bool(metrics.get("win_rate",     0) >= wr_floor),
        "sharpe_floor":   round(float(sharpe_floor), 3),
        "exp_floor":      round(float(exp_floor),    3),
        "wr_floor":       round(float(wr_floor),     1),
    }


# ---------------------------------------------------------------------------
# DEPLOYMENT READINESS
# ---------------------------------------------------------------------------

def assess_deployment_readiness(
    baseline_metrics:   Dict,
    gated_metrics:      Dict,
    direction_baseline: Dict,
    direction_gated:    Dict,
    baseline_cmp:       Dict,
    gated_cmp:          Dict,
    total_oos_trades:   int,
    min_oos_trades:     int,
    config:             Dict,
) -> str:
    """
    Returns a deployment readiness verdict with reasoning.
    This is the structured decision framework - it removes emotion from
    the live deployment decision.
    """
    issues   = []
    warnings = []

    # --- Gate 1: Minimum trade count ---
    if total_oos_trades < min_oos_trades:
        issues.append(
            f"Insufficient OOS trades: {total_oos_trades}/{min_oos_trades} required. "
            f"Keep accumulating data."
        )

    # --- Gate 2: Expectancy must be positive ---
    if baseline_metrics.get("expectancy", 0) <= 0:
        issues.append(
            f"Baseline expectancy is negative or zero "
            f"({baseline_metrics.get('expectancy', 0):.3f}R). "
            f"System is not profitable OOS."
        )

    # --- Gate 3: Baseline comparison floors ---
    if not baseline_cmp["sharpe_pass"]:
        issues.append(
            f"Sharpe {baseline_metrics.get('sharpe_ratio', 0):.3f} is below "
            f"floor {baseline_cmp['sharpe_floor']:.3f} "
            f"(60% of Window 3 baseline {config['baseline']['sharpe']:.3f})."
        )
    if not baseline_cmp["exp_pass"]:
        issues.append(
            f"Expectancy {baseline_metrics.get('expectancy', 0):.3f}R is below "
            f"floor {baseline_cmp['exp_floor']:.3f}R "
            f"(60% of Window 3 baseline {config['baseline']['expectancy']:.3f}R)."
        )

    # --- Gate 4: 70-74 SHORT edge check ---
    s_count = direction_baseline.get("band_70_74_short_count", 0)
    s_exp   = direction_baseline.get("band_70_74_short_exp",   0.0)
    s_wr    = direction_baseline.get("band_70_74_short_wr",    0.0)

    if s_count >= 20:
        if s_exp < 0.10:
            issues.append(
                f"70-74 SHORT edge expectancy has collapsed OOS: "
                f"E {s_exp:.3f}R on {s_count} trades. "
                f"OOS baseline was +0.458R (96 trades, Jun 2026). "
                f"Edge is gone - do not deploy."
            )
        elif s_exp < 0.20:
            warnings.append(
                f"70-74 SHORT edge is degrading: "
                f"E {s_exp:.3f}R on {s_count} trades "
                f"(OOS baseline +0.458R). Monitor closely."
            )
        # else: edge is holding, no action needed
    else:
        warnings.append(
            f"70-74 SHORT edge: only {s_count} OOS trades "
            f"(need 20+ for reliable read). Cannot confirm edge yet."
        )

    # --- Gate 5: Max drawdown check ---
    max_dd = baseline_metrics.get("max_drawdown_pct", 0)
    if max_dd < -20.0:
        issues.append(
            f"Max drawdown {max_dd:.1f}% exceeds 20% threshold at 0.5% risk. "
            f"Monte Carlo showed 0% probability of this - investigate."
        )
    elif max_dd < -15.0:
        warnings.append(
            f"Max drawdown {max_dd:.1f}% is elevated. "
            f"Within Monte Carlo bounds but worth monitoring."
        )

    # --- Gate 6: 75-79 gate effectiveness (informational only) ---
    # Gate is confirmed OOS (+0.609 Sharpe, Jun 2026).
    # No longer raises a warning - it is enabled permanently in config.
    # Kept as a monitoring note in the print output but not in verdict logic.
    gated_sharpe    = gated_metrics.get("sharpe_ratio", 0)
    baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0)
    if gated_sharpe < baseline_sharpe - 0.3:
        # Only warn if the gate starts hurting - that would be a new finding
        warnings.append(
            f"75-79 gate is now HURTING performance OOS "
            f"({gated_sharpe - baseline_sharpe:.3f} Sharpe). "
            f"Consider disabling it - the in-sample improvement may have stopped generalising."
        )
    elif gated_sharpe < baseline_sharpe - 0.3:
        warnings.append(
            f"75-79 gate is hurting performance OOS "
            f"({gated_sharpe - baseline_sharpe:.3f} Sharpe). "
            f"The in-sample improvement may not generalise."
        )

    # --- Verdict ---
    if issues:
        verdict = "NOT READY"
        reason  = "Blocking issues must be resolved before live deployment:"
    elif warnings:
        verdict = "CONDITIONAL"
        reason  = "No blocking issues, but monitor these before deploying:"
    else:
        verdict = "READY"
        reason  = (
            f"All gates passed with {total_oos_trades} OOS trades. "
            f"System is ready for live deployment at 0.5% risk."
        )

    return verdict, reason, issues, warnings


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def load_log(log_file: str) -> List[Dict]:
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, "r") as f:
            return json.load(f)
    except Exception:
        return []


def append_log(log_file: str, entry: Dict):
    log = load_log(log_file)

    # Remove any existing entry for the same run_date before appending.
    # This ensures reruns on the same day overwrite rather than accumulate.
    run_date = entry.get("run_date")
    log = [e for e in log if e.get("run_date") != run_date]

    log.append(entry)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2, default=str)


def save_run_results(results_dir: str, run_date: str, data: Dict):
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"fv_{run_date}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Results saved to {filename}")


# ---------------------------------------------------------------------------
# PRINT HELPERS
# ---------------------------------------------------------------------------

def print_metrics_block(label: str, metrics: Dict, direction: Dict, cmp: Dict, baseline: Dict):
    print(f"\n  --- {label} ---")
    print(f"  Trades      : {metrics.get('total_trades', 0)}")
    print(f"  Win Rate    : {metrics.get('win_rate', 0):.2f}%  "
          f"(baseline {baseline['win_rate']:.1f}%  "
          f"delta {cmp['wr_delta']:+.1f}pp  "
          f"{'PASS' if cmp['wr_pass'] else 'FAIL'})")
    print(f"  Expectancy  : {metrics.get('expectancy', 0):.3f}R  "
          f"(baseline {baseline['expectancy']:.3f}R  "
          f"delta {cmp['exp_delta']:+.3f}R  "
          f"{'PASS' if cmp['exp_pass'] else 'FAIL'})")
    print(f"  Sharpe      : {metrics.get('sharpe_ratio', 0):.3f}  "
          f"(baseline {baseline['sharpe']:.3f}  "
          f"delta {cmp['sharpe_delta']:+.3f}  "
          f"{'PASS' if cmp['sharpe_pass'] else 'FAIL'})")
    print(f"  Max DD      : {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Net Return  : {metrics.get('net_return_r', 0):.1f}R")

    print(f"\n  Direction split:")
    print(f"    LONG  {direction['long_count']:>4} trades  "
          f"WR {direction['long_wr']:>5.1f}%  "
          f"E {direction['long_exp']:>+.3f}R")
    print(f"    SHORT {direction['short_count']:>4} trades  "
          f"WR {direction['short_wr']:>5.1f}%  "
          f"E {direction['short_exp']:>+.3f}R")


def print_70_74_short_block(direction_baseline: Dict, direction_gated: Dict, config: Dict):
    print(f"\n  --- 70-74 SHORT Edge (Highest Overfitting Risk) ---")
    print(f"  In-sample reference: 39 trades | WR 59.0% | E +1.064R")
    print()

    b = direction_baseline
    g = direction_gated

    print(f"  {'Metric':<30} {'Baseline (no gate)':>20} {'Gated (75-79 excl)':>20}")
    print(f"  {'-'*72}")
    print(f"  {'70-74 SHORT trades':<30} "
          f"{b['band_70_74_short_count']:>20} "
          f"{g['band_70_74_short_count']:>20}")
    print(f"  {'70-74 SHORT WR':<30} "
          f"{b['band_70_74_short_wr']:>19.1f}% "
          f"{g['band_70_74_short_wr']:>19.1f}%")
    print(f"  {'70-74 SHORT Expectancy':<30} "
          f"{b['band_70_74_short_exp']:>19.3f}R "
          f"{g['band_70_74_short_exp']:>19.3f}R")
    print(f"  {'70-74 LONG trades':<30} "
          f"{b['band_70_74_long_count']:>20} "
          f"{g['band_70_74_long_count']:>20}")
    print(f"  {'70-74 LONG WR':<30} "
          f"{b['band_70_74_long_wr']:>19.1f}% "
          f"{g['band_70_74_long_wr']:>19.1f}%")

    # Expectancy-based assessment - WR threshold removed because
    # in-sample 59% WR on 39 trades was overfitted.
    # OOS baseline established at 41.7% WR / +0.458R (96 trades, Jun 2026).
    assessment_cfg  = config.get("band_70_74_short_assessment", {})
    min_trades      = assessment_cfg.get("min_trades_for_evaluation", 20)
    holding_floor   = assessment_cfg.get("holding_exp_threshold",     0.30)
    degrading_floor = assessment_cfg.get("degrading_exp_threshold",   0.10)

    s_count = b["band_70_74_short_count"]
    s_wr    = b["band_70_74_short_wr"]
    s_exp   = b["band_70_74_short_exp"]

    if s_count < min_trades:
        print(f"\n  STATUS: Only {s_count} OOS trades - too few to evaluate edge.")
        print(f"  Need at least {min_trades} trades for a directional read.")
    elif s_exp >= holding_floor:
        print(f"\n  STATUS: Edge HOLDING "
              f"(WR {s_wr:.1f}%, E {s_exp:.3f}R on {s_count} trades).")
        print(f"  Positive expectancy confirmed OOS. Continue monitoring.")
    elif s_exp >= degrading_floor:
        print(f"\n  STATUS: Edge DEGRADING "
              f"(WR {s_wr:.1f}%, E {s_exp:.3f}R on {s_count} trades).")
        print(f"  Still positive but below OOS baseline of 0.458R. Monitor closely.")
    else:
        print(f"\n  STATUS: Edge GONE "
              f"(WR {s_wr:.1f}%, E {s_exp:.3f}R on {s_count} trades).")
        print(f"  Expectancy near zero or negative. Do not deploy.")


def print_75_79_gate_block(direction_baseline: Dict, direction_gated: Dict,
                            baseline_metrics: Dict, gated_metrics: Dict):
    print(f"\n  --- 75-79 Band Gate Effectiveness ---")
    print(f"  In-sample improvement: +0.798 Sharpe when 75-79 excluded")
    print()

    b_count = direction_baseline.get("band_75_79_count", 0)
    b_wr    = direction_baseline.get("band_75_79_wr",    0.0)
    b_exp   = direction_baseline.get("band_75_79_exp",   0.0)

    print(f"  75-79 band trades (baseline): {b_count}")
    print(f"  75-79 band WR               : {b_wr:.1f}%")
    print(f"  75-79 band Expectancy        : {b_exp:.3f}R")

    sharpe_delta = (gated_metrics.get("sharpe_ratio",  0)
                    - baseline_metrics.get("sharpe_ratio", 0))
    exp_delta    = (gated_metrics.get("expectancy",    0)
                    - baseline_metrics.get("expectancy",   0))

    print(f"\n  Gate effect on overall system:")
    print(f"    Sharpe delta    : {sharpe_delta:+.3f}")
    print(f"    Expectancy delta: {exp_delta:+.3f}R")

    if b_count < 10:
        print(f"\n  STATUS: Too few 75-79 trades to evaluate gate OOS.")
    elif sharpe_delta > 0.2:
        print(f"\n  STATUS: Gate is HELPING OOS (+{sharpe_delta:.3f} Sharpe).")
        print(f"  In-sample finding generalises. Recommend enabling permanently.")
    elif sharpe_delta > -0.1:
        print(f"\n  STATUS: Gate is NEUTRAL OOS ({sharpe_delta:+.3f} Sharpe).")
        print(f"  No harm, no clear benefit yet. Continue monitoring.")
    else:
        print(f"\n  STATUS: Gate is HURTING OOS ({sharpe_delta:+.3f} Sharpe).")
        print(f"  In-sample improvement does not generalise. Consider disabling.")


def print_trend_summary(log: List[Dict]):
    """Prints a trend table across all historical runs."""
    if len(log) < 2:
        return

    print(f"\n  --- Historical Run Trend ---")
    print(f"  {'Run Date':<12} {'Trades':>7} {'WR':>7} {'E (R)':>8} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'70-74S WR':>10} {'Verdict':>12}")
    print(f"  {'-'*80}")

    for entry in log[-10:]:   # show last 10 runs
        m = entry.get("baseline_metrics", {})
        d = entry.get("direction_baseline", {})
        print(
            f"  {entry.get('run_date', 'N/A'):<12} "
            f"{m.get('total_trades', 0):>7} "
            f"{m.get('win_rate', 0):>6.1f}% "
            f"{m.get('expectancy', 0):>8.3f} "
            f"{m.get('sharpe_ratio', 0):>8.3f} "
            f"{m.get('max_drawdown_pct', 0):>7.1f}% "
            f"{d.get('band_70_74_short_wr', 0):>9.1f}% "
            f"{entry.get('verdict', 'N/A'):>12}"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("FORWARD VALIDATION RUNNER")
    print("=" * 80)
    print(f"Run date    : {date.today()}")
    print(f"In-sample   : {BACKTEST_CONFIG['start_date']} → "
          f"{BACKTEST_CONFIG['in_sample_end_date']}")

    # --- Resolve forward window ---
    forward_start, forward_end = resolve_forward_window(BACKTEST_CONFIG)
    print(f"Forward     : {forward_start} → {forward_end}")

    forward_days = (
        pd.Timestamp(forward_end,   tz="UTC") -
        pd.Timestamp(forward_start, tz="UTC")
    ).days
    print(f"Window      : {forward_days} days")
    print(f"Baseline    : {BACKTEST_CONFIG['baseline']['source']}")
    print(f"             Sharpe {BACKTEST_CONFIG['baseline']['sharpe']} | "
          f"E {BACKTEST_CONFIG['baseline']['expectancy']}R | "
          f"WR {BACKTEST_CONFIG['baseline']['win_rate']}%")
    print("=" * 80)

    # --- Minimum window check ---
    min_days = BACKTEST_CONFIG.get("forward_min_days", 45)
    if not check_minimum_window(forward_start, forward_end, min_days):
        print("\n  Exiting. Run again when more data is available.")
        return

    # --- Build forward config ---
    forward_config = build_forward_config(BACKTEST_CONFIG, forward_start, forward_end)

    # --- Load data (force refresh to extend CSV cache to today) ---
    print(f"\n{'='*80}")
    print("DATA FETCH")
    print("=" * 80)
    print("  Forcing data refresh to extend cache to forward end date...")
    print("  This will download new candles from Binance for all symbols.")
    print("  Subsequent runs will use the updated cache.\n")

    data_loader = HistoricalDataLoader(forward_config)

    for symbol in forward_config["symbols"]:
        print(f"  Fetching {symbol}...")
        df = data_loader.data_loader_force_refresh(symbol) \
            if hasattr(data_loader, "data_loader_force_refresh") \
            else data_loader.fetcher.get_historical_data(
                symbol        = symbol,
                interval      = "1h",
                start_date    = forward_config["start_date"],
                end_date      = forward_config["end_date"],
                force_refresh = True,
            )
        if df is not None:
            print(f"  {symbol}: {len(df):,} candles fetched "
                  f"({df.index.min().date()} → {df.index.max().date()})")
        else:
            print(f"  {symbol}: FAILED - check network connection")

    # Reload data_loader with fresh cache
    data_loader = HistoricalDataLoader(forward_config)

    # --- Generate signals ---
    print(f"\n{'='*80}")
    print("SIGNAL GENERATION (Forward Window)")
    print("=" * 80)
    all_signals = generate_forward_signals(forward_config, data_loader)

    if not all_signals:
        print("\n  No signals generated in forward window.")
        print("  Possible causes:")
        print("  - Forward window too short (market data not yet available)")
        print("  - All symbols in non-trending regime")
        print("  - Data fetch failed for all symbols")
        return

    # --- Apply 75-79 gate ---
    gated_signals, n_removed = apply_75_79_gate(all_signals)
    print(f"\n  75-79 gate: {n_removed} signals removed "
          f"({len(gated_signals)} remaining)")

    # --- Simulate trades: baseline (no gate) ---
    print(f"\n{'='*80}")
    print("TRADE SIMULATION")
    print("=" * 80)

    print(f"\n  Simulating baseline (no 75-79 gate)...")
    baseline_trades = simulate_forward_trades(all_signals, forward_config, data_loader)
    print(f"  Baseline trades: {len(baseline_trades)}")

    print(f"\n  Simulating gated (75-79 excluded)...")
    gated_trades = simulate_forward_trades(gated_signals, forward_config, data_loader)
    print(f"  Gated trades   : {len(gated_trades)}")

    if not baseline_trades:
        print("\n  No trades simulated. Forward window may be too short.")
        return

    # --- Calculate metrics ---
    metrics_calc = MetricsCalculator(
        initial_capital = BACKTEST_CONFIG["initial_capital"],
        risk_per_trade  = BACKTEST_CONFIG["risk_per_trade"],
    )

    baseline_metrics = metrics_calc.calculate_metrics(baseline_trades, "Forward_Baseline")
    gated_metrics    = (
        metrics_calc.calculate_metrics(gated_trades, "Forward_Gated")
        if gated_trades
        else {"error": "No gated trades"}
    )

    direction_baseline = extract_direction_metrics(baseline_trades)
    direction_gated    = extract_direction_metrics(gated_trades) if gated_trades else {}

    baseline_cmp = compare_to_baseline(baseline_metrics, BACKTEST_CONFIG["baseline"])
    gated_cmp    = (
        compare_to_baseline(gated_metrics, BACKTEST_CONFIG["baseline"])
        if gated_trades
        else {}
    )

    # --- Load historical log for cumulative trade count ---
    log = load_log(LOG_FILE)
    prior_trades = sum(
        entry.get("baseline_metrics", {}).get("total_trades", 0)
        for entry in log
    )
    # Avoid double-counting if this run date already exists in log
    run_date_str = date.today().strftime("%Y-%m-%d")
    prior_trades_excl_today = sum(
        entry.get("baseline_metrics", {}).get("total_trades", 0)
        for entry in log
        if entry.get("run_date") != run_date_str
    )
    total_oos_trades = prior_trades_excl_today + len(baseline_trades)

    # --- Print results ---
    print(f"\n{'='*80}")
    print("FORWARD VALIDATION RESULTS")
    print("=" * 80)

    baseline_ref = BACKTEST_CONFIG["baseline"]
    print_metrics_block(
        "Baseline (no 75-79 gate)",
        baseline_metrics, direction_baseline, baseline_cmp, baseline_ref
    )

    if not isinstance(gated_metrics, dict) or "error" not in gated_metrics:
        print_metrics_block(
            "Gated (75-79 excluded)",
            gated_metrics, direction_gated, gated_cmp, baseline_ref
        )

    # --- 70-74 SHORT edge ---
    print(f"\n{'='*80}")
    print("70-74 SHORT EDGE TRACKING")
    print("=" * 80)
    print_70_74_short_block(direction_baseline, direction_gated, BACKTEST_CONFIG)

    # --- 75-79 gate effectiveness ---
    print(f"\n{'='*80}")
    print("75-79 BAND GATE EFFECTIVENESS")
    print("=" * 80)
    if not isinstance(gated_metrics, dict) or "error" not in gated_metrics:
        print_75_79_gate_block(
            direction_baseline, direction_gated,
            baseline_metrics, gated_metrics
        )

    # --- Deployment readiness ---
    print(f"\n{'='*80}")
    print("DEPLOYMENT READINESS ASSESSMENT")
    print("=" * 80)
    print(f"\n  Cumulative OOS trades (all runs): {total_oos_trades} "
          f"/ {BACKTEST_CONFIG['min_oos_trades_for_deployment']} required")

    verdict, reason, issues, warnings_list = assess_deployment_readiness(
        baseline_metrics   = baseline_metrics,
        gated_metrics      = gated_metrics if not isinstance(gated_metrics, dict)
                             or "error" not in gated_metrics else {},
        direction_baseline = direction_baseline,
        direction_gated    = direction_gated,
        baseline_cmp       = baseline_cmp,
        gated_cmp          = gated_cmp,
        total_oos_trades   = total_oos_trades,
        min_oos_trades     = BACKTEST_CONFIG["min_oos_trades_for_deployment"],
        config             = BACKTEST_CONFIG,
    )

    print(f"\n  VERDICT: {verdict}")
    print(f"  {reason}")

    if issues:
        print(f"\n  Blocking issues:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

    if warnings_list:
        print(f"\n  Warnings:")
        for i, w in enumerate(warnings_list, 1):
            print(f"    {i}. {w}")

    # --- Historical trend ---
    print(f"\n{'='*80}")
    print("HISTORICAL RUN TREND")
    print("=" * 80)
    print_trend_summary(log)

    # --- Save results ---
    run_entry = {
        "run_date":          run_date_str,
        "forward_start":     forward_start,
        "forward_end":       forward_end,
        "forward_days":      forward_days,
        "total_signals":     len(all_signals),
        "gated_signals":     len(gated_signals),
        "n_75_79_removed":   n_removed,
        "baseline_metrics":  {
            k: v for k, v in baseline_metrics.items()
            if not k.startswith("_")
        },
        "gated_metrics": {
            k: v for k, v in gated_metrics.items()
            if not k.startswith("_")
        } if not isinstance(gated_metrics, dict) or "error" not in gated_metrics else {},
        "direction_baseline": direction_baseline,
        "direction_gated":    direction_gated,
        "baseline_comparison": baseline_cmp,
        "gated_comparison":    gated_cmp,
        "total_oos_trades":    total_oos_trades,
        "verdict":             verdict,
    }

    append_log(LOG_FILE, run_entry)
    save_run_results(RESULTS_DIR, run_date_str, run_entry)

    print(f"\n{'='*80}")
    print(f"  Run complete. Next run recommended in 1-2 weeks.")
    print(f"  Do not change config parameters between runs.")
    print(f"  OOS trades needed for deployment decision: "
          f"{max(0, BACKTEST_CONFIG['min_oos_trades_for_deployment'] - total_oos_trades)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
