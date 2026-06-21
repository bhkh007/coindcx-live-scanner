# src/validation_tracker.py
"""
Validation Tracker
------------------
Reads forward_validation_log.json and produces a summary of how the
system is performing across all forward validation runs.

Run this any time to see the current state of validation without
re-running the full forward validation.

Usage:
  python src/validation_tracker.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from typing import List, Dict

from backtest_framework.config import BACKTEST_CONFIG

LOG_FILE = "forward_validation_log.json"


def load_log(log_file: str) -> List[Dict]:
    if not os.path.exists(log_file):
        print(f"No log file found at {log_file}.")
        print("Run src/run_forward_validation.py first.")
        return []
    with open(log_file, "r") as f:
        return json.load(f)


def print_header():
    print("=" * 80)
    print("FORWARD VALIDATION TRACKER")
    print("=" * 80)
    baseline = BACKTEST_CONFIG["baseline"]
    print(f"  In-sample period : {BACKTEST_CONFIG['start_date']} → "
          f"{BACKTEST_CONFIG['in_sample_end_date']}")
    print(f"  Baseline (W3)    : Sharpe {baseline['sharpe']} | "
          f"E {baseline['expectancy']}R | WR {baseline['win_rate']}%")
    print(f"  Deployment gate  : {BACKTEST_CONFIG['min_oos_trades_for_deployment']} "
          f"OOS trades minimum")
    print("=" * 80)


def print_run_summary(log: List[Dict]):
    if not log:
        print("\n  No runs recorded yet.")
        return

    print(f"\n  Total runs recorded: {len(log)}")
    print(f"  First run : {log[0].get('run_date', 'N/A')}")
    print(f"  Last run  : {log[-1].get('run_date', 'N/A')}")

    total_oos = log[-1].get("total_oos_trades", 0) if log else 0
    min_req   = BACKTEST_CONFIG["min_oos_trades_for_deployment"]
    remaining = max(0, min_req - total_oos)
    pct_done  = min(100.0, total_oos / min_req * 100)

    print(f"\n  OOS Trade Progress:")
    bar_len   = 40
    filled    = int(bar_len * pct_done / 100)
    bar       = "#" * filled + "-" * (bar_len - filled)
    print(f"  [{bar}] {pct_done:.1f}%")
    print(f"  {total_oos} / {min_req} trades  ({remaining} remaining)")


def print_metrics_trend(log: List[Dict]):
    if not log:
        return

    print(f"\n{'='*80}")
    print("METRICS TREND (Baseline - no 75-79 gate)")
    print("=" * 80)

    baseline = BACKTEST_CONFIG["baseline"]

    print(f"\n  {'Run Date':<12} {'Days':>5} {'Trades':>7} "
          f"{'WR':>7} {'E (R)':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Verdict':>12}")
    print(f"  {'-'*75}")

    for entry in log:
        m = entry.get("baseline_metrics", {})
        print(
            f"  {entry.get('run_date', 'N/A'):<12} "
            f"{entry.get('forward_days', 0):>5} "
            f"{m.get('total_trades', 0):>7} "
            f"{m.get('win_rate', 0):>6.1f}% "
            f"{m.get('expectancy', 0):>8.3f} "
            f"{m.get('sharpe_ratio', 0):>8.3f} "
            f"{m.get('max_drawdown_pct', 0):>7.1f}% "
            f"{entry.get('verdict', 'N/A'):>12}"
        )

    # Trend direction (last 3 runs)
    if len(log) >= 3:
        recent = log[-3:]
        sharpes = [e.get("baseline_metrics", {}).get("sharpe_ratio", 0) for e in recent]
        exps    = [e.get("baseline_metrics", {}).get("expectancy",   0) for e in recent]

        sharpe_trend = "improving" if sharpes[-1] > sharpes[0] else "declining"
        exp_trend    = "improving" if exps[-1]    > exps[0]    else "declining"

        print(f"\n  Trend (last 3 runs):")
        print(f"    Sharpe     : {sharpe_trend} "
              f"({sharpes[0]:.3f} → {sharpes[-1]:.3f})")
        print(f"    Expectancy : {exp_trend} "
              f"({exps[0]:.3f}R → {exps[-1]:.3f}R)")


def print_70_74_short_trend(log: List[Dict]):
    if not log:
        return

    print(f"\n{'='*80}")
    print("70-74 SHORT EDGE TREND (Highest Overfitting Risk)")
    print("=" * 80)
    print(f"  In-sample reference: 39 trades | WR 59.0% | E +1.064R")
    print()

    print(f"  {'Run Date':<12} {'Trades':>7} {'WR':>8} {'E (R)':>8} {'Status':>12}")
    print(f"  {'-'*55}")

    for entry in log:
        d = entry.get("direction_baseline", {})
        count = d.get("band_70_74_short_count", 0)
        wr    = d.get("band_70_74_short_wr",    0.0)
        exp   = d.get("band_70_74_short_exp",   0.0)

        if count < 10:
            status = "TOO FEW"
        elif exp >= 0.30:
            status = "HOLDING"
        elif exp >= 0.10:
            status = "DEGRADING"
        else:
            status = "GONE"

        print(
            f"  {entry.get('run_date', 'N/A'):<12} "
            f"{count:>7} "
            f"{wr:>7.1f}% "
            f"{exp:>8.3f} "
            f"{status:>12}"
        )

    # Cumulative across all runs
    all_short_trades = []
    for entry in log:
        d = entry.get("direction_baseline", {})
        count = d.get("band_70_74_short_count", 0)
        wr    = d.get("band_70_74_short_wr",    0.0)
        # Approximate: reconstruct wins from WR and count
        wins  = round(count * wr / 100)
        all_short_trades.extend(["WIN"] * wins + ["LOSS"] * (count - wins))

    if all_short_trades:
        # Note: cumulative_wr is approximate (reconstructed from per-run WR)
        # Use the latest run's expectancy as the primary assessment metric
        latest_exp = log[-1].get("direction_baseline", {}).get(
            "band_70_74_short_exp", 0.0
        ) if log else 0.0
        cumulative_count = len(all_short_trades)
        cumulative_wr    = (sum(1 for r in all_short_trades if r == "WIN")
                            / cumulative_count * 100)

        print(f"\n  Cumulative OOS: {cumulative_count} trades | "
              f"WR ~{cumulative_wr:.1f}% | Latest E {latest_exp:.3f}R")
        print(f"  OOS baseline established: 41.7% WR / +0.458R "
              f"(96 trades, Jun 2026)")

        if cumulative_count >= 20:
            if latest_exp >= 0.30:
                print(f"  ASSESSMENT: Edge holding. Positive expectancy confirmed OOS.")
            elif latest_exp >= 0.10:
                print(f"  ASSESSMENT: Edge degrading but still positive. Monitor.")
            else:
                print(f"  ASSESSMENT: Edge gone. Expectancy near zero or negative.")
        else:
            print(f"  ASSESSMENT: Need {20 - cumulative_count} more trades.")


def print_75_79_gate_trend(log: List[Dict]):
    if not log:
        return

    print(f"\n{'='*80}")
    print("75-79 BAND GATE TREND")
    print("=" * 80)
    print(f"  In-sample improvement: +0.798 Sharpe when gate enabled")
    print()

    print(f"  {'Run Date':<12} {'Removed':>8} "
          f"{'Base Sharpe':>12} {'Gate Sharpe':>12} "
          f"{'Delta':>8} {'Status':>12}")
    print(f"  {'-'*70}")

    for entry in log:
        bm = entry.get("baseline_metrics", {})
        gm = entry.get("gated_metrics",    {})
        removed = entry.get("n_75_79_removed", 0)

        b_sharpe = bm.get("sharpe_ratio", 0)
        g_sharpe = gm.get("sharpe_ratio", 0) if gm else 0
        delta    = g_sharpe - b_sharpe

        if not gm:
            status = "NO DATA"
        elif delta > 0.2:
            status = "HELPING"
        elif delta > -0.1:
            status = "NEUTRAL"
        else:
            status = "HURTING"

        print(
            f"  {entry.get('run_date', 'N/A'):<12} "
            f"{removed:>8} "
            f"{b_sharpe:>12.3f} "
            f"{g_sharpe:>12.3f} "
            f"{delta:>+8.3f} "
            f"{status:>12}"
        )


def print_deployment_status(log: List[Dict]):
    if not log:
        return

    print(f"\n{'='*80}")
    print("DEPLOYMENT STATUS")
    print("=" * 80)

    latest = log[-1]
    total_oos = latest.get("total_oos_trades", 0)
    min_req   = BACKTEST_CONFIG["min_oos_trades_for_deployment"]
    verdict   = latest.get("verdict", "UNKNOWN")

    print(f"\n  Latest verdict    : {verdict}")
    print(f"  As of             : {latest.get('run_date', 'N/A')}")
    print(f"  OOS trades        : {total_oos} / {min_req}")

    # Count consecutive positive runs
    positive_runs = 0
    for entry in reversed(log):
        if entry.get("verdict") in ("READY", "CONDITIONAL"):
            positive_runs += 1
        else:
            break

    print(f"  Consecutive non-blocking runs: {positive_runs}")

    if verdict == "READY" and total_oos >= min_req:
        print(f"\n  SYSTEM IS READY FOR LIVE DEPLOYMENT")
        print(f"  Deploy at 0.5% risk per trade, max 5 concurrent positions.")
        print(f"  Start with 25% of intended capital for the first 2 weeks.")
    elif verdict == "CONDITIONAL":
        print(f"\n  System is close. Address warnings before deploying.")
    else:
        print(f"\n  System is NOT ready. Continue accumulating OOS data.")
        remaining = max(0, min_req - total_oos)
        print(f"  Estimated runs remaining: ~{max(1, remaining // 50)} "
              f"(assuming ~50 trades per run)")


def main():
    log = load_log(LOG_FILE)
    print_header()
    print_run_summary(log)
    print_metrics_trend(log)
    print_70_74_short_trend(log)
    print_75_79_gate_trend(log)
    print_deployment_status(log)
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
