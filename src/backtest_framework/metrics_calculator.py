# src/backtest_framework/metrics_calculator.py
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict

MIN_RELIABLE_SAMPLE = 30  # below this, stats are noise

class MetricsCalculator:
    """Calculate comprehensive trading metrics"""

    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade

    # -----------------------------------------------------------------------
    # PRIMARY METRICS
    # -----------------------------------------------------------------------

    def calculate_metrics(self, trades: List, version_name: str) -> Dict:
        """Calculate all performance metrics for a version"""

        if not trades:
            return {"error": "No trades", "version": version_name}

        total_trades = len(trades)
        winners = [t for t in trades if t.result == "WIN"]
        losers  = [t for t in trades if t.result == "LOSS"]

        win_count  = len(winners)
        loss_count = len(losers)
        win_rate   = win_count / total_trades if total_trades > 0 else 0

        r_values = [t.r_multiple for t in trades]
        avg_r    = np.mean(r_values)
        median_r = np.median(r_values)
        std_r    = np.std(r_values)

        avg_win  = np.mean([t.r_multiple for t in winners]) if winners else 0
        avg_loss = np.mean([abs(t.r_multiple) for t in losers]) if losers else 0

        gross_profit = sum(t.r_multiple for t in winners)
        gross_loss   = abs(sum(t.r_multiple for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # ---- correct compounded equity curve --------------------------------
        equity_curve = self._build_equity_curve(r_values)
        final_equity  = equity_curve[-1]
        net_return_pct = ((final_equity - self.initial_capital)
                          / self.initial_capital * 100)

        # ---- drawdown on actual equity curve --------------------------------
        max_drawdown_pct, max_drawdown_r = self._calculate_drawdown(
            r_values, equity_curve
        )

        # ---- Sharpe on per-trade returns ------------------------------------
        per_trade_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = (
            (np.mean(per_trade_returns) / np.std(per_trade_returns))
            * np.sqrt(252)
            if np.std(per_trade_returns) > 0
            else 0
        )

        # ---- losing streak --------------------------------------------------
        max_losing_streak = self._max_losing_streak(trades)

        # ---- confluence breakdown (requires audit fields) -------------------
        confluence_breakdown = self._confluence_breakdown(trades)

        # ---- confidence breakdown -------------------------------------------
        confidence_breakdown = self._confidence_breakdown(trades)

        return {
            "version":              version_name,
            "total_trades":         total_trades,
            "win_count":            win_count,
            "loss_count":           loss_count,
            "win_rate":             round(win_rate * 100, 2),
            "avg_r":                round(avg_r, 3),
            "median_r":             round(median_r, 3),
            "std_r":                round(std_r, 3),
            "avg_win_r":            round(avg_win, 3),
            "avg_loss_r":           round(avg_loss, 3),
            "profit_factor":        round(profit_factor, 3),
            "expectancy":           round(expectancy, 3),
            # R-based (for comparison across versions)
            "net_return_r":         round(sum(r_values), 2),
            # Correct compounded dollar return
            "net_return_pct":       round(net_return_pct, 2),
            "final_equity":         round(final_equity, 2),
            # Drawdown in both R and percentage
            "max_drawdown_r":       round(max_drawdown_r, 2),
            "max_drawdown_pct":     round(max_drawdown_pct, 2),
            "sharpe_ratio":         round(sharpe, 3),
            "max_losing_streak":    max_losing_streak,
            "best_trade":           round(max(r_values), 2),
            "worst_trade":          round(min(r_values), 2),
            # Breakdown tables for audit
            "confluence_breakdown": confluence_breakdown,
            "confidence_breakdown": confidence_breakdown,
            # Raw R series for Monte Carlo
            "_r_series":            r_values,
            "_trades":              trades,          # ← ADD THIS LINE
        }

    # -----------------------------------------------------------------------
    # EQUITY CURVE
    # -----------------------------------------------------------------------

    def _build_equity_curve(self, r_values: List[float]) -> np.ndarray:
        """
        Builds a compounded equity curve.
        Each trade risks risk_per_trade fraction of current equity.
        A WIN returns avg_win_r * risk_per_trade * equity.
        A LOSS loses 1R * risk_per_trade * equity.
        """
        equity = self.initial_capital
        curve  = [equity]

        for r in r_values:
            pnl    = r * self.risk_per_trade * equity
            equity = max(equity + pnl, 0.01)  # floor at 1 cent, not zero
            curve.append(equity)

        return np.array(curve)

    # -----------------------------------------------------------------------
    # DRAWDOWN
    # -----------------------------------------------------------------------

    def _calculate_drawdown(
        self,
        r_values: List[float],
        equity_curve: np.ndarray,
    ):
        """
        Returns (max_drawdown_pct, max_drawdown_r).
        max_drawdown_pct: largest peak-to-trough drop as % of peak equity
        max_drawdown_r:   same drop expressed in R-multiples
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns   = (equity_curve - running_max) / running_max * 100
        max_dd_pct  = float(np.min(drawdowns))

        # Express in R: how many R-multiples does the worst drawdown represent
        cumulative_r = np.cumsum([0] + r_values)
        running_max_r = np.maximum.accumulate(cumulative_r)
        drawdowns_r   = cumulative_r - running_max_r
        max_dd_r      = float(np.min(drawdowns_r))

        return max_dd_pct, max_dd_r

    # -----------------------------------------------------------------------
    # LOSING STREAK
    # -----------------------------------------------------------------------

    def _max_losing_streak(self, trades: List) -> int:
        max_streak     = 0
        current_streak = 0
        for t in trades:
            if t.result == "LOSS":
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    # -----------------------------------------------------------------------
    # CONFLUENCE BREAKDOWN
    # -----------------------------------------------------------------------

    def _confluence_breakdown(self, trades: List) -> Dict:
        """
        Groups trades by confluence bucket and calculates
        win rate, expectancy, profit factor per bucket.
        Requires Trade.confluence_pct field.
        """
        if not hasattr(trades[0], 'confluence_pct'):
            return {}

        buckets = {
            "40-49": [], "50-59": [], "60-69": [],
            "70-79": [], "80-89": [], "90+":   [],
        }

        for t in trades:
            c = t.confluence_pct
            if   c < 50:  buckets["40-49"].append(t)
            elif c < 60:  buckets["50-59"].append(t)
            elif c < 70:  buckets["60-69"].append(t)
            elif c < 80:  buckets["70-79"].append(t)
            elif c < 90:  buckets["80-89"].append(t)
            else:         buckets["90+"].append(t)

        return self._bucket_stats(buckets)

    # -----------------------------------------------------------------------
    # CONFIDENCE BREAKDOWN
    # -----------------------------------------------------------------------

    def _confidence_breakdown(self, trades: List) -> Dict:
        """
        Groups trades by final_confidence bucket.
        Includes regime split per bucket to expose directional bias.
        Requires Trade.final_confidence and Trade.regime fields.
        """
        if not hasattr(trades[0], 'final_confidence'):
            return {}

        buckets = {
            "50-54": [], "55-59": [], "60-64": [],
            "65-69": [], "70-74": [], "75-79": [], "80+": [],
        }

        for t in trades:
            c = t.final_confidence
            if   c < 55:  buckets["50-54"].append(t)
            elif c < 60:  buckets["55-59"].append(t)
            elif c < 65:  buckets["60-64"].append(t)
            elif c < 70:  buckets["65-69"].append(t)
            elif c < 75:  buckets["70-74"].append(t)
            elif c < 80:  buckets["75-79"].append(t)
            else:         buckets["80+"].append(t)

        result = self._bucket_stats(buckets)

        # Augment each bucket with regime split
        for label, bucket_trades in buckets.items():
            if not bucket_trades:
                continue
            longs  = [t for t in bucket_trades if t.regime == "TRENDING_BULLISH"]
            shorts = [t for t in bucket_trades if t.regime == "TRENDING_BEARISH"]

            def regime_stats(subset):
                if not subset:
                    return None
                wins = sum(1 for t in subset if t.result == "WIN")
                r_vals = [t.r_multiple for t in subset]
                avg_win  = float(np.mean([t.r_multiple for t in subset if t.result == "WIN"])) \
                        if any(t.result == "WIN" for t in subset) else 0.0
                avg_loss = float(np.mean([abs(t.r_multiple) for t in subset if t.result == "LOSS"])) \
                        if any(t.result == "LOSS" for t in subset) else 0.0
                wr = wins / len(subset)
                exp = (wr * avg_win) - ((1 - wr) * avg_loss)
                return {
                    "count":      len(subset),
                    "win_rate":   round(wr * 100, 1),
                    "expectancy": round(exp, 3),
                }

            if label in result and result[label].get("count", 0) > 0:
                result[label]["long_stats"]  = regime_stats(longs)
                result[label]["short_stats"] = regime_stats(shorts)

        return result
    
    def component_breakdown_by_confidence_bucket(
        self,
        trades: List,
        bucket_low: float,
        bucket_high: float,
        label: str = "",
    ) -> Dict:
        """
        For trades in a specific confidence range, shows the average
        component scores (structure, mtf, momentum, etc.) broken down
        by win vs loss. Requires Trade.base_confidence and audit fields.

        Use this to diagnose WHY a confidence bucket underperforms.
        """
        bucket_trades = [
            t for t in trades
            if bucket_low <= t.final_confidence < bucket_high
        ]

        if not bucket_trades:
            return {"count": 0, "label": label}

        winners = [t for t in bucket_trades if t.result == "WIN"]
        losers  = [t for t in bucket_trades if t.result == "LOSS"]

        def avg(lst, attr):
            vals = [getattr(t, attr) for t in lst if hasattr(t, attr)]
            return round(float(np.mean(vals)), 2) if vals else None

        return {
            "label":             label,
            "count":             len(bucket_trades),
            "win_count":         len(winners),
            "loss_count":        len(losers),
            "win_rate":          round(len(winners) / len(bucket_trades) * 100, 1),
            "avg_expectancy":    round(
                np.mean([t.r_multiple for t in bucket_trades]), 3
            ),
            "avg_confluence_pct_winners":  avg(winners, "confluence_pct"),
            "avg_confluence_pct_losers":   avg(losers,  "confluence_pct"),
            "avg_base_confidence_winners": avg(winners, "base_confidence"),
            "avg_base_confidence_losers":  avg(losers,  "base_confidence"),
            "avg_multiplier_winners":      avg(winners, "multiplier_applied"),
            "avg_multiplier_losers":       avg(losers,  "multiplier_applied"),
            # Confluence distribution within bucket
            "confluence_dist": self._sub_confluence_dist(bucket_trades),
            # Regime split
            "regime_dist": self._regime_dist(bucket_trades),
        }

    def _sub_confluence_dist(self, trades: List) -> Dict:
        """Confluence distribution within a trade subset."""
        bands = {"<50": 0, "50-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90+": 0}
        for t in trades:
            c = t.confluence_pct
            if   c < 50: bands["<50"]   += 1
            elif c < 60: bands["50-59"] += 1
            elif c < 70: bands["60-69"] += 1
            elif c < 80: bands["70-79"] += 1
            elif c < 90: bands["80-89"] += 1
            else:        bands["90+"]   += 1
        total = len(trades)
        return {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in bands.items() if v > 0
        }

    def _regime_dist(self, trades: List) -> Dict:
        """Regime split within a trade subset."""
        dist: Dict[str, Dict] = {}
        for t in trades:
            regime = getattr(t, "regime", "UNKNOWN")
            if regime not in dist:
                dist[regime] = {"count": 0, "wins": 0}
            dist[regime]["count"] += 1
            if t.result == "WIN":
                dist[regime]["wins"] += 1
        return {
            k: {
                "count":    v["count"],
                "win_rate": round(v["wins"] / v["count"] * 100, 1)
            }
            for k, v in dist.items()
        }

    # -----------------------------------------------------------------------
    # BUCKET STATS HELPER
    # -----------------------------------------------------------------------

    # In _bucket_stats, replace the profit_factor line and add sample size flag:

    def _bucket_stats(self, buckets: Dict) -> Dict:
        result = {}
        for label, bucket_trades in buckets.items():
            if not bucket_trades:
                result[label] = {"count": 0}
                continue

            r_vals   = [t.r_multiple for t in bucket_trades]
            wins     = [t for t in bucket_trades if t.result == "WIN"]
            losses   = [t for t in bucket_trades if t.result == "LOSS"]
            win_rate = len(wins) / len(bucket_trades)
            avg_win  = np.mean([t.r_multiple for t in wins])        if wins   else 0
            avg_loss = np.mean([abs(t.r_multiple) for t in losses]) if losses else 0
            gp       = sum(t.r_multiple for t in wins)
            gl       = abs(sum(t.r_multiple for t in losses))

            # Cap infinity: if no losses exist, PF is not infinite - it is
            # unreliable due to sample size. Cap at 9.99 and flag it.
            if gl > 0:
                pf = gp / gl
            elif gp > 0:
                pf = 9.99   # wins only - cap, do not report infinity
            else:
                pf = 0.0

            exp      = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            cum_r    = np.cumsum(r_vals)
            run_max  = np.maximum.accumulate(cum_r)
            dd       = float(np.min(cum_r - run_max))

            count    = len(bucket_trades)
            reliable = count >= MIN_RELIABLE_SAMPLE

            result[label] = {
                "count":          count,
                "win_rate":       round(win_rate * 100, 2),
                "expectancy":     round(exp, 3),
                "profit_factor":  round(pf, 3),
                "max_dd_r":       round(dd, 2),
                "reliable":       reliable,   # False = treat stats as indicative only
            }
        return result

    # -----------------------------------------------------------------------
    # MONTE CARLO
    # -----------------------------------------------------------------------

    def monte_carlo(
        self,
        r_series: List[float],
        n_simulations: int = 10000,
        n_trades: int = None,
    ) -> Dict:
        """
        Runs Monte Carlo simulation by randomly resampling the trade sequence.
        Returns probability of various drawdown levels and ruin.

        Parameters
        ----------
        r_series      : list of R-multiples from actual trades
        n_simulations : number of random sequences to simulate
        n_trades      : trades per simulation (defaults to len(r_series))
        """
        if not r_series:
            return {}

        r_array  = np.array(r_series)
        n        = n_trades or len(r_series)
        results  = {
            "final_equity":    [],
            "max_drawdown_pct": [],
            "ruin":            0,
        }

        for _ in range(n_simulations):
            # Resample with replacement
            sim_r  = np.random.choice(r_array, size=n, replace=True)
            equity = self.initial_capital
            peak   = equity
            max_dd = 0.0
            ruined = False

            for r in sim_r:
                equity = equity + r * self.risk_per_trade * equity
                if equity <= 0:
                    equity = 0
                    ruined = True
                    break
                if equity > peak:
                    peak = equity
                dd = (equity - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd

            results["final_equity"].append(equity)
            results["max_drawdown_pct"].append(max_dd)
            if ruined:
                results["ruin"] += 1

        eq_arr = np.array(results["final_equity"])
        dd_arr = np.array(results["max_drawdown_pct"])

        return {
            "n_simulations":         n_simulations,
            "n_trades":              n,
            # Equity distribution
            "median_final_equity":   round(float(np.median(eq_arr)), 2),
            "p10_final_equity":      round(float(np.percentile(eq_arr, 10)), 2),
            "p90_final_equity":      round(float(np.percentile(eq_arr, 90)), 2),
            # Drawdown distribution
            # dd_arr contains negative values (e.g. -51.3%)
            # p10 = worst 10% of outcomes (most negative)
            # p90 = best 10% of outcomes (least negative)
            "median_max_drawdown":   round(float(np.median(dd_arr)), 2),
            "worst_10pct_drawdown":  round(float(np.percentile(dd_arr, 10)), 2),
            "best_10pct_drawdown":   round(float(np.percentile(dd_arr, 90)), 2),
            # Probability thresholds
            "prob_drawdown_20pct":   round(float(np.mean(dd_arr < -20)) * 100, 1),
            "prob_drawdown_50pct":   round(float(np.mean(dd_arr < -50)) * 100, 1),
            "prob_drawdown_80pct":   round(float(np.mean(dd_arr < -80)) * 100, 1),
            "prob_ruin":             round(results["ruin"] / n_simulations * 100, 2),
        }

    # -----------------------------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------------------------

    def compare_versions(self, results: Dict[str, Dict]) -> pd.DataFrame:
        display_cols = [
            "total_trades", "win_count", "loss_count", "win_rate",
            "avg_r", "median_r", "std_r", "avg_win_r", "avg_loss_r",
            "profit_factor", "expectancy",
            "net_return_r", "net_return_pct", "final_equity",
            "max_drawdown_r", "max_drawdown_pct",
            "sharpe_ratio", "max_losing_streak",
            "best_trade", "worst_trade",
        ]
        rows = []
        for version_name, metrics in results.items():
            if "error" in metrics:
                continue
            row = {"version": version_name}
            for col in display_cols:
                row[col] = metrics.get(col, None)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("version")
        return df