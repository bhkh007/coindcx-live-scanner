# src/backtest_framework/data_loader.py
"""
Historical Data Loader
----------------------
Wraps BinanceHistoricalFetcher and provides the interface that
signal_generator.py and run_backtest.py expect.
"""

import pandas as pd
from typing import Dict, Optional

from backtest_framework.historical_data_fetcher import BinanceHistoricalFetcher


class HistoricalDataLoader:

    def __init__(self, config: Dict):
        self.config  = config
        self.fetcher = BinanceHistoricalFetcher(data_dir="historical_data")
        # In-memory cache so each symbol is loaded from disk only once
        self._cache: Dict[str, pd.DataFrame] = {}
        self._reported_cache_hits: set = set()   # ADD THIS

    # -----------------------------------------------------------------------
    # PRIMARY METHOD - called by run_backtest.py
    # -----------------------------------------------------------------------

    def load_historical_data(
        self,
        symbol:        str,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Returns 1h OHLCV DataFrame for symbol covering the config date
        range plus a 60-day warmup buffer.
        """

        if not force_refresh and symbol in self._cache:
            df = self._cache[symbol]
            if symbol not in self._reported_cache_hits:
                print(
                    f"[DataLoader] Memory hit | {symbol} | "
                    f"{len(df):,} candles | "
                    f"{df.index.min().date()} → {df.index.max().date()}"
                )
                self._reported_cache_hits.add(symbol)
            return df

        df = self.fetcher.get_historical_data(
            symbol        = symbol,
            interval      = "1h",
            start_date    = self.config["start_date"],
            end_date      = self.config["end_date"],
            force_refresh = force_refresh,
        )

        if df is None or df.empty:
            print(f"[DataLoader] No data for {symbol}")
            return None

        if len(df) < 50:
            print(
                f"[DataLoader] Insufficient candles for {symbol}: "
                f"{len(df)} (need ≥ 50)"
            )
            return None

        df = self._clean(df, symbol)
        if df is None:
            return None

        self._cache[symbol] = df
        return df

    # -----------------------------------------------------------------------
    # MULTI-TIMEFRAME - called by signal_generator.py
    # -----------------------------------------------------------------------

    def get_multi_timeframe_data(
        self,
        symbol:    str,
        timestamp: pd.Timestamp,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Returns OHLCV data resampled to 5m / 15m / 1h / 4h,
        all cut off at timestamp to prevent lookahead bias.
        """

        df = self.load_historical_data(symbol)
        if df is None or df.empty:
            return None

        # Ensure timezone consistency
        ts = timestamp
        if df.index.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        elif df.index.tz is None and ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        data_until = df[df.index <= ts]
        if len(data_until) < 50:
            return None

        agg = {
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }

        try:
            mtf = {
                "5m":  data_until.resample("5min").agg(agg).dropna(),
                "15m": data_until.resample("15min").agg(agg).dropna(),
                "1h":  data_until.copy(),
                "4h":  data_until.resample("4h").agg(agg).dropna(),
            }
        except Exception as e:
            print(f"[DataLoader] Resample error {symbol} @ {ts}: {e}")
            return None

        minimums = {"5m": 20, "15m": 20, "1h": 50, "4h": 10}
        for tf, min_n in minimums.items():
            if len(mtf[tf]) < min_n:
                return None

        return mtf

    def clear_cache(self):
        self._cache.clear()

    # -----------------------------------------------------------------------
    # INTERNAL
    # -----------------------------------------------------------------------

    def _clean(
        self,
        df:     pd.DataFrame,
        symbol: str,
    ) -> Optional[pd.DataFrame]:
        """Removes rows with corrupt OHLCV values."""

        n_before = len(df)

        price_cols = ["open", "high", "low", "close"]
        df = df[(df[price_cols] > 0).all(axis=1)]
        df = df[df["high"] >= df["low"]]
        df = df[(df["close"] >= df["low"]) & (df["close"] <= df["high"])]
        df = df[(df["open"]  >= df["low"]) & (df["open"]  <= df["high"])]

        removed = n_before - len(df)
        if removed > 0:
            print(f"[DataLoader] Removed {removed} corrupt rows from {symbol}")

        if df.empty:
            print(f"[DataLoader] All rows corrupt for {symbol}")
            return None

        return df
