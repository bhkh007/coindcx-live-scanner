# src/backtest_framework/historical_data_fetcher.py
"""
Historical Data Fetcher for Backtesting
---------------------------------------
Uses Binance public API - zero authentication required.
CoinDCX candles endpoint confirmed to:
  - Ignore startTime/endTime/from/to parameters entirely
  - Hard cap at 1000 most recent candles regardless of limit parameter
  - Provide no pagination mechanism
Binance klines endpoint supports full date-range pagination via
startTime/endTime and returns up to 1000 candles per request.
This file is used ONLY for backtesting. Live scanning continues
to use CoinDCXMarketData from market_data.py unchanged.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

BINANCE_BASE_URL      = "https://api.binance.com"
BINANCE_KLINES_PATH   = "/api/v3/klines"
MAX_CANDLES_PER_REQ   = 1000
REQUEST_DELAY_SECONDS = 0.25   # stay well under Binance 1200 req/min limit

# Maps your config timeframe strings to Binance interval strings
INTERVAL_MAP = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "8h":  "8h",
    "12h": "12h",
    "1d":  "1d",
}

# Milliseconds per candle for each interval - used to estimate coverage
MS_PER_INTERVAL = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "2h":  7_200_000,
    "4h":  14_400_000,
    "6h":  21_600_000,
    "8h":  28_800_000,
    "12h": 43_200_000,
    "1d":  86_400_000,
}


# ---------------------------------------------------------------------------
# FETCHER CLASS
# ---------------------------------------------------------------------------

class BinanceHistoricalFetcher:
    """
    Fetches full-range historical OHLCV data from Binance public API.
    Handles pagination automatically.
    Saves results to CSV. Validates coverage before returning cached data.
    """

    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "Accept":     "application/json",
            "User-Agent": "backtest-historical-fetcher/1.0",
        })

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------------

    def get_historical_data(
        self,
        symbol:        str,
        interval:      str,
        start_date:    str,
        end_date:      str,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame covering start_date to end_date (inclusive),
        plus a 60-day warmup buffer prepended before start_date so that
        indicators like EMA50 and ATR14 are fully warmed up at the first
        scan timestamp.

        Parameters
        ----------
        symbol        : "BTCUSDT", "ETHUSDT", etc.
        interval      : "1h", "4h", "15m", etc.
        start_date    : "YYYY-MM-DD"  (backtest start)
        end_date      : "YYYY-MM-DD"  (backtest end)
        force_refresh : bypass all caches and re-fetch from Binance

        Returns
        -------
        pd.DataFrame  columns: open, high, low, close, volume
                      index:   DatetimeIndex UTC
        None if fetch fails completely.
        """

        csv_path = self._csv_path(symbol, interval)

        # ---- try disk cache first ----------------------------------------
        if not force_refresh and os.path.exists(csv_path):
            cached = self._load_csv(csv_path)
            if cached is not None:
                if self._covers(cached, start_date, end_date):
                    print(
                        f"[Fetcher] Cache hit  | {symbol} {interval} | "
                        f"{cached.index.min().date()} → "
                        f"{cached.index.max().date()} | "
                        f"{len(cached):,} candles"
                    )
                    return self._slice(cached, start_date, end_date)

                print(
                    f"[Fetcher] Cache miss | {symbol} {interval} | "
                    f"cached {cached.index.min().date()} → "
                    f"{cached.index.max().date()} | "
                    f"need   {start_date} → {end_date} | "
                    f"re-fetching from Binance..."
                )

        # ---- fetch from Binance ------------------------------------------
        # Add 60-day warmup buffer before start_date
        fetch_start = (
            pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(days=60)
        ).strftime("%Y-%m-%d")

        df = self._paginate(symbol, interval, fetch_start, end_date)

        if df is None or df.empty:
            print(f"[Fetcher] FAILED | {symbol} {interval}")
            return None

        self._save_csv(df, csv_path)

        print(
            f"[Fetcher] Saved    | {symbol} {interval} | "
            f"{df.index.min().date()} → {df.index.max().date()} | "
            f"{len(df):,} candles | {csv_path}"
        )

        return self._slice(df, start_date, end_date)

    # -----------------------------------------------------------------------
    # PAGINATION ENGINE
    # -----------------------------------------------------------------------

    def _paginate(
        self,
        symbol:     str,
        interval:   str,
        start_date: str,
        end_date:   str,
    ) -> Optional[pd.DataFrame]:
        """
        Issues repeated requests to Binance, advancing the window by one
        millisecond past the last received candle each time, until
        end_date is reached or Binance returns fewer than MAX_CANDLES_PER_REQ.
        """

        binance_interval = INTERVAL_MAP.get(interval)
        if binance_interval is None:
            print(f"[Fetcher] Unknown interval '{interval}'")
            return None

        start_ms   = self._to_ms(start_date)
        end_ms     = self._to_ms(end_date, end_of_day=True)
        cursor_ms  = start_ms
        all_rows   = []
        req_num    = 0

        print(
            f"[Fetcher] Fetching | {symbol} {interval} | "
            f"{start_date} → {end_date}"
        )

        while cursor_ms < end_ms:

            batch = self._fetch_batch(
                symbol   = symbol,
                interval = binance_interval,
                start_ms = cursor_ms,
                end_ms   = end_ms,
            )

            # None means a hard API error - abort
            if batch is None:
                print(
                    f"[Fetcher] API error on request {req_num + 1} "
                    f"for {symbol}. Aborting pagination."
                )
                break

            # Empty list means no more data exists
            if len(batch) == 0:
                break

            all_rows.extend(batch)
            req_num += 1

            last_open_ms  = batch[-1][0]
            cursor_ms     = last_open_ms + 1
            last_date_str = pd.Timestamp(last_open_ms, unit="ms").date()

            print(
                f"[Fetcher]   req {req_num:3d} | "
                f"{len(batch):4d} candles | "
                f"total {len(all_rows):6,} | "
                f"up to {last_date_str}"
            )

            # Fewer than max means we have reached the end of available data
            if len(batch) < MAX_CANDLES_PER_REQ:
                break

            time.sleep(REQUEST_DELAY_SECONDS)

        if not all_rows:
            return None

        return self._parse(all_rows)

    def _fetch_batch(
        self,
        symbol:   str,
        interval: str,
        start_ms: int,
        end_ms:   int,
    ) -> Optional[list]:
        """Single HTTP request to Binance klines endpoint."""

        params = {
            "symbol":    symbol.upper(),
            "interval":  interval,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     MAX_CANDLES_PER_REQ,
        }

        for attempt in range(3):
            try:
                resp = self.session.get(
                    url     = f"{BINANCE_BASE_URL}{BINANCE_KLINES_PATH}",
                    params  = params,
                    timeout = 20,
                )

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(
                        f"[Fetcher] Rate limited (429). "
                        f"Waiting {wait}s before retry {attempt + 1}/3..."
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 451:
                    print(
                        f"[Fetcher] Binance geo-restricted (451) for "
                        f"{symbol}. "
                        f"If you are in a restricted region, use a VPN or "
                        f"switch to a different data source."
                    )
                    return None

                print(
                    f"[Fetcher] HTTP {resp.status_code} | "
                    f"{symbol} | {resp.text[:200]}"
                )
                return None

            except requests.exceptions.Timeout:
                print(
                    f"[Fetcher] Timeout on attempt {attempt + 1}/3 "
                    f"for {symbol}"
                )
                time.sleep(5)

            except requests.exceptions.RequestException as e:
                print(f"[Fetcher] Request error for {symbol}: {e}")
                return None

        return None

    # -----------------------------------------------------------------------
    # PARSING
    # -----------------------------------------------------------------------

    def _parse(self, raw: list) -> pd.DataFrame:
        """
        Converts raw Binance kline arrays to a clean OHLCV DataFrame.

        Binance kline array layout:
          [0]  open_time  (ms)
          [1]  open       (str)
          [2]  high       (str)
          [3]  low        (str)
          [4]  close      (str)
          [5]  volume     (str)
          [6..11] ignored
        """

        records = [
            {
                "timestamp": pd.Timestamp(c[0], unit="ms", tz="UTC"),
                "open":      float(c[1]),
                "high":      float(c[2]),
                "low":       float(c[3]),
                "close":     float(c[4]),
                "volume":    float(c[5]),
            }
            for c in raw
        ]

        df = pd.DataFrame(records).set_index("timestamp").sort_index()

        # Remove duplicates that can appear at pagination boundaries
        df = df[~df.index.duplicated(keep="first")]

        return df

    # -----------------------------------------------------------------------
    # CSV HELPERS
    # -----------------------------------------------------------------------

    def _csv_path(self, symbol: str, interval: str) -> str:
        return os.path.join(
            self.data_dir,
            f"{symbol}_{interval}_binance.csv"
        )

    def _save_csv(self, df: pd.DataFrame, path: str):
        df.to_csv(path)

    def _load_csv(self, path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(
                path,
                index_col  = "timestamp",
                parse_dates = True,
            )
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            required = {"open", "high", "low", "close", "volume"}
            missing  = required - set(df.columns)
            if missing:
                print(f"[Fetcher] CSV missing columns {missing}: {path}")
                return None

            return df

        except Exception as e:
            print(f"[Fetcher] CSV load error {path}: {e}")
            return None

    # -----------------------------------------------------------------------
    # DATE / COVERAGE UTILITIES
    # -----------------------------------------------------------------------

    def _to_ms(self, date_str: str, end_of_day: bool = False) -> int:
        """Converts 'YYYY-MM-DD' to UTC millisecond timestamp."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.timestamp() * 1000)

    def _covers(
        self,
        df:         pd.DataFrame,
        start_date: str,
        end_date:   str,
    ) -> bool:
        """
        Returns True if the cached DataFrame covers the required range.
        Allows a 2-day tolerance on the end boundary to handle weekends
        and exchange downtime.
        """
        if df is None or df.empty:
            return False

        need_start = pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(days=60)
        need_end   = pd.Timestamp(end_date,   tz="UTC")

        if df.index.min() > need_start:
            return False
        if df.index.max() < need_end - pd.Timedelta(days=2):
            return False

        return True

    def _slice(
        self,
        df:         pd.DataFrame,
        start_date: str,
        end_date:   str,
    ) -> pd.DataFrame:
        """
        Returns rows from 60 days before start_date through end_date.
        The 60-day prefix is the indicator warmup buffer.
        """
        buf_start = pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(days=60)
        end       = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=1)
        return df.loc[buf_start:end].copy()