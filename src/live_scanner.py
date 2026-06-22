"""
live_scanner.py
---------------
Live signal scanner for paper trading.

Architecture:
  - Startup : fetch 300 1h candles per symbol from Binance (existing fetcher)
  - Schedule: every 4 hours at HH:02, fetch latest closed candle from CoinDCX
  - Stitch  : overwrite final Binance candle close with CoinDCX live price
  - Signal  : run identical signal generation code as generate_oos_signals()
  - Gate    : apply 75-79 confidence exclusion (same as OOS validation)
  - Output  : console print + append to live_signals_log.json

Data sources:
  - Historical warmup (300 candles): Binance REST (existing BinanceHistoricalFetcher)
  - Live candle (most recent closed) : CoinDCX REST candle endpoint
  - Price stitch: CoinDCX close overwrites final Binance candle close

Run:
  python src/live_scanner.py
  Runs indefinitely. Ctrl+C to stop.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from backtest_framework.config import BACKTEST_CONFIG
from backtest_framework.signal_generator import BacktestSignalGenerator, Signal, SignalResult
from backtest_framework.historical_data_fetcher import BinanceHistoricalFetcher
from sync_logs_to_excel import sync_json_to_excel

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

LIVE_CONFIG = {
    # Symbols to scan - must match backtest config
    "symbols": BACKTEST_CONFIG["symbols"],  # ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT"]

    # CoinDCX symbol mapping: Binance symbol -> CoinDCX instrument name
    # Format: B-{BASE}_{QUOTE} for third-party (Binance-underlying) futures
    "coindcx_symbol_map": {
        "BTCUSDT":  "B-BTC_USDT",
        "ETHUSDT":  "B-ETH_USDT",
        "SOLUSDT":  "B-SOL_USDT",
        "BNBUSDT":  "B-BNB_USDT",
        "NEARUSDT": "B-NEAR_USDT",
    },

    # How many 1h candles to keep in the rolling buffer per symbol
    # Signal generator needs ~50 minimum; 300 gives stable indicator warmup
    "lookback_candles": 300,

    # Scan interval in hours - must match backtest (4h)
    "scan_interval_hours": 4,

    # Minutes past the hour to poll (buffer for exchange candle settlement)
    "poll_offset_minutes": 2,

    # CoinDCX REST base URL
    "coindcx_base_url": "https://public.coindcx.com",

    # Candle interval for CoinDCX API
    # Check CoinDCX docs for exact interval string - common values: "1h", "60", "60m"
    "coindcx_interval": "1h",

    # Output log file
    "signals_log_file": "live_signals_log.json",

    # 75-79 confidence gate (same as OOS validation - permanent config)
    "apply_75_79_gate": True,

    # Minimum confidence to log a signal (must match backtest GOOD threshold)
    "good_threshold": BACKTEST_CONFIG["signal_thresholds"]["GOOD"],

    # Rate limit handling
    "rate_limit_retry_seconds": 60,
    "max_retries": 3,
}

# ---------------------------------------------------------------------------
# COINDCX CANDLE FETCHER
# ---------------------------------------------------------------------------

class CoinDCXCandleFetcher:
    """
    Fetches the most recently CLOSED 1h candle from CoinDCX REST API.

    CoinDCX candle endpoint (from API docs):
      GET /market_data/candles
      Params: pair, interval, limit, startTime, endTime (all optional except pair)

    Returns array of candles. We take the second-to-last entry (last entry
    is the still-forming candle). Timestamp is verified before use.
    """

    def __init__(self, config: Dict):
        self.base_url    = config["coindcx_base_url"]
        self.interval    = config["coindcx_interval"]
        self.symbol_map  = config["coindcx_symbol_map"]
        self.max_retries = config["max_retries"]
        self.retry_wait  = config["rate_limit_retry_seconds"]

    def fetch_latest_closed_candle(
        self,
        binance_symbol: str,
        expected_close_time: pd.Timestamp,
    ) -> Optional[Dict]:
        """
        Fetches the most recently closed 1h candle for the given symbol.

        Args:
            binance_symbol    : e.g. "BTCUSDT"
            expected_close_time: the candle open time we expect to be closed
                                 e.g. pd.Timestamp("2026-06-21 04:00:00", tz="UTC")

        Returns:
            Dict with keys: open, high, low, close, volume, timestamp
            None if fetch fails or timestamp does not match expected.
        """
        coindcx_symbol = self.symbol_map.get(binance_symbol)
        if not coindcx_symbol:
            print(f"  [CoinDCX] No symbol mapping for {binance_symbol}")
            return None

        url    = f"{self.base_url}/market_data/candles"
        params = {
            "pair":     coindcx_symbol,
            "interval": self.interval,
            "limit":    5,  # fetch last 5, take second-to-last (index -2)
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=10)

                # Rate limit handling
                if response.status_code == 429:
                    print(f"  [CoinDCX] Rate limited (429). "
                          f"Waiting {self.retry_wait}s before retry {attempt}/{self.max_retries}...")
                    time.sleep(self.retry_wait)
                    continue

                if response.status_code != 200:
                    print(f"  [CoinDCX] HTTP {response.status_code} for {binance_symbol} "
                          f"(attempt {attempt}/{self.max_retries})")
                    if attempt < self.max_retries:
                        time.sleep(5)
                    continue

                data = response.json()

                if not data or len(data) < 2:
                    print(f"  [CoinDCX] Insufficient candle data for {binance_symbol} "
                          f"(got {len(data) if data else 0} candles, need at least 2)")
                    return None

                # Second-to-last = most recently CLOSED candle
                # Last entry is still forming
                closed_candle_raw = data[-2]

                # Parse and verify timestamp
                candle = self._parse_candle(closed_candle_raw, binance_symbol)
                if candle is None:
                    return None

                # Timestamp verification: confirm this is the candle we expect
                candle_ts = candle["timestamp"]
                if not self._verify_timestamp(candle_ts, expected_close_time, binance_symbol):
                    return None

                return candle

            except requests.exceptions.Timeout:
                print(f"  [CoinDCX] Timeout for {binance_symbol} "
                      f"(attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    time.sleep(5)

            except requests.exceptions.ConnectionError as e:
                print(f"  [CoinDCX] Connection error for {binance_symbol}: {e} "
                      f"(attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    time.sleep(10)

            except Exception as e:
                print(f"  [CoinDCX] Unexpected error for {binance_symbol}: {e}")
                return None

        print(f"  [CoinDCX] All {self.max_retries} attempts failed for {binance_symbol}")
        return None

    def _parse_candle(self, raw: Dict, symbol: str) -> Optional[Dict]:
        """
        Parse a raw CoinDCX candle dict into a standardised format.

        CoinDCX candle fields (from API docs):
          open, high, low, close, volume, time (Unix ms timestamp)

        Adjust field names here if the actual API response differs.
        """
        try:
            # CoinDCX returns time in milliseconds Unix timestamp
            # Field name may be "time" or "open_time" - adjust if needed
            raw_time = raw.get("time") or raw.get("open_time") or raw.get("t")
            if raw_time is None:
                print(f"  [CoinDCX] Cannot find timestamp field in candle: {list(raw.keys())}")
                return None

            timestamp = pd.Timestamp(int(raw_time), unit="ms", tz="UTC")

            return {
                "timestamp": timestamp,
                "open":      float(raw.get("open",   raw.get("o", 0))),
                "high":      float(raw.get("high",   raw.get("h", 0))),
                "low":       float(raw.get("low",    raw.get("l", 0))),
                "close":     float(raw.get("close",  raw.get("c", 0))),
                "volume":    float(raw.get("volume", raw.get("v", 0))),
            }

        except Exception as e:
            print(f"  [CoinDCX] Failed to parse candle for {symbol}: {e} | raw={raw}")
            return None

    def _verify_timestamp(
    self,
    candle_ts: pd.Timestamp,
    expected_ts: pd.Timestamp,
    symbol: str,
    ) -> bool:
        diff_hours = abs((candle_ts - expected_ts).total_seconds()) / 3600

        # CoinDCX uses IST-offset 4h boundaries (shifted ~1h from UTC midnight).
        # Allow up to one full interval (4h) difference.
        # We only reject if the candle is more than 4h stale - that indicates
        # a genuine data problem (exchange returned yesterday's candle).
        if diff_hours <= 4.0:
            if diff_hours > 1.0:
                print(f"  [CoinDCX] {symbol}: candle boundary offset {diff_hours:.1f}h "
                    f"(CoinDCX uses IST-anchored intervals) - accepted")
            return True

        print(f"  [CoinDCX] Timestamp mismatch for {symbol}: "
            f"got {candle_ts}, expected {expected_ts} "
            f"(diff {diff_hours:.1f}h) - candle is stale, skipping")
        return False

# ---------------------------------------------------------------------------
# CANDLE BUFFER MANAGER
# ---------------------------------------------------------------------------

class CandleBufferManager:
    """
    Manages a rolling buffer of 1h OHLCV candles per symbol.

    Startup: populated from Binance historical data (300 candles)
    Updates: new closed candle from CoinDCX stitched onto buffer
    Stitch : CoinDCX close price overwrites the final Binance candle close
             to reflect actual tradable prices on CoinDCX
    """

    def __init__(self, symbols: List[str], lookback: int):
        self.symbols  = symbols
        self.lookback = lookback
        self.buffers: Dict[str, pd.DataFrame] = {}

    def initialise_from_binance(self, fetcher: BinanceHistoricalFetcher) -> bool:
        """
        Fetch historical warmup candles from Binance for all symbols.
        Returns True if all symbols loaded successfully.
        """
        print(f"\n  [Buffer] Initialising {self.lookback}-candle warmup from Binance...")
        success = True

        # Calculate start date: lookback candles * 1h = lookback hours ago
        now        = pd.Timestamp.now(tz="UTC")
        start_date = (now - pd.Timedelta(hours=self.lookback + 10)).strftime("%Y-%m-%d")
        end_date   = now.strftime("%Y-%m-%d")

        for symbol in self.symbols:
            try:
                df = fetcher.get_historical_data(
                    symbol        = symbol,
                    interval      = "1h",
                    start_date    = start_date,
                    end_date      = end_date,
                    force_refresh = True,
                )

                if df is None or df.empty:
                    print(f"  [Buffer] FAILED to load {symbol} from Binance")
                    success = False
                    continue

                # Keep only the most recent `lookback` candles
                df = df.sort_index().tail(self.lookback).copy()

                # Ensure standard column names
                df.columns = [c.lower() for c in df.columns]
                self.buffers[symbol] = df

                print(f"  [Buffer] {symbol}: {len(df)} candles loaded "
                      f"({df.index.min().date()} → {df.index.max().date()})")

            except Exception as e:
                print(f"  [Buffer] Error loading {symbol}: {e}")
                success = False

        return success

    def stitch_and_update(
        self,
        symbol: str,
        coindcx_candle: Dict,
    ) -> bool:
        """
        Stitch CoinDCX closed candle onto the Binance buffer.

        Steps:
        1. Build a new row from the CoinDCX candle
        2. Overwrite the close of the final existing row with CoinDCX close
           (reflects actual tradable price on CoinDCX)
        3. Append the new CoinDCX row to the buffer
        4. Drop oldest row to maintain rolling window size

        Returns True if stitch succeeded.
        """
        if symbol not in self.buffers:
            print(f"  [Buffer] No buffer for {symbol} - cannot stitch")
            return False

        buf = self.buffers[symbol]

        # Step 2: overwrite final Binance candle close with CoinDCX close
        # This ensures the last indicator value reflects CoinDCX price
        if len(buf) > 0:
            buf.iloc[-1, buf.columns.get_loc("close")] = coindcx_candle["close"]

        # Step 3: build new row from CoinDCX candle and append
        new_row = pd.DataFrame(
            [{
                "open":   coindcx_candle["open"],
                "high":   coindcx_candle["high"],
                "low":    coindcx_candle["low"],
                "close":  coindcx_candle["close"],
                "volume": coindcx_candle["volume"],
            }],
            index=[coindcx_candle["timestamp"]],
        )
        new_row.index.name = buf.index.name

        buf = pd.concat([buf, new_row])

        # Step 4: maintain rolling window
        buf = buf.sort_index().tail(self.lookback)

        self.buffers[symbol] = buf
        return True

    def get_mtf_data(self, symbol: str, scan_time: pd.Timestamp) -> Optional[Dict]:
        """
        Build the multi-timeframe data dict for signal generation.
        Identical structure to generate_oos_signals() in run_backtest.py.
        """
        if symbol not in self.buffers:
            return None

        buf = self.buffers[symbol]
        if len(buf) < 50:
            print(f"  [Buffer] {symbol}: only {len(buf)} candles - need 50+ for signal generation")
            return None

        # Filter to data up to scan_time (no lookahead)
        if buf.index.tz is not None and scan_time.tzinfo is None:
            scan_time = scan_time.tz_localize("UTC")
        elif buf.index.tz is None and scan_time.tzinfo is not None:
            scan_time = scan_time.tz_localize(None)

        data_up_to = buf[buf.index <= scan_time]
        if len(data_up_to) < 50:
            return None

        agg = {
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }

        return {
            "5m":  data_up_to.resample("5min").agg(agg).dropna(),
            "15m": data_up_to.resample("15min").agg(agg).dropna(),
            "1h":  data_up_to.copy(),
            "4h":  data_up_to.resample("4h").agg(agg).dropna(),
        }


# ---------------------------------------------------------------------------
# SIGNAL LOGGER
# ---------------------------------------------------------------------------

class SignalLogger:
    """
    Logs fired signals to console and live_signals_log.json.

    Log entry format matches paper trading tracking needs:
      - timestamp, symbol, direction, confidence, entry_price
      - stop_loss, take_profit (derived from signal if available)
      - gate_applied, run_id
    """

    def __init__(self, log_file: str):
        self.log_file = log_file

    def log_signal(self, signal: SignalResult, scan_time: pd.Timestamp, gate_applied: bool):
        """Print to console and append to JSON log."""

        entry  = getattr(signal, "entry_price",  None)
        stop   = getattr(signal, "stop_loss",    None)
        target = getattr(signal, "target_price", None)  # FIXED: was "take_profit"

        direction = "LONG" if "BULLISH" in str(getattr(signal, "regime", "")) else "SHORT"

        # Gate label: gate active means 75-79 band was excluded.
        # Signals reaching this method have already passed the gate.
        if gate_applied:
            gate_label = f"75-79 gate active | confidence {signal.confidence:.1f} PASSED"
        else:
            gate_label = "no gate"

        # Console output
        print(f"\n  {'='*60}")
        print(f"  SIGNAL FIRED")
        print(f"  {'='*60}")
        print(f"  Time       : {scan_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Symbol     : {signal.symbol}")
        print(f"  Direction  : {direction}")
        print(f"  Confidence : {signal.confidence:.1f}")
        print(f"  Entry      : {entry}")
        print(f"  Stop Loss  : {stop}")
        print(f"  Take Profit: {target}")
        print(f"  Gate       : {gate_label}")
        print(f"  {'='*60}")

        # Derive confidence band for paper trading split analysis
        conf = float(signal.confidence)
        if conf < 70:
            confidence_band = "<70"
        elif conf < 75:
            confidence_band = "70-74"
        elif conf < 80:
            confidence_band = "75-79"
        elif conf < 85:
            confidence_band = "80-84"
        else:
            confidence_band = "85+"

        # JSON log entry
        entry_dict = {
            "scan_time":        scan_time.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol":           signal.symbol,
            "direction":        direction,
            "confidence":       round(conf, 2),
            "confidence_band":  confidence_band,          # AUTO-POPULATED
            "entry_price":      float(entry)  if entry  is not None else None,
            "stop_loss":        float(stop)   if stop   is not None else None,
            "take_profit":      float(target) if target is not None else None,
            "regime":           str(getattr(signal, "regime", "")),
            "gate_active":      gate_applied,
            "gate_status":      "passed" if gate_applied else "not_applied",
            "logged_at":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            # Paper trading outcome fields - filled in manually
            "actual_fill":      None,   # Your actual CoinDCX entry price
            "outcome":          None,   # WIN / LOSS / OPEN
            "actual_exit":      None,
            "r_multiple":       None,   # (actual_exit - actual_fill) / (actual_fill - stop_loss)
            "notes":            None,
        }

        self._append_log(entry_dict)

    def log_scan_summary(
        self,
        scan_time: pd.Timestamp,
        n_signals: int,
        n_removed_gate: int,
        symbols_scanned: List[str],
    ):
        """Log a scan cycle summary even when no signals fire."""
        print(f"\n  [Scanner] {scan_time.strftime('%Y-%m-%d %H:%M UTC')} | "
              f"Scanned: {', '.join(symbols_scanned)} | "
              f"Signals: {n_signals} fired | "
              f"Gate removed: {n_removed_gate}")

        summary = {
            "scan_time":       scan_time.strftime("%Y-%m-%d %H:%M:%S"),
            "type":            "scan_summary",
            "symbols_scanned": symbols_scanned,
            "signals_fired":   n_signals,
            "gate_removed":    n_removed_gate,
            "logged_at":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._append_log(summary)

    def _append_log(self, entry: Dict):
        """Append entry to JSON log file."""
        log = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    log = json.load(f)
            except Exception:
                log = []

        log.append(entry)

        with open(self.log_file, "w") as f:
            json.dump(log, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# LIVE SCANNER
# ---------------------------------------------------------------------------

class LiveScanner:
    """
    Main live scanner orchestrator.

    Runs indefinitely on a 4-hour schedule (at HH:02).
    Each cycle:
      1. Determine expected closed candle time
      2. Fetch closed candle from CoinDCX for each symbol
      3. Stitch onto Binance buffer
      4. Run signal generator (identical to generate_oos_signals)
      5. Apply 75-79 gate
      6. Log signals to console + JSON
    """

    def __init__(self, config: Dict):
        self.config          = config
        self.signal_gen_config = self._build_signal_gen_config()
        data_directory = config.get("data_dir", "historical_data")
        self.fetcher = BinanceHistoricalFetcher(data_dir=data_directory)
        # self.fetcher         = BinanceHistoricalFetcher(config)
        self.coindcx_fetcher = CoinDCXCandleFetcher(config)
        self.buffer_manager  = CandleBufferManager(
            symbols  = config["symbols"],
            lookback = config["lookback_candles"],
        )
        self.signal_generator = BacktestSignalGenerator(self.signal_gen_config, "none")
        self.logger           = SignalLogger(config["signals_log_file"])

    def _build_signal_gen_config(self) -> Dict:
        """
        Build a config dict for BacktestSignalGenerator.
        Inherits all signal parameters from BACKTEST_CONFIG.
        Dates are set to today (not used in live mode, but required by constructor).
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {
            **BACKTEST_CONFIG,
            "start_date": today,
            "end_date":   today,
            "multiplier_versions": {"none": BACKTEST_CONFIG["multiplier_versions"]["none"]},
        }

    def initialise(self) -> bool:
        """
        Load historical warmup buffers from Binance.
        Must succeed before scanner can run.
        """
        print("\n" + "=" * 60)
        print("LIVE SCANNER - INITIALISING")
        print("=" * 60)
        print(f"  Symbols     : {', '.join(self.config['symbols'])}")
        print(f"  Lookback    : {self.config['lookback_candles']} candles")
        print(f"  Interval    : {self.config['scan_interval_hours']}h")
        print(f"  Poll offset : +{self.config['poll_offset_minutes']}min past hour")
        print(f"  Gate        : 75-79 exclusion = {self.config['apply_75_79_gate']}")
        print(f"  Log file    : {self.config['signals_log_file']}")

        ok = self.buffer_manager.initialise_from_binance(self.fetcher)
        if not ok:
            print("\n  [Scanner] INIT FAILED - one or more symbols could not be loaded")
            print("  Check network connection and Binance API availability")
            return False

        print(f"\n  [Scanner] Initialisation complete. Ready to scan.")
        return True

    def run_once(self, scan_time: pd.Timestamp) -> Tuple[int, int]:
        """
        Run a single scan cycle at the given scan_time.

        Returns (n_signals_fired, n_removed_by_gate).
        """
        print(f"\n  [Scanner] Running scan cycle: "
              f"{scan_time.strftime('%Y-%m-%d %H:%M UTC')}")

        # Expected closed candle: the 4h candle that just closed
        # e.g. if scan_time is 04:02, the closed candle opened at 00:00 and closed at 04:00
        interval_hours = self.config["scan_interval_hours"]
        expected_candle_open = scan_time.floor(f"{interval_hours}h")

        all_signals:   List[SignalResult] = []
        symbols_ok:    List[str]          = []

        for symbol in self.config["symbols"]:
            # Fetch latest closed candle from CoinDCX
            candle = self.coindcx_fetcher.fetch_latest_closed_candle(
                binance_symbol    = symbol,
                expected_close_time = expected_candle_open,
            )

            if candle is None:
                print(f"  [Scanner] {symbol}: skipping - CoinDCX fetch failed")
                continue

            # Stitch onto buffer
            ok = self.buffer_manager.stitch_and_update(symbol, candle)
            if not ok:
                print(f"  [Scanner] {symbol}: skipping - buffer stitch failed")
                continue

            # Build MTF data dict
            mtf_data = self.buffer_manager.get_mtf_data(symbol, scan_time)
            if mtf_data is None:
                print(f"  [Scanner] {symbol}: skipping - insufficient buffer data")
                continue

            # Generate signal - identical logic to generate_oos_signals()
            signal = self.signal_generator.generate_signal(scan_time, symbol, mtf_data)

            if signal and signal.signal == Signal.GOOD:
                all_signals.append(signal)

            symbols_ok.append(symbol)

        # Apply 75-79 gate (same as OOS validation)
        n_removed = 0
        if self.config["apply_75_79_gate"]:
            gated_signals = [s for s in all_signals if not (75 <= s.confidence < 80)]
            n_removed     = len(all_signals) - len(gated_signals)
        else:
            gated_signals = all_signals

        # Log each fired signal
        for signal in gated_signals:
            self.logger.log_signal(
                signal      = signal,
                scan_time   = scan_time,
                gate_applied = self.config["apply_75_79_gate"],
            )

        # Log scan summary (even if no signals)
        self.logger.log_scan_summary(
            scan_time       = scan_time,
            n_signals       = len(gated_signals),
            n_removed_gate  = n_removed,
            symbols_scanned = symbols_ok,
        )

        # Auto-sync signals to Excel after every scan cycle
        try:
            sync_json_to_excel(
                json_path  = self.config["signals_log_file"],
                excel_path = "paper_trading_tracker.xlsx",
            )
        except Exception as e:
            print(f"  [Sync] Warning: Excel sync failed this cycle: {e}")
            print(f"  [Sync] Scanner continues normally. Run sync_logs_to_excel.py manually to recover.")

        return len(gated_signals), n_removed

    def run_forever(self):
        """
        Main loop. Runs indefinitely on a 4-hour schedule.

        Schedule: fires at 00:02, 04:02, 08:02, 12:02, 16:02, 20:02 UTC
        Ctrl+C to stop cleanly.
        """
        print("\n" + "=" * 60)
        print("LIVE SCANNER - RUNNING")
        print("=" * 60)
        print("  Schedule: every 4h at HH:02 UTC")
        print("  Press Ctrl+C to stop\n")

        interval_hours  = self.config["scan_interval_hours"]
        offset_minutes  = self.config["poll_offset_minutes"]

        while True:
            now = pd.Timestamp.now(tz="UTC")

            # Calculate next fire time: next 4h boundary + offset_minutes
            # e.g. if now is 03:45, next boundary is 04:00, fire at 04:02
            next_boundary = (now + pd.Timedelta(hours=interval_hours)).floor(
                f"{interval_hours}h"
            )
            # Recalculate: floor to next 4h boundary from now
            hours_since_midnight = now.hour + now.minute / 60
            next_boundary_hour   = (
                int(hours_since_midnight // interval_hours) + 1
            ) * interval_hours

            if next_boundary_hour >= 24:
                # Roll over to next day
                next_fire = (now + pd.Timedelta(days=1)).normalize() + pd.Timedelta(
                    minutes=offset_minutes
                )
            else:
                next_fire = now.normalize() + pd.Timedelta(
                    hours=next_boundary_hour,
                    minutes=offset_minutes,
                )

            wait_seconds = (next_fire - now).total_seconds()

            print(f"  [Scheduler] Next scan: {next_fire.strftime('%Y-%m-%d %H:%M UTC')} "
                  f"(in {wait_seconds/60:.1f} min)")

            try:
                time.sleep(max(0, wait_seconds))
            except KeyboardInterrupt:
                print("\n  [Scanner] Stopped by user. Exiting cleanly.")
                break

            # Run scan cycle
            try:
                scan_time = pd.Timestamp.now(tz="UTC")
                self.run_once(scan_time)
            except KeyboardInterrupt:
                print("\n  [Scanner] Stopped by user. Exiting cleanly.")
                break
            except Exception as e:
                print(f"\n  [Scanner] ERROR in scan cycle: {e}")
                print(f"  [Scanner] Continuing to next cycle...")
                import traceback
                traceback.print_exc()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    scanner = LiveScanner(LIVE_CONFIG)

    # Initialise warmup buffers
    if not scanner.initialise():
        sys.exit(1)

    # Option: run once immediately for testing, then start the loop
    # Uncomment the line below to run a single scan cycle right now
    #scanner.run_once(pd.Timestamp.now(tz="UTC"))

    # Run indefinitely on schedule
    scanner.run_forever()


if __name__ == "__main__":
    main()
