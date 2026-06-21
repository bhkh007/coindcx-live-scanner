# indicators.py

import pandas as pd

class TechnicalIndicators:

    @staticmethod
    def calculate_rsi(closes, period=14):

        series = pd.Series(closes)

        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        return round(rsi.iloc[-1], 2)

    @staticmethod
    def get_trend(closes):

        """ Determine trend using EMA alignment
        Returns: BULLISH BEARISH SIDEWAYS """

        if len(closes) < 50:
            return "SIDEWAYS"

        series = pd.Series(closes)

        ema20 = series.ewm(span=20).mean()
        ema50 = series.ewm(span=50).mean()

        current_price = series.iloc[-1]

        ema20_last = ema20.iloc[-1]
        ema50_last = ema50.iloc[-1]

        # Strong bullish trend
        if current_price > ema20_last > ema50_last:
            return "BULLISH"

        # Strong bearish trend
        elif current_price < ema20_last < ema50_last:
            return "BEARISH"

        return "SIDEWAYS"