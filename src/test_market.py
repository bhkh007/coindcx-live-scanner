# test_market.py
from market_scanner import CoinDCXMarketScanner

def main():
    scanner = CoinDCXMarketScanner()
    scanner.scan_markets(top_n=10)

if __name__ == "__main__":
    main()
