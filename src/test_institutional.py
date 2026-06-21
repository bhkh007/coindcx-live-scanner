# test_institutional.py
from institutional_scanner import InstitutionalScanner

def main():
    scanner = InstitutionalScanner()
    scanner.scan_markets(top_n=10)

if __name__ == "__main__":
    main()