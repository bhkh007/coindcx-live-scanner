import re
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any

class HistoricalSignalAnalyzer:
    def __init__(self):
        self.signals = []
        
    def parse_terminal_output(self, log_text: str) -> List[Dict]:
        """Parse terminal output to extract signal data"""
        
        signals = []
        
        # Pattern to match signal blocks
        signal_pattern = r"={80,}\n📊 (\w+) \| (\w+) \| Confidence: ([\d.]+)/100\n={80,}\n🎯 Setup: (\w+) \| RR: ([\d.]+)\n💰 Current: ([\d.]+) \| Entry: ([\d.]+) \([+-][\d.]+%\)\n🛑 Stop: ([\d.]+) \| 🎯 Target: ([\d.]+)\n\n📋 CONFLUENCE: ([\d.]+)/(\d+) \(([\d.]+)%\)"
        
        matches = re.findall(signal_pattern, log_text, re.MULTILINE)
        
        for match in matches:
            signal_data = {
                "symbol": match[0],
                "signal": match[1],
                "confidence": float(match[2]),
                "regime": match[3],
                "rr_ratio": float(match[4]),
                "price": float(match[5]),
                "entry": float(match[6]),
                "stop_loss": float(match[7]),
                "target": float(match[8]),
                "confluence_score": float(match[9]),
                "confluence_max": float(match[10]),
                "confluence_pct": float(match[11]),
                "timestamp": datetime.now().timestamp()
            }
            signals.append(signal_data)
        
        return signals
    
    def parse_confidence_breakdowns(self, log_text: str) -> List[Dict]:
        """Parse confidence breakdown from logs"""
        
        breakdowns = []
        
        # Pattern for confidence breakdown
        breakdown_pattern = r"🔍 Confidence breakdown for (\w+): ({[^}]+})"
        
        matches = re.findall(breakdown_pattern, log_text)
        
        for match in matches:
            try:
                # Convert string dict to actual dict
                breakdown_str = match[1].replace("'", '"')
                breakdown = json.loads(breakdown_str)
                breakdown["symbol"] = match[0]
                breakdowns.append(breakdown)
            except:
                pass
        
        return breakdowns
    
    def add_historical_entry(self, symbol: str, confidence: float, signal: str, 
                             confluence_pct: float, base_score: float = None):
        """Manually add a historical entry"""
        
        self.signals.append({
            "symbol": symbol,
            "confidence": confidence,
            "signal": signal,
            "confluence_pct": confluence_pct,
            "base_score": base_score,
            "timestamp": datetime.now().timestamp()
        })
    
    def save_to_json(self, filename: str = "historical_signals.json"):
        """Save collected signals to JSON"""
        
        with open(filename, 'w') as f:
            json.dump(self.signals, f, indent=2)
        print(f"✅ Saved {len(self.signals)} signals to {filename}")
    
    def load_from_json(self, filename: str = "historical_signals.json"):
        """Load historical signals from JSON"""
        
        with open(filename, 'r') as f:
            self.signals = json.load(f)
        print(f"✅ Loaded {len(self.signals)} signals from {filename}")
    
    def analyze_multiplier_impact(self):
        """Analyze how multiplier affects confidence"""
        
        print("\n" + "="*60)
        print("MULTIPLIER IMPACT ANALYSIS")
        print("="*60)
        
        impacts = []
        for signal in self.signals:
            if signal.get('base_score') and signal.get('confluence_pct'):
                base = signal['base_score']
                final = signal['confidence']
                reduction = base - final
                reduction_pct = (reduction / base) * 100
                
                impacts.append({
                    "symbol": signal['symbol'],
                    "base": base,
                    "final": final,
                    "reduction": reduction,
                    "reduction_pct": reduction_pct,
                    "confluence": signal['confluence_pct']
                })
        
        if impacts:
            avg_reduction = sum(i['reduction'] for i in impacts) / len(impacts)
            avg_reduction_pct = sum(i['reduction_pct'] for i in impacts) / len(impacts)
            
            print(f"📊 Based on {len(impacts)} signals:")
            print(f"   Average reduction: {avg_reduction:.1f} points ({avg_reduction_pct:.1f}%)")
            print(f"   Min reduction: {min(i['reduction'] for i in impacts):.1f} points")
            print(f"   Max reduction: {max(i['reduction'] for i in impacts):.1f} points")
            
            # Group by confluence bucket
            buckets = {
                "<50%": [],
                "50-59%": [],
                "60-69%": [],
                "70-79%": [],
                "80%+": []
            }
            
            for impact in impacts:
                conf = impact['confluence']
                if conf < 50:
                    buckets["<50%"].append(impact)
                elif conf < 60:
                    buckets["50-59%"].append(impact)
                elif conf < 70:
                    buckets["60-69%"].append(impact)
                elif conf < 80:
                    buckets["70-79%"].append(impact)
                else:
                    buckets["80%+"].append(impact)
            
            print(f"\n📊 By Confluence Bucket:")
            for bucket, items in buckets.items():
                if items:
                    avg_reduction = sum(i['reduction_pct'] for i in items) / len(items)
                    print(f"   {bucket:8} ({len(items)} signals): {avg_reduction:.1f}% avg reduction")
    
    def analyze_signal_distribution(self):
        """Analyze signal distribution by confluence"""
        
        print("\n" + "="*60)
        print("SIGNAL DISTRIBUTION ANALYSIS")
        print("="*60)
        
        buckets = {
            "<50%": {"GOOD": 0, "WATCHLIST": 0, "IGNORE": 0, "total": 0},
            "50-59%": {"GOOD": 0, "WATCHLIST": 0, "IGNORE": 0, "total": 0},
            "60-69%": {"GOOD": 0, "WATCHLIST": 0, "IGNORE": 0, "total": 0},
            "70-79%": {"GOOD": 0, "WATCHLIST": 0, "IGNORE": 0, "total": 0},
            "80%+": {"GOOD": 0, "WATCHLIST": 0, "IGNORE": 0, "total": 0}
        }
        
        for signal in self.signals:
            conf = signal.get('confluence_pct', 0)
            signal_type = signal.get('signal', 'UNKNOWN')
            
            if conf < 50:
                bucket = "<50%"
            elif conf < 60:
                bucket = "50-59%"
            elif conf < 70:
                bucket = "60-69%"
            elif conf < 80:
                bucket = "70-79%"
            else:
                bucket = "80%+"
            
            buckets[bucket]["total"] += 1
            if signal_type in buckets[bucket]:
                buckets[bucket][signal_type] += 1
        
        print(f"\n{'Confluence':<12} {'Total':<8} {'GOOD':<8} {'WATCHLIST':<12} {'IGNORE':<8}")
        print("-" * 50)
        
        for bucket, stats in buckets.items():
            if stats["total"] > 0:
                print(f"{bucket:<12} {stats['total']:<8} {stats['GOOD']:<8} {stats['WATCHLIST']:<12} {stats['IGNORE']:<8}")
        
        # Calculate conversion rates
        print(f"\n📊 Signal Quality by Confluence:")
        for bucket, stats in buckets.items():
            if stats["total"] > 0:
                good_rate = (stats["GOOD"] / stats["total"]) * 100
                watch_rate = (stats["WATCHLIST"] / stats["total"]) * 100
                print(f"   {bucket:8}: {good_rate:.0f}% GOOD, {watch_rate:.0f}% WATCHLIST")
    
    def find_multiplier_signal_changes(self):
        """Find signals that changed classification due to multiplier"""
        
        print("\n" + "="*60)
        print("MULTIPLIER SIGNAL CHANGES")
        print("="*60)
        
        # Based on your observed thresholds
        # If confidence after multiplier drops below threshold
        
        good_to_watchlist = []
        watchlist_to_ignore = []
        
        for signal in self.signals:
            if signal.get('base_score') and signal.get('confluence_pct'):
                base = signal['base_score']
                final = signal['confidence']
                conf_pct = signal['confluence_pct']
                
                # Determine classification with and without multiplier
                # Using GOOD threshold >= 65, WATCHLIST >= 50
                without_multiplier = "GOOD" if base >= 65 else ("WATCHLIST" if base >= 50 else "IGNORE")
                with_multiplier = signal.get('signal', 'UNKNOWN')
                
                if without_multiplier == "GOOD" and with_multiplier == "WATCHLIST":
                    good_to_watchlist.append({
                        "symbol": signal['symbol'],
                        "base": base,
                        "final": final,
                        "confluence": conf_pct
                    })
                elif without_multiplier == "WATCHLIST" and with_multiplier == "IGNORE":
                    watchlist_to_ignore.append({
                        "symbol": signal['symbol'],
                        "base": base,
                        "final": final,
                        "confluence": conf_pct
                    })
        
        print(f"\n📊 Signals that would be GOOD without multiplier but became WATCHLIST:")
        print(f"   Count: {len(good_to_watchlist)}")
        for item in good_to_watchlist:
            print(f"   - {item['symbol']}: {item['base']:.1f} → {item['final']:.1f} (conf={item['confluence']:.0f}%)")
        
        print(f"\n📊 Signals that would be WATCHLIST without multiplier but became IGNORE:")
        print(f"   Count: {len(watchlist_to_ignore)}")
        for item in watchlist_to_ignore:
            print(f"   - {item['symbol']}: {item['base']:.1f} → {item['final']:.1f} (conf={item['confluence']:.0f}%)")

# ============ MANUAL DATA ENTRY FROM YOUR LOGS ============

def extract_from_your_logs():
    """Extract data from the logs you've shared"""
    
    analyzer = HistoricalSignalAnalyzer()
    
    # From your most recent output (June 1, 2025)
    historical_data = [
        # ETHUSDT entries
        {"symbol": "ETHUSDT", "confidence": 65.0, "signal": "GOOD", "confluence_pct": 64.1, "base_score": 72.0},
        {"symbol": "ETHUSDT", "confidence": 68.4, "signal": "GOOD", "confluence_pct": 64.5, "base_score": 72.0},
        {"symbol": "ETHUSDT", "confidence": 65.0, "signal": "WATCHLIST", "confluence_pct": 62.5, "base_score": 72.0},
        {"symbol": "ETHUSDT", "confidence": 59.2, "signal": "WATCHLIST", "confluence_pct": 60.0, "base_score": 65.8},
        
        # BTCUSDT entries
        {"symbol": "BTCUSDT", "confidence": 66.7, "signal": "GOOD", "confluence_pct": 65.0, "base_score": 69.5},
        {"symbol": "BTCUSDT", "confidence": 66.0, "signal": "GOOD", "confluence_pct": 65.0, "base_score": 69.5},
        {"symbol": "BTCUSDT", "confidence": 62.7, "signal": "GOOD", "confluence_pct": 64.0, "base_score": 69.5},
        {"symbol": "BTCUSDT", "confidence": 59.4, "signal": "WATCHLIST", "confluence_pct": 62.0, "base_score": 65.8},
        {"symbol": "BTCUSDT", "confidence": 58.1, "signal": "WATCHLIST", "confluence_pct": 62.0, "base_score": 64.5},
        
        # NEARUSDT entries
        {"symbol": "NEARUSDT", "confidence": 55.0, "signal": "IGNORE", "confluence_pct": 42.5, "base_score": 63.35},
        {"symbol": "NEARUSDT", "confidence": 54.8, "signal": "IGNORE", "confluence_pct": 42.0, "base_score": 63.35},
        {"symbol": "NEARUSDT", "confidence": 54.4, "signal": "WATCHLIST", "confluence_pct": 50.0, "base_score": 63.35},
    ]
    
    for data in historical_data:
        analyzer.add_historical_entry(
            symbol=data["symbol"],
            confidence=data["confidence"],
            signal=data["signal"],
            confluence_pct=data["confluence_pct"],
            base_score=data.get("base_score")
        )
    
    return analyzer

def main():
    print("📊 HISTORICAL SIGNAL ANALYZER")
    print("="*60)
    
    # Extract from your logs
    analyzer = extract_from_your_logs()
    
    # Save for future use
    analyzer.save_to_json("historical_signals.json")
    
    # Run analyses
    analyzer.analyze_signal_distribution()
    analyzer.analyze_multiplier_impact()
    analyzer.find_multiplier_signal_changes()
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()