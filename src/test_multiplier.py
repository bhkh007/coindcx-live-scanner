# test_multiplier.py - New file to verify multiplier works
from backtest_framework.signal_generator import BacktestSignalGenerator

def test_multiplier_effect():
    """Test that multipliers actually produce different results"""
    
    config = {'signal_thresholds': {'GOOD': 65, 'WATCHLIST': 50}}
    
    # Test signal with known values
    test_analysis = {
        'mtf_score': 70,
        'momentum_score': 75,
        'rr_ratio': 2.0,
        'volume_confirmed': True,
        'volume_24h': 100_000_000,
        'entry_distance_pct': 0.35
    }
    
    base_scores = {
        "structure": 70,
        "mtf": 70,
        "momentum": 75,
        "liquidity": 70,
        "rr": 70,
        "entry": 80,
        "rsi": 50
    }
    
    versions = ["current", "soft", "linear", "none"]
    
    print("="*60)
    print("MULTIPLIER VERIFICATION TEST")
    print("="*60)
    
    for version in versions:
        generator = BacktestSignalGenerator(config, version)
        
        # Calculate confluence
        confluence_pct, _ = generator.calculate_confluence(test_analysis)
        
        # Get multiplier
        multiplier = generator.get_multiplier(confluence_pct)
        
        # Calculate base confidence
        base_confidence = generator.calculate_base_confidence(base_scores)
        
        final_confidence = base_confidence * multiplier
        
        print(f"\n{version.upper()} VERSION:")
        print(f"  Confluence: {confluence_pct:.1f}%")
        print(f"  Multiplier: {multiplier}")
        print(f"  Base Confidence: {base_confidence:.1f}")
        print(f"  Final Confidence: {final_confidence:.1f}")
        print(f"  Difference: {base_confidence - final_confidence:.1f} points")
    
    print("\n" + "="*60)
    print("If all versions show the same multiplier, the bug is confirmed.")
    print("="*60)

if __name__ == "__main__":
    test_multiplier_effect()