#!/usr/bin/env python3
"""
Main entry point for NASDAQ Double Bottom Pattern Scanner.

This script runs the scanner with a subset of popular stocks
for quick demonstration. Use command-line arguments for custom scans.

Usage:
    python main.py                      # Run with default settings (20 stocks)
    python main.py --max-stocks 100     # Scan 100 stocks
    python main.py --symbols AAPL MSFT  # Scan specific symbols
    python main.py --plot               # Generate charts for top patterns
    python main.py --help               # Show all options
"""

import sys

from double_bottom_scanner import (
    DEFAULT_CONFIG,
    get_nasdaq_symbols,
    scan_stocks,
    print_summary,
    save_results_to_csv,
    plot_top_patterns,
)


def run_demo_scan():
    """Run a demonstration scan on a subset of stocks."""
    
    print("\n" + "="*70)
    print("NASDAQ DOUBLE BOTTOM PATTERN SCANNER - Demo Mode")
    print("="*70)
    
    config = DEFAULT_CONFIG.copy()
    
    demo_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
        'NFLX', 'INTC', 'PYPL', 'ADBE', 'CSCO', 'QCOM', 'AVGO', 'TXN',
        'COST', 'SBUX', 'GILD', 'MRNA',
    ]
    
    print(f"\nScanning {len(demo_symbols)} stocks for double bottom patterns...")
    print(f"Configuration:")
    print(f"  - Price tolerance: {config['price_tolerance']*100:.1f}%")
    print(f"  - Min peak height: {config['min_peak_height']*100:.1f}%")
    print(f"  - Separation range: {config['min_separation']}-{config['max_separation']} days")
    print(f"  - Lookback period: {config['lookback_days']} days (~2 years)")
    print("-"*70)
    
    results = scan_stocks(demo_symbols, config, verbose=True)
    
    print_summary(results)
    
    save_results_to_csv(results, 'double_bottom_results.csv')
    
    if not results.empty and len(results) >= 1:
        print("\nGenerating chart for top pattern...")
        plot_top_patterns(results, config, num_plots=1, save_dir='charts')
    
    print("\n" + "="*70)
    print("SCAN COMPLETE!")
    print("="*70)
    
    if not results.empty:
        print(f"\nFound {len(results)} pattern(s).")
        print(f"Results saved to: double_bottom_results.csv")
        print(f"Charts saved to: charts/")
    else:
        print("\nNo double bottom patterns detected in the scanned stocks.")
        print("This could mean the market conditions don't currently show these patterns,")
        print("or try scanning more stocks with: python main.py --max-stocks 100")
    
    print("\nFor more options, run: python main.py --help")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    if len(sys.argv) > 1:
        from double_bottom_scanner import main as scanner_main
        scanner_main()
    else:
        run_demo_scan()
