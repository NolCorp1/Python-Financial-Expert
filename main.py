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
    python main.py --backtest           # Run backtesting on detected patterns
    python main.py --help               # Show all options
"""

import argparse
import sys
import os

from double_bottom_scanner import (
    DEFAULT_CONFIG,
    get_nasdaq_symbols,
    scan_stocks,
    print_summary,
    save_results_to_csv,
    plot_top_patterns,
    download_stock_data,
    detect_double_bottom,
)

from strategy import TradeSignal, generate_signals
from backtester import (Backtest, BacktestResult, plot_equity_curve, print_trade_blotter,
                        run_backtest as run_backtest_v2, BacktestConfig, group_signals_by_symbol)
from metrics import (calculate_metrics, print_metrics, 
                     compute_split_metrics, print_metrics_report, enrich_trades)
from alerts import AlertManager, check_and_alert


def run_backtest(symbols: list, config: dict, 
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 stop_loss_buffer: float = 0.02,
                 max_hold_days: int = 60,
                 use_trailing_stop: bool = False,
                 trailing_stop_pct: float = 0.05,
                 verbose: bool = True) -> BacktestResult:
    """
    Run backtesting on detected patterns.
    
    Args:
        symbols: List of stock symbols to scan
        config: Configuration dictionary for pattern detection
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
        stop_loss_buffer: Buffer percentage for stop loss
        max_hold_days: Maximum holding period
        use_trailing_stop: Whether to use trailing stops
        trailing_stop_pct: Trailing stop percentage
        verbose: Whether to show progress
        
    Returns:
        BacktestResult object
    """
    print("\n" + "="*70)
    print("DOUBLE BOTTOM PATTERN BACKTESTER")
    print("="*70)
    print(f"Initial Capital:   ${initial_capital:,.2f}")
    print(f"Position Size:     {position_size*100:.0f}% per trade")
    print(f"Stop Loss Buffer:  {stop_loss_buffer*100:.1f}%")
    print(f"Max Hold Days:     {max_hold_days}")
    print(f"Trailing Stop:     {'Enabled' if use_trailing_stop else 'Disabled'}")
    print("="*70)
    
    print(f"\nStep 1: Downloading price data for {len(symbols)} symbols...")
    
    price_data = {}
    from tqdm import tqdm
    
    iterator = tqdm(symbols, desc="Downloading data") if verbose else symbols
    for symbol in iterator:
        df = download_stock_data(symbol, years=config['download_years'])
        if df is not None and len(df) > 100:
            price_data[symbol] = df
    
    print(f"Successfully downloaded data for {len(price_data)} symbols")
    
    print("\nStep 2: Detecting double bottom patterns...")
    
    all_patterns = []
    all_signals = []
    
    iterator = tqdm(price_data.items(), desc="Detecting patterns") if verbose else price_data.items()
    for symbol, df in iterator:
        lookback = min(config['lookback_days'], len(df))
        df_recent = df.iloc[-lookback:]
        
        patterns = detect_double_bottom(df_recent, config)
        
        for pattern in patterns:
            pattern['symbol'] = symbol
            all_patterns.append(pattern)
            
            signals = generate_signals(df, [pattern], stop_loss_buffer)
            for signal in signals:
                signal.symbol = symbol
            all_signals.extend(signals)
    
    print(f"Found {len(all_patterns)} patterns, generated {len(all_signals)} signals")
    
    print("\nStep 3: Running backtest simulation...")
    
    backtester = Backtest(
        initial_capital=initial_capital,
        position_size=position_size,
        max_hold_days=max_hold_days,
        use_trailing_stop=use_trailing_stop,
        trailing_stop_pct=trailing_stop_pct
    )
    
    result = backtester.run(all_signals, price_data)
    
    print(f"Executed {len(result.trades)} trades")
    
    return result, all_patterns


def run_demo_scan(run_backtest_mode: bool = False, backtest_args: dict = None):
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
    
    if run_backtest_mode:
        args = backtest_args or {}
        result, patterns = run_backtest(
            demo_symbols, config,
            initial_capital=args.get('initial_capital', 10000),
            position_size=args.get('position_size', 0.1),
            stop_loss_buffer=args.get('stop_loss_buffer', 0.02),
            max_hold_days=args.get('max_hold_days', 60),
            use_trailing_stop=args.get('trailing_stop', False),
        )
        
        metrics = calculate_metrics(result)
        print_metrics(metrics)
        
        print_trade_blotter(result)
        
        os.makedirs('charts', exist_ok=True)
        plot_equity_curve(result, save_path='charts/equity_curve.png', show=False)
        
        alert_manager = AlertManager()
        new_alerts = alert_manager.check_for_alerts(patterns)
        for alert in new_alerts:
            alert_manager.send_alert(alert)
        
        print("\n" + "="*70)
        print("BACKTEST COMPLETE!")
        print("="*70)
        print(f"Total Trades: {len(result.trades)}")
        print(f"Final Capital: ${result.final_capital:,.2f}")
        print(f"Total Return: {metrics['total_return']:+.2f}%")
        print(f"Equity curve saved to: charts/equity_curve.png")
        
        return result
    
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


def main():
    """Main entry point with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='NASDAQ Double Bottom Pattern Scanner with Backtesting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Specific stock symbols to scan (space-separated)')
    parser.add_argument('--max-stocks', type=int, default=50,
                       help='Maximum number of NASDAQ stocks to scan')
    parser.add_argument('--price-tolerance', type=float, default=0.04,
                       help='Max price difference between bottoms (decimal)')
    parser.add_argument('--min-peak-height', type=float, default=0.06,
                       help='Minimum peak height above bottoms (decimal)')
    parser.add_argument('--min-separation', type=int, default=20,
                       help='Minimum days between bottoms')
    parser.add_argument('--max-separation', type=int, default=200,
                       help='Maximum days between bottoms')
    parser.add_argument('--lookback-days', type=int, default=504,
                       help='Days of history to analyze')
    parser.add_argument('--output', type=str, default='double_bottom_results.csv',
                       help='Output CSV filename')
    parser.add_argument('--plot', action='store_true',
                       help='Generate charts for top patterns')
    parser.add_argument('--num-plots', type=int, default=5,
                       help='Number of charts to generate')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtesting on detected patterns (legacy)')
    parser.add_argument('--backtest-v2', action='store_true',
                       help='Run backtesting with FORMING/CONFIRMED split metrics')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial capital for backtesting')
    parser.add_argument('--position-size', type=float, default=0.1,
                       help='Position size as fraction of capital (0.1 = 10%%)')
    parser.add_argument('--stop-loss-buffer', type=float, default=0.02,
                       help='Stop loss buffer below support (0.02 = 2%%)')
    parser.add_argument('--max-hold-days', type=int, default=60,
                       help='Maximum holding period in days')
    parser.add_argument('--trailing-stop', action='store_true',
                       help='Enable trailing stop loss')
    parser.add_argument('--trailing-stop-pct', type=float, default=0.05,
                       help='Trailing stop percentage from high (0.05 = 5%%)')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config['price_tolerance'] = args.price_tolerance
    config['min_peak_height'] = args.min_peak_height
    config['min_separation'] = args.min_separation
    config['max_separation'] = args.max_separation
    config['lookback_days'] = args.lookback_days
    
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_nasdaq_symbols(max_symbols=args.max_stocks)
    
    if args.backtest:
        print("\n" + "="*60)
        print("NASDAQ DOUBLE BOTTOM PATTERN SCANNER - BACKTEST MODE")
        print("="*60)
        print(f"Scanning {len(symbols)} symbol(s)...")
        print(f"Configuration:")
        print(f"  - Price tolerance: {config['price_tolerance']*100:.1f}%")
        print(f"  - Min peak height: {config['min_peak_height']*100:.1f}%")
        print(f"  - Separation range: {config['min_separation']}-{config['max_separation']} days")
        print(f"  - Lookback period: {config['lookback_days']} days")
        print("="*60)
        
        result, patterns = run_backtest(
            symbols, config,
            initial_capital=args.initial_capital,
            position_size=args.position_size,
            stop_loss_buffer=args.stop_loss_buffer,
            max_hold_days=args.max_hold_days,
            use_trailing_stop=args.trailing_stop,
            trailing_stop_pct=args.trailing_stop_pct,
            verbose=not args.quiet
        )
        
        metrics = calculate_metrics(result)
        print_metrics(metrics)
        
        print_trade_blotter(result)
        
        os.makedirs('charts', exist_ok=True)
        plot_equity_curve(result, save_path='charts/equity_curve.png', show=False)
        
        alert_manager = AlertManager()
        new_alerts = alert_manager.check_for_alerts(patterns)
        for alert in new_alerts:
            alert_manager.send_alert(alert)
        
        print("\n" + "="*70)
        print("BACKTEST COMPLETE!")
        print("="*70)
        print(f"Total Trades: {len(result.trades)}")
        print(f"Final Capital: ${result.final_capital:,.2f}")
        print(f"Total Return: {metrics['total_return']:+.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Equity curve saved to: charts/equity_curve.png")
        print("="*70 + "\n")
        
        return result
    
    if args.backtest_v2:
        print("\n" + "="*60)
        print("NASDAQ DOUBLE BOTTOM PATTERN SCANNER - BACKTEST V2")
        print("="*60)
        print(f"Scanning {len(symbols)} symbol(s) with FORMING/CONFIRMED split metrics...")
        print(f"Initial Capital: ${args.initial_capital:,.2f}")
        print("="*60)
        
        results, price_data = scan_stocks(symbols, config, verbose=not args.quiet, return_price_data=True)
        
        if results.empty:
            print("\nNo patterns found. Cannot run backtest.")
            return results
        
        all_signals = []
        for sym in price_data:
            df = price_data[sym]
            patterns = results[results['symbol'] == sym].to_dict('records')
            signals = generate_signals(df, patterns)
            all_signals.extend(signals)
        
        print(f"Generated {len(all_signals)} trade signals from {len(results)} patterns")
        
        signals_by_symbol = group_signals_by_symbol(all_signals)
        
        cfg = BacktestConfig(
            initial_capital=args.initial_capital,
            risk_fraction_per_trade=0.01,
            max_positions=10,
            one_position_per_symbol=True,
            slippage_bps=5.0,
            commission_per_trade=1.0
        )
        
        trades_df, equity_df = run_backtest_v2(signals_by_symbol, price_data, cfg)
        
        os.makedirs('outputs', exist_ok=True)
        trades_df.to_csv('outputs/trades.csv', index=False)
        equity_df.to_csv('outputs/equity.csv', index=False)
        
        enriched_trades = enrich_trades(trades_df)
        split_metrics = compute_split_metrics(enriched_trades, equity_df)
        print_metrics_report(split_metrics)
        
        import json
        with open('outputs/metrics.json', 'w') as f:
            serializable_metrics = {}
            for key, val in split_metrics.items():
                serializable_metrics[key] = {k: (v if not isinstance(v, float) or not (v != v) else None) for k, v in val.items()}
            json.dump(serializable_metrics, f, indent=2, default=str)
        
        print("\nOutputs saved to:")
        print("  - outputs/trades.csv")
        print("  - outputs/equity.csv")
        print("  - outputs/metrics.json")
        print("="*60 + "\n")
        
        return trades_df, equity_df, split_metrics
    
    print("\n" + "="*60)
    print("NASDAQ DOUBLE BOTTOM PATTERN SCANNER")
    print("="*60)
    print(f"Price tolerance: {config['price_tolerance']*100:.1f}%")
    print(f"Min peak height: {config['min_peak_height']*100:.1f}%")
    print(f"Separation range: {config['min_separation']}-{config['max_separation']} days")
    print(f"Lookback period: {config['lookback_days']} days")
    print("="*60 + "\n")
    
    print(f"Scanning {len(symbols)} symbol(s)...")
    
    results = scan_stocks(symbols, config, verbose=not args.quiet)
    
    print_summary(results)
    
    save_results_to_csv(results, args.output)
    
    if args.plot and not results.empty:
        print(f"\nGenerating {args.num_plots} chart(s)...")
        plot_top_patterns(results, config, num_plots=args.num_plots)
    
    return results


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        run_demo_scan()
