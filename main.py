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
import json
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

from universe import (
    get_nasdaq_symbols_cached,
    get_demo_symbols,
    passes_liquidity_filter,
    filter_universe_by_liquidity,
    refresh_symbol_cache,
)

from strategy import TradeSignal, generate_signals
from backtester import (Backtest, BacktestResult, plot_equity_curve, print_trade_blotter,
                        run_backtest as run_backtest_v2, BacktestConfig, group_signals_by_symbol)
from metrics import (calculate_metrics, print_metrics, 
                     compute_split_metrics, print_metrics_report, enrich_trades)
from alerts import AlertManager, check_and_alert


def scan_stocks_with_liquidity(
    symbols: list,
    config: dict,
    use_liquidity_filter: bool = True,
    liquidity_config: dict = None,
    verbose: bool = True
):
    """
    Scan stocks for double bottom patterns with optional liquidity filtering.
    
    Args:
        symbols: List of stock symbols
        config: Pattern detection config
        use_liquidity_filter: Whether to apply liquidity filter
        liquidity_config: Liquidity filter parameters
        verbose: Show progress
        
    Returns:
        Tuple of (results_df, price_data_dict)
    """
    from tqdm import tqdm
    import pandas as pd
    
    if liquidity_config is None:
        liquidity_config = {
            'min_price': 5.0,
            'min_avg_dollar_vol': 20_000_000,
            'window': 20,
        }
    
    n_total = len(symbols)
    n_no_data = 0
    n_failed_liquidity = 0
    n_scanned = 0
    n_patterns = 0
    n_signals = 0
    
    price_data = {}
    all_patterns = []
    liquidity_report = []
    
    iterator = tqdm(symbols, desc="Downloading & filtering") if verbose else symbols
    
    for symbol in iterator:
        df = download_stock_data(symbol, years=config['download_years'])
        
        if df is None or len(df) < 100:
            n_no_data += 1
            liquidity_report.append({
                'symbol': symbol,
                'pass_liquidity': False,
                'price_last': None,
                'avg_dollar_vol_20': None,
                'reason': 'no_data'
            })
            continue
        
        if use_liquidity_filter:
            passed, diag = passes_liquidity_filter(
                df,
                min_price=liquidity_config.get('min_price', 5.0),
                min_avg_dollar_vol=liquidity_config.get('min_avg_dollar_vol', 20_000_000),
                window=liquidity_config.get('window', 20)
            )
            
            liquidity_report.append({
                'symbol': symbol,
                'pass_liquidity': passed,
                'price_last': diag.get('price_last'),
                'avg_dollar_vol_20': diag.get('avg_dollar_vol_20'),
                'reason': diag.get('reason')
            })
            
            if not passed:
                n_failed_liquidity += 1
                continue
        
        lookback = min(config['lookback_days'], len(df))
        df_recent = df.iloc[-lookback:]
        price_data[symbol] = df_recent
        n_scanned += 1
        
        patterns = detect_double_bottom(df_recent, config)
        n_patterns += len(patterns)
        
        for pattern in patterns:
            pattern['symbol'] = symbol
            bottom2_date = pattern['bottom2_date']
            date_str = pd.Timestamp(bottom2_date).strftime('%Y%m%d')
            pattern['pattern_id'] = f"{symbol}_{date_str}_{pattern['status']}"
            all_patterns.append(pattern)
    
    os.makedirs('outputs', exist_ok=True)
    liq_df = pd.DataFrame(liquidity_report)
    liq_df.to_csv('outputs/liquidity_filter_report.csv', index=False)
    
    print(f"\nUniverse: {n_total} symbols | Data OK: {n_total - n_no_data} | " 
          f"Liquidity pass: {n_scanned} | Patterns: {n_patterns}")
    
    if not all_patterns:
        return pd.DataFrame(), price_data
    
    results_df = pd.DataFrame(all_patterns)
    
    column_order = [
        'pattern_id', 'symbol', 'status', 'score', 'strength_score',
        'bottom1_date', 'bottom1_price',
        'peak_date', 'peak_price', 'neckline',
        'bottom2_date', 'bottom2_price',
        'breakout_date',
        'current_price', 'target_price',
        'height', 'avg_bottom', 'separation_days',
        'forming_trigger_level', 'forming_trigger_reason',
        'price_diff_pct', 'peak_height_pct',
        'volume_confirmation', 'rsi_divergence',
        'bottom1_idx', 'peak_idx', 'bottom2_idx',
    ]
    
    available_cols = [c for c in column_order if c in results_df.columns]
    extra_cols = [c for c in results_df.columns if c not in column_order]
    results_df = results_df[available_cols + extra_cols]
    
    results_df = results_df.sort_values(
        ['status', 'score', 'bottom2_date'],
        ascending=[False, False, False]
    )
    
    return results_df, price_data


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
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to strategy config JSON (from optimize.py)')
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
    
    parser.add_argument('--universe', type=str, default='demo',
                       choices=['demo', 'nasdaq', 'custom'],
                       help='Universe selection: demo (20 stocks), nasdaq (full cached list), custom (use --symbols)')
    parser.add_argument('--min-price', type=float, default=5.0,
                       help='Minimum stock price for liquidity filter')
    parser.add_argument('--min-dollar-vol', type=float, default=20_000_000,
                       help='Minimum average dollar volume (20M default)')
    parser.add_argument('--liquidity-window', type=int, default=20,
                       help='Lookback window for avg dollar volume calculation')
    parser.add_argument('--disable-liquidity-filter', action='store_true',
                       help='Disable liquidity filtering (for debugging)')
    parser.add_argument('--symbols-cache-max-age-hours', type=int, default=24,
                       help='Max age in hours for cached NASDAQ symbol list')
    parser.add_argument('--refresh-symbol-cache', action='store_true',
                       help='Force refresh of NASDAQ symbol cache before running')
    
    parser.add_argument('--risk-confirmed', type=float, default=0.01,
                       help='Risk fraction per CONFIRMED trade (0.01 = 1%%)')
    parser.add_argument('--risk-forming', type=float, default=0.006,
                       help='Risk fraction per FORMING trade (0.006 = 0.6%%)')
    parser.add_argument('--max-positions-total', type=int, default=10,
                       help='Maximum total open positions')
    parser.add_argument('--max-positions-forming', type=int, default=3,
                       help='Maximum concurrent FORMING positions')
    parser.add_argument('--forming-max-hold-days', type=int, default=60,
                       help='Maximum days to hold FORMING trades')
    parser.add_argument('--forming-no-progress-days', type=int, default=20,
                       help='Days before no-progress check for FORMING')
    parser.add_argument('--forming-no-progress-r', type=float, default=0.5,
                       help='MFE threshold for no-progress rule (0.5 = +0.5R)')
    parser.add_argument('--forming-no-progress-action', type=str, default='EXIT',
                       choices=['EXIT', 'TIGHTEN_STOP'],
                       help='Action when FORMING shows no progress')
    parser.add_argument('--confirmed-move-stop-to-be-at-r', type=float, default=0.5,
                       help='Move CONFIRMED stop to BE at this MFE (0 to disable)')
    
    parser.add_argument('--confirmed-partial-tp-enabled', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable partial take profit for CONFIRMED trades')
    parser.add_argument('--confirmed-partial-tp-at-r', type=float, default=1.0,
                       help='Take partial profit at this R-multiple')
    parser.add_argument('--confirmed-partial-tp-fraction', type=float, default=0.5,
                       help='Fraction of position to sell at partial TP (0.5 = 50%%)')
    
    parser.add_argument('--confirmed-trailing-enabled', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable trailing stop for CONFIRMED trades')
    parser.add_argument('--confirmed-trailing-start-r', type=float, default=1.0,
                       help='Start trailing after reaching this R-multiple')
    parser.add_argument('--confirmed-trailing-atr-mult', type=float, default=2.0,
                       help='ATR multiplier for trailing stop distance')
    parser.add_argument('--atr-length', type=int, default=14,
                       help='ATR calculation period')
    
    parser.add_argument('--forming-tighten-stop-to-r', type=float, default=-0.25,
                       help='R-multiple to tighten FORMING stop to when no-progress-action is TIGHTEN_STOP')
    
    parser.add_argument('--use-correlation-caps', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable correlation caps for entries')
    parser.add_argument('--corr-lookback-days', type=int, default=60,
                       help='Lookback window for correlation computation')
    parser.add_argument('--max-corr-to-existing', type=float, default=0.80,
                       help='Max correlation to existing positions (skip if exceeded)')
    parser.add_argument('--use-cluster-caps', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable cluster-based position caps')
    parser.add_argument('--n-clusters', type=int, default=8,
                       help='Number of clusters for grouping symbols')
    parser.add_argument('--max-positions-per-cluster', type=int, default=2,
                       help='Max positions allowed per cluster')
    
    parser.add_argument('--use-regime-filter', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable market regime filter')
    parser.add_argument('--regime-symbol', type=str, default='QQQ',
                       help='Symbol for regime detection (QQQ or SPY)')
    parser.add_argument('--regime-fast-ma', type=int, default=50,
                       help='Fast MA period for trend detection')
    parser.add_argument('--regime-slow-ma', type=int, default=200,
                       help='Slow MA period for trend detection')
    parser.add_argument('--regime-vol-lookback', type=int, default=20,
                       help='ATR lookback for volatility detection')
    parser.add_argument('--regime-vol-high-threshold', type=float, default=0.03,
                       help='ATR%% threshold for high volatility (0.03 = 3%%)')
    parser.add_argument('--regime-soft-gate', type=str, default='true',
                       choices=['true', 'false'],
                       help='Use soft gating (reduce risk) instead of hard skip')
    parser.add_argument('--regime-downtrend-forming-mult', type=float, default=0.85,
                       help='Risk multiplier for FORMING in downtrend (0.85 = 85%%)')
    parser.add_argument('--regime-highvol-forming-mult', type=float, default=0.70,
                       help='Risk multiplier for FORMING in high vol (0.70 = 70%%)')
    parser.add_argument('--regime-highvol-confirmed-mult', type=float, default=0.85,
                       help='Risk multiplier for CONFIRMED in high vol (0.85 = 85%%)')
    
    parser.add_argument('--use-score-risk-scaling', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable score-based risk scaling (higher score = larger position)')
    parser.add_argument('--score-risk-alpha', type=float, default=1.0,
                       help='Shape parameter for score->risk curve (1.0 = linear)')
    parser.add_argument('--score-risk-min-mult', type=float, default=0.60,
                       help='Min risk multiplier for score=0 (0.60 = 60%%)')
    parser.add_argument('--score-risk-max-mult', type=float, default=1.20,
                       help='Max risk multiplier for score=100 (1.20 = 120%%)')
    parser.add_argument('--score-risk-apply-to', type=str, default='BOTH',
                       choices=['BOTH', 'FORMING', 'CONFIRMED'],
                       help='Apply score scaling to BOTH, FORMING, or CONFIRMED entries')
    parser.add_argument('--score-risk-missing-policy', type=str, default='NEUTRAL',
                       choices=['NEUTRAL', 'MIN'],
                       help='Policy for missing scores: NEUTRAL (1.0x) or MIN (min_mult)')
    parser.add_argument('--max-risk-fraction-per-trade', type=float, default=0.02,
                       help='Absolute max risk fraction after all multipliers (safety cap)')
    
    # Portfolio allocation (Task 20)
    parser.add_argument('--use-portfolio-allocator', type=str, default='true',
                       choices=['true', 'false'],
                       help='Enable portfolio capital allocation (signals compete for daily budget)')
    parser.add_argument('--daily-risk-budget', type=float, default=0.04,
                       help='Max total risk allocated per day as fraction of equity (0.04 = 4%%)')
    parser.add_argument('--weekly-risk-budget', type=float, default=None,
                       help='Optional rolling weekly risk budget cap (None = disabled)')
    parser.add_argument('--daily-risk-budget-forming', type=float, default=0.015,
                       help='Max daily risk for FORMING entries (None = use daily_risk_budget)')
    parser.add_argument('--daily-risk-budget-confirmed', type=float, default=None,
                       help='Max daily risk for CONFIRMED entries (None = use daily_risk_budget)')
    parser.add_argument('--allocation-rank-metric', type=str, default='score_weighted',
                       help='Metric for ranking signals in allocation (score_weighted)')
    parser.add_argument('--max-signals-per-day', type=int, default=None,
                       help='Soft cap on signals per day after ranking (None = unlimited)')
    parser.add_argument('--allocation-scaling-mode', type=str, default='PROPORTIONAL',
                       choices=['PROPORTIONAL', 'HARD_CUTOFF'],
                       help='How to handle over-budget: PROPORTIONAL scales down, HARD_CUTOFF drops')
    parser.add_argument('--min-allocation-scale', type=float, default=0.25,
                       help='Floor for allocation scaling (0.25 = 25%% of original)')
    
    parser.add_argument('--symbols-seed', type=int, default=None,
                       help='Random seed for deterministic symbol subset selection')
    parser.add_argument('--parity-check', action='store_true',
                       help='Run parity check to verify config produces same results')
    
    parser.add_argument('--min-pattern-score', type=float, default=0.0,
                       help='Minimum pattern quality score (0-100) to trade. 0 = no filter')
    parser.add_argument('--top-k-per-day', type=int, default=0,
                       help='Keep only top K signals per day by score. 0 = disabled')
    parser.add_argument('--top-k-per-week', type=int, default=0,
                       help='Keep only top K signals per week by score. 0 = disabled')
    parser.add_argument('--score-policy', type=str, default='RAW',
                       choices=['RAW', 'INVERT'],
                       help='Score policy: RAW (default) or INVERT (100 - score)')
    parser.add_argument('--trend-score-mode', type=str, default='NEUTRAL',
                       choices=['NEUTRAL', 'ABOVE_MA200', 'BELOW_MA200'],
                       help='Trend scoring mode: NEUTRAL removes trend influence (default)')
    
    args = parser.parse_args()
    
    loaded_config = None
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                loaded_config = json.load(f)
            print("\n" + "=" * 60)
            print("CONFIG LOADED")
            print("=" * 60)
            print(f"Config file: {args.config}")
            print(f"Generated: {loaded_config.get('meta', {}).get('generated_at', 'unknown')}")
            print(f"Regime mode: {loaded_config.get('regime', {}).get('mode', 'unknown')}")
            print(f"Robustness pass rate: {loaded_config.get('robustness', {}).get('selected', {}).get('pass_rate', 0):.1%}")
            print("=" * 60 + "\n")
        else:
            print(f"Warning: Config file not found: {args.config}")
    
    if args.refresh_symbol_cache:
        refresh_symbol_cache()
    
    config = DEFAULT_CONFIG.copy()
    
    if loaded_config:
        det = loaded_config.get('detection', {})
        if det.get('price_tolerance') is not None:
            args.price_tolerance = det['price_tolerance']
        if det.get('min_peak_height') is not None:
            args.min_peak_height = det['min_peak_height']
        if det.get('min_separation') is not None:
            args.min_separation = det['min_separation']
        if det.get('max_separation') is not None:
            args.max_separation = det['max_separation']
        if det.get('lookback_days') is not None:
            args.lookback_days = det['lookback_days']
        
        port = loaded_config.get('portfolio', {})
        if port.get('initial_capital') is not None:
            args.initial_capital = port['initial_capital']
        if port.get('risk_fraction_confirmed') is not None:
            args.risk_confirmed = port['risk_fraction_confirmed']
        if port.get('risk_fraction_forming') is not None:
            args.risk_forming = port['risk_fraction_forming']
        if port.get('max_positions_total') is not None:
            args.max_positions_total = port['max_positions_total']
        if port.get('max_positions_forming') is not None:
            args.max_positions_forming = port['max_positions_forming']
        
        exits = loaded_config.get('exits', {})
        confirmed_exits = exits.get('confirmed', {})
        if confirmed_exits.get('move_stop_to_be_at_r') is not None:
            args.confirmed_move_stop_to_be_at_r = confirmed_exits['move_stop_to_be_at_r']
        if confirmed_exits.get('partial_tp_enabled') is not None:
            args.confirmed_partial_tp_enabled = 'true' if confirmed_exits['partial_tp_enabled'] else 'false'
        if confirmed_exits.get('partial_tp_at_r') is not None:
            args.confirmed_partial_tp_at_r = confirmed_exits['partial_tp_at_r']
        if confirmed_exits.get('partial_tp_fraction') is not None:
            args.confirmed_partial_tp_fraction = confirmed_exits['partial_tp_fraction']
        if confirmed_exits.get('trailing_enabled') is not None:
            args.confirmed_trailing_enabled = 'true' if confirmed_exits['trailing_enabled'] else 'false'
        if confirmed_exits.get('trailing_start_r') is not None:
            args.confirmed_trailing_start_r = confirmed_exits['trailing_start_r']
        if confirmed_exits.get('trailing_atr_mult') is not None:
            args.confirmed_trailing_atr_mult = confirmed_exits['trailing_atr_mult']
        
        forming_exits = exits.get('forming', {})
        if forming_exits.get('max_hold_days') is not None:
            args.forming_max_hold_days = forming_exits['max_hold_days']
        if forming_exits.get('no_progress_days') is not None:
            args.forming_no_progress_days = forming_exits['no_progress_days']
        if forming_exits.get('no_progress_r') is not None:
            args.forming_no_progress_r = forming_exits['no_progress_r']
        if forming_exits.get('no_progress_action') is not None:
            args.forming_no_progress_action = forming_exits['no_progress_action']
        if forming_exits.get('tighten_stop_to_r') is not None:
            args.forming_tighten_stop_to_r = forming_exits['tighten_stop_to_r']
        
        if exits.get('atr_length') is not None:
            args.atr_length = exits['atr_length']
        
        corr = loaded_config.get('correlation_caps', {})
        if corr.get('enabled') is not None:
            args.use_correlation_caps = 'true' if corr['enabled'] else 'false'
        if corr.get('lookback_days') is not None:
            args.corr_lookback_days = corr['lookback_days']
        if corr.get('max_corr_to_existing') is not None:
            args.max_corr_to_existing = corr['max_corr_to_existing']
        
        cluster = loaded_config.get('cluster_caps', {})
        if cluster.get('enabled') is not None:
            args.use_cluster_caps = 'true' if cluster['enabled'] else 'false'
        if cluster.get('n_clusters') is not None:
            args.n_clusters = cluster['n_clusters']
        if cluster.get('max_positions_per_cluster') is not None:
            args.max_positions_per_cluster = cluster['max_positions_per_cluster']
        
        regime = loaded_config.get('regime', {})
        if regime.get('enabled') is not None:
            args.use_regime_filter = 'true' if regime['enabled'] else 'false'
        if regime.get('symbol') is not None:
            args.regime_symbol = regime['symbol']
        if regime.get('fast_ma') is not None:
            args.regime_fast_ma = regime['fast_ma']
        if regime.get('slow_ma') is not None:
            args.regime_slow_ma = regime['slow_ma']
        if regime.get('vol_lookback') is not None:
            args.regime_vol_lookback = regime['vol_lookback']
        if regime.get('vol_high_threshold') is not None:
            args.regime_vol_high_threshold = regime['vol_high_threshold']
        if regime.get('mode') == 'SOFT_GATE':
            args.regime_soft_gate = 'true'
        elif regime.get('mode') == 'HARD_SKIP':
            args.regime_soft_gate = 'false'
        if regime.get('downtrend_forming_mult') is not None:
            args.regime_downtrend_forming_mult = regime['downtrend_forming_mult']
        if regime.get('highvol_forming_mult') is not None:
            args.regime_highvol_forming_mult = regime['highvol_forming_mult']
        if regime.get('highvol_confirmed_mult') is not None:
            args.regime_highvol_confirmed_mult = regime['highvol_confirmed_mult']
        
        scoring = loaded_config.get('scoring', {})
        if scoring.get('score_policy') is not None:
            args.score_policy = scoring['score_policy']
        if scoring.get('min_pattern_score') is not None:
            args.min_pattern_score = scoring['min_pattern_score']
        if scoring.get('top_k_per_day') is not None:
            args.top_k_per_day = scoring['top_k_per_day']
        if scoring.get('trend_score_mode') is not None:
            args.trend_score_mode = scoring['trend_score_mode']
        
        score_risk = loaded_config.get('score_risk_scaling', {})
        if score_risk.get('enabled') is not None:
            args.use_score_risk_scaling = 'true' if score_risk['enabled'] else 'false'
        if score_risk.get('alpha') is not None:
            args.score_risk_alpha = score_risk['alpha']
        if score_risk.get('min_mult') is not None:
            args.score_risk_min_mult = score_risk['min_mult']
        if score_risk.get('max_mult') is not None:
            args.score_risk_max_mult = score_risk['max_mult']
        if score_risk.get('apply_to') is not None:
            args.score_risk_apply_to = score_risk['apply_to']
        if score_risk.get('missing_policy') is not None:
            args.score_risk_missing_policy = score_risk['missing_policy']
        if score_risk.get('max_risk_fraction_per_trade') is not None:
            args.max_risk_fraction_per_trade = score_risk['max_risk_fraction_per_trade']
        
        liq = loaded_config.get('liquidity', {})
        if liq.get('min_price') is not None:
            args.min_price = liq['min_price']
        if liq.get('min_avg_dollar_vol') is not None:
            args.min_dollar_vol = liq['min_avg_dollar_vol']
        if liq.get('filter_window') is not None:
            args.liquidity_window = liq['filter_window']
        if liq.get('enabled') is not None:
            args.disable_liquidity_filter = not liq['enabled']
    config['price_tolerance'] = args.price_tolerance
    config['min_peak_height'] = args.min_peak_height
    config['min_separation'] = args.min_separation
    config['max_separation'] = args.max_separation
    config['lookback_days'] = args.lookback_days
    
    import random as _random
    from manifest import generate_run_id, create_manifest, save_manifest
    from price_cache import get_cache_stats
    
    symbols_requested = args.max_stocks or 0
    
    if args.symbols:
        symbols = sorted(args.symbols)
        universe_name = 'custom'
        symbols_requested = len(symbols)
    elif args.universe == 'nasdaq':
        all_symbols = get_nasdaq_symbols_cached(
            max_age_hours=args.symbols_cache_max_age_hours,
            limit=None
        )
        all_symbols = sorted(all_symbols)
        
        if args.max_stocks and args.max_stocks < len(all_symbols):
            if args.symbols_seed is not None:
                _random.seed(args.symbols_seed)
                symbols = sorted(_random.sample(all_symbols, args.max_stocks))
            else:
                symbols = all_symbols[:args.max_stocks]
        else:
            symbols = all_symbols
        universe_name = 'nasdaq'
        symbols_requested = args.max_stocks or len(all_symbols)
    elif args.universe == 'demo':
        symbols = sorted(get_demo_symbols())
        if args.max_stocks:
            if args.symbols_seed is not None:
                _random.seed(args.symbols_seed)
                symbols = sorted(_random.sample(symbols, min(args.max_stocks, len(symbols))))
            else:
                symbols = symbols[:args.max_stocks]
        universe_name = 'demo'
        symbols_requested = args.max_stocks or len(symbols)
    else:
        symbols = sorted(get_nasdaq_symbols(max_symbols=args.max_stocks))
        universe_name = 'legacy'
        symbols_requested = len(symbols)
    
    run_id = generate_run_id()
    
    use_liquidity_filter = not args.disable_liquidity_filter
    liquidity_config = {
        'min_price': args.min_price,
        'min_avg_dollar_vol': args.min_dollar_vol,
        'window': args.liquidity_window,
    }
    
    if args.parity_check and args.config:
        print("\n" + "="*60)
        print("PARITY CHECK MODE")
        print("="*60)
        print("Running backtest twice to verify reproducibility...")
        print("Config:", args.config)
        print("="*60 + "\n")
        
        import subprocess
        import tempfile
        
        base_cmd = [
            'python', 'main.py',
            '--config', args.config,
            '--universe', args.universe,
            '--max-stocks', str(args.max_stocks),
            '--backtest-v2',
        ]
        if args.symbols_seed is not None:
            base_cmd += ['--symbols-seed', str(args.symbols_seed)]
        
        print("Run 1...")
        r1 = subprocess.run(base_cmd, capture_output=True, text=True)
        
        if r1.returncode != 0:
            print(f"Run 1 failed with exit code {r1.returncode}")
            print(r1.stderr[-500:] if r1.stderr else "No stderr")
            return False
        
        if not os.path.exists('outputs/trades.csv') or not os.path.exists('outputs/metrics.json'):
            print("Run 1 did not produce expected output files (trades.csv, metrics.json)")
            print("This may indicate no patterns were found. Parity check requires patterns.")
            return False
        
        with open('outputs/trades.csv', 'r') as f:
            trades1 = f.read()
        with open('outputs/metrics.json', 'r') as f:
            metrics1 = json.load(f)
        
        print("Run 2...")
        r2 = subprocess.run(base_cmd, capture_output=True, text=True)
        
        if r2.returncode != 0:
            print(f"Run 2 failed with exit code {r2.returncode}")
            print(r2.stderr[-500:] if r2.stderr else "No stderr")
            return False
        
        if not os.path.exists('outputs/trades.csv') or not os.path.exists('outputs/metrics.json'):
            print("Run 2 did not produce expected output files")
            return False
        
        with open('outputs/trades.csv', 'r') as f:
            trades2 = f.read()
        with open('outputs/metrics.json', 'r') as f:
            metrics2 = json.load(f)
        
        trades_match = trades1 == trades2
        
        try:
            equity1 = metrics1.get('ALL', {}).get('end_equity', 0)
            equity2 = metrics2.get('ALL', {}).get('end_equity', 0)
            equity_diff = abs(equity1 - equity2) if equity1 and equity2 else 0
            equity_match = equity_diff < 0.01
        except:
            equity_match = False
            equity_diff = float('inf')
        
        try:
            dd1 = metrics1.get('ALL', {}).get('max_dd_pct', 0)
            dd2 = metrics2.get('ALL', {}).get('max_dd_pct', 0)
            dd_diff = abs((dd1 or 0) - (dd2 or 0))
            dd_match = dd_diff < 0.01
        except:
            dd_match = False
            dd_diff = float('inf')
        
        print("\n" + "="*60)
        print("PARITY CHECK RESULTS")
        print("="*60)
        print(f"Trades match: {'PASS' if trades_match else 'FAIL'}")
        print(f"End equity match: {'PASS' if equity_match else 'FAIL'} (diff: ${equity_diff:.2f})")
        print(f"Max DD match: {'PASS' if dd_match else 'FAIL'} (diff: {dd_diff:.2f}%)")
        
        if trades_match and equity_match and dd_match:
            print("\nOVERALL: PASS - Results are reproducible!")
        else:
            print("\nOVERALL: FAIL - Results differ between runs!")
            
            diff_path = 'outputs/parity_diff.txt'
            with open(diff_path, 'w') as f:
                f.write("PARITY CHECK DIFF REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Trades match: {trades_match}\n")
                f.write(f"Equity diff: ${equity_diff:.2f}\n")
                f.write(f"DD diff: {dd_diff:.2f}%\n")
                f.write("\nMetrics 1:\n")
                json.dump(metrics1, f, indent=2, default=str)
                f.write("\n\nMetrics 2:\n")
                json.dump(metrics2, f, indent=2, default=str)
            print(f"Diff saved to: {diff_path}")
        
        print("="*60 + "\n")
        return trades_match and equity_match and dd_match
    
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
        print(f"Universe: {universe_name} ({len(symbols)} symbols)")
        print(f"Liquidity filter: {'Disabled' if args.disable_liquidity_filter else f'min_price=${args.min_price}, min_vol=${args.min_dollar_vol/1e6:.0f}M'}")
        print(f"Initial Capital: ${args.initial_capital:,.2f}")
        print("="*60)
        
        symbols_with_regime = list(symbols)
        regime_symbol = args.regime_symbol.upper()
        if args.use_regime_filter.lower() == 'true' and regime_symbol not in symbols_with_regime:
            symbols_with_regime.append(regime_symbol)
        
        results, price_data = scan_stocks_with_liquidity(
            symbols_with_regime, config, 
            use_liquidity_filter=use_liquidity_filter,
            liquidity_config=liquidity_config,
            verbose=not args.quiet
        )
        
        if results.empty:
            print("\nNo patterns found. Cannot run backtest.")
            return results
        
        if args.use_regime_filter.lower() == 'true':
            results = results[results['symbol'] != regime_symbol]
        
        all_signals = []
        for sym in price_data:
            if args.use_regime_filter.lower() == 'true' and sym == regime_symbol:
                continue
            df = price_data[sym]
            patterns = results[results['symbol'] == sym].to_dict('records')
            signals = generate_signals(
                df, patterns,
                compute_scores=True,
                price_tolerance=args.price_tolerance,
                min_peak_height=args.min_peak_height,
                min_pattern_score=0,
                top_k_per_day=0,
                top_k_per_week=0,
                score_policy=args.score_policy,
                trend_mode=args.trend_score_mode,
            )
            all_signals.extend(signals)
        
        from pattern_scoring import filter_signals_by_score
        if args.min_pattern_score > 0 or args.top_k_per_day > 0 or args.top_k_per_week > 0:
            before_count = len(all_signals)
            all_signals = filter_signals_by_score(
                all_signals,
                min_score=args.min_pattern_score,
                top_k_per_day=args.top_k_per_day,
                top_k_per_week=args.top_k_per_week
            )
            print(f"Score filtering: {before_count} -> {len(all_signals)} signals")
        
        print(f"Generated {len(all_signals)} trade signals from {len(results)} patterns")
        
        signals_by_symbol = group_signals_by_symbol(all_signals)
        
        cfg = BacktestConfig(
            initial_capital=args.initial_capital,
            risk_fraction_per_trade=args.risk_confirmed,
            max_positions=args.max_positions_total,
            one_position_per_symbol=True,
            slippage_bps=5.0,
            commission_per_trade=1.0,
            risk_fraction_confirmed=args.risk_confirmed,
            risk_fraction_forming=args.risk_forming,
            max_positions_total=args.max_positions_total,
            max_positions_forming=args.max_positions_forming,
            max_positions_confirmed=args.max_positions_total,
            forming_max_hold_days=args.forming_max_hold_days,
            forming_no_progress_days=args.forming_no_progress_days,
            forming_no_progress_r=args.forming_no_progress_r,
            forming_no_progress_action=args.forming_no_progress_action,
            forming_tighten_stop_to_r=args.forming_tighten_stop_to_r,
            confirmed_move_stop_to_be_at_r=args.confirmed_move_stop_to_be_at_r,
            confirmed_partial_tp_enabled=args.confirmed_partial_tp_enabled.lower() == 'true',
            confirmed_partial_tp_at_r=args.confirmed_partial_tp_at_r,
            confirmed_partial_tp_fraction=args.confirmed_partial_tp_fraction,
            confirmed_trailing_enabled=args.confirmed_trailing_enabled.lower() == 'true',
            confirmed_trailing_start_r=args.confirmed_trailing_start_r,
            confirmed_trailing_atr_mult=args.confirmed_trailing_atr_mult,
            atr_length=args.atr_length,
            use_correlation_caps=args.use_correlation_caps.lower() == 'true',
            corr_lookback_days=args.corr_lookback_days,
            max_corr_to_existing=args.max_corr_to_existing,
            use_cluster_caps=args.use_cluster_caps.lower() == 'true',
            n_clusters=args.n_clusters,
            max_positions_per_cluster=args.max_positions_per_cluster,
            use_regime_filter=args.use_regime_filter.lower() == 'true',
            regime_symbol=args.regime_symbol.upper(),
            regime_trend_fast_ma=args.regime_fast_ma,
            regime_trend_slow_ma=args.regime_slow_ma,
            regime_vol_lookback=args.regime_vol_lookback,
            regime_vol_high_threshold=args.regime_vol_high_threshold,
            regime_soft_gate=args.regime_soft_gate.lower() == 'true',
            regime_downtrend_forming_risk_mult=args.regime_downtrend_forming_mult,
            regime_highvol_forming_risk_mult=args.regime_highvol_forming_mult,
            regime_highvol_confirmed_risk_mult=args.regime_highvol_confirmed_mult,
            use_score_risk_scaling=args.use_score_risk_scaling.lower() == 'true',
            score_risk_alpha=args.score_risk_alpha,
            score_risk_min_mult=args.score_risk_min_mult,
            score_risk_max_mult=args.score_risk_max_mult,
            score_risk_apply_to=args.score_risk_apply_to.upper(),
            score_risk_missing_policy=args.score_risk_missing_policy.upper(),
            max_risk_fraction_per_trade=args.max_risk_fraction_per_trade,
            use_portfolio_allocator=args.use_portfolio_allocator.lower() == 'true',
            daily_risk_budget=args.daily_risk_budget,
            weekly_risk_budget=args.weekly_risk_budget,
            daily_risk_budget_forming=args.daily_risk_budget_forming,
            daily_risk_budget_confirmed=args.daily_risk_budget_confirmed,
            allocation_rank_metric=args.allocation_rank_metric,
            max_signals_per_day=args.max_signals_per_day,
            allocation_scaling_mode=args.allocation_scaling_mode.upper(),
            min_allocation_scale=args.min_allocation_scale
        )
        
        score_status = "ON" if cfg.use_score_risk_scaling else "OFF"
        print(f"Score risk scaling: {score_status} | apply_to={cfg.score_risk_apply_to} | "
              f"alpha={cfg.score_risk_alpha:.1f} | min={cfg.score_risk_min_mult:.2f} | "
              f"max={cfg.score_risk_max_mult:.2f} | cap={cfg.max_risk_fraction_per_trade:.3f}")
        
        alloc_status = "ON" if cfg.use_portfolio_allocator else "OFF"
        forming_budget = f"{cfg.daily_risk_budget_forming*100:.1f}%" if cfg.daily_risk_budget_forming else "N/A"
        print(f"Portfolio allocator: {alloc_status} | daily_budget={cfg.daily_risk_budget*100:.1f}% | "
              f"forming_budget={forming_budget} | scaling={cfg.allocation_scaling_mode}")
        
        trades_df, equity_df = run_backtest_v2(signals_by_symbol, price_data, cfg)
        
        os.makedirs('outputs', exist_ok=True)
        trades_df.to_csv('outputs/trades.csv', index=False)
        equity_df.to_csv('outputs/equity.csv', index=False)
        
        enriched_trades = enrich_trades(trades_df)
        split_metrics = compute_split_metrics(enriched_trades, equity_df)
        print_metrics_report(split_metrics)
        
        from metrics import compute_score_bucket_metrics, compute_threshold_performance, compute_feature_attribution_report
        if 'pattern_score' in trades_df.columns and trades_df['pattern_score'].notna().any():
            bucket_report = compute_score_bucket_metrics(trades_df)
            if not bucket_report.empty:
                bucket_report.to_csv('outputs/score_bucket_report.csv', index=False)
                print("\n" + "-" * 50)
                print("SCORE BUCKET ANALYSIS")
                print("-" * 50)
                for _, row in bucket_report.iterrows():
                    print(f"Score {row['score_range']:>8}: {row['trade_count']:3d} trades, "
                          f"WR={row['win_rate']:5.1f}%, AvgR={row['avg_r']:+.2f}, "
                          f"PF={row['profit_factor']:.2f}")
            
            thresh_report = compute_threshold_performance(trades_df)
            if not thresh_report.empty:
                thresh_report.to_csv('outputs/score_threshold_report.csv', index=False)
                print("\n" + "-" * 50)
                print("THRESHOLD COMPARISON")
                print("-" * 50)
                for _, row in thresh_report.iterrows():
                    print(f"Min score {int(row['min_score']):>3}: {int(row['trade_count']):3d} trades, "
                          f"WR={row['win_rate']:5.1f}%, AvgR={row['avg_r']:+.2f}, "
                          f"PnL=${row['total_pnl']:,.0f}")
            
            attr_report = compute_feature_attribution_report(trades_df)
            if not attr_report.empty:
                attr_report.to_csv('outputs/feature_attribution_report.csv', index=False)
                print("\n" + "-" * 50)
                print("FEATURE ATTRIBUTION SUMMARY")
                print("-" * 50)
                pos_corr = attr_report[attr_report['corr_to_r'] > 0].nlargest(2, 'corr_to_r')
                neg_corr = attr_report[attr_report['corr_to_r'] < 0].nsmallest(2, 'corr_to_r')
                if not pos_corr.empty:
                    print("Top 2 positively correlated with R:")
                    for _, row in pos_corr.iterrows():
                        print(f"  {row['feature']:>15}: corr={row['corr_to_r']:+.3f}")
                if not neg_corr.empty:
                    print("Top 2 negatively correlated with R:")
                    for _, row in neg_corr.iterrows():
                        print(f"  {row['feature']:>15}: corr={row['corr_to_r']:+.3f}")
        
        with open('outputs/metrics.json', 'w') as f:
            serializable_metrics = {}
            for key, val in split_metrics.items():
                serializable_metrics[key] = {k: (v if not isinstance(v, float) or not (v != v) else None) for k, v in val.items()}
            json.dump(serializable_metrics, f, indent=2, default=str)
        
        print("\nOutputs saved to:")
        print("  - outputs/trades.csv")
        print("  - outputs/equity.csv")
        print("  - outputs/metrics.json")
        if 'pattern_score' in trades_df.columns:
            print("  - outputs/score_bucket_report.csv")
            print("  - outputs/score_threshold_report.csv")
            print("  - outputs/feature_attribution_report.csv")
        
        cache_stats = get_cache_stats('data/price_cache')
        
        backtest_config_dict = {
            'detection': {
                'price_tolerance': args.price_tolerance,
                'min_peak_height': args.min_peak_height,
                'min_separation': args.min_separation,
                'max_separation': args.max_separation,
                'lookback_days': args.lookback_days,
            },
            'portfolio': {
                'initial_capital': args.initial_capital,
                'risk_confirmed': args.risk_confirmed,
                'risk_forming': args.risk_forming,
                'max_positions_total': args.max_positions_total,
                'max_positions_forming': args.max_positions_forming,
            },
            'regime': {
                'enabled': args.use_regime_filter.lower() == 'true',
                'soft_gate': args.regime_soft_gate.lower() == 'true',
            }
        }
        
        manifest = create_manifest(
            run_id=run_id,
            mode='backtest',
            command_line=' '.join(['python', 'main.py'] + sys.argv[1:]),
            symbols_used=symbols,
            symbols_requested=symbols_requested,
            config=backtest_config_dict,
            cache_stats=cache_stats,
            date_range=None,
            universe_type=universe_name,
            liquidity_config=liquidity_config,
            extra_info={
                'symbols_seed': args.symbols_seed,
                'metrics': {
                    'total_return': split_metrics.get('ALL', {}).get('total_return_pct', 0),
                    'max_dd': split_metrics.get('ALL', {}).get('max_dd_pct', 0),
                    'trades': split_metrics.get('ALL', {}).get('trade_count', 0),
                }
            }
        )
        
        manifest_path, symbols_path = save_manifest(manifest, symbols)
        print(f"  - {manifest_path}")
        print(f"  - {symbols_path}")
        print(f"\nRun ID: {run_id}")
        print("="*60 + "\n")
        
        return trades_df, equity_df, split_metrics
    
    print("\n" + "="*60)
    print("NASDAQ DOUBLE BOTTOM PATTERN SCANNER")
    print("="*60)
    print(f"Universe: {universe_name} ({len(symbols)} symbols)")
    print(f"Liquidity filter: {'Disabled' if args.disable_liquidity_filter else f'min_price=${args.min_price}, min_vol=${args.min_dollar_vol/1e6:.0f}M'}")
    print(f"Price tolerance: {config['price_tolerance']*100:.1f}%")
    print(f"Min peak height: {config['min_peak_height']*100:.1f}%")
    print(f"Separation range: {config['min_separation']}-{config['max_separation']} days")
    print(f"Lookback period: {config['lookback_days']} days")
    print("="*60 + "\n")
    
    results, price_data = scan_stocks_with_liquidity(
        symbols, config, 
        use_liquidity_filter=use_liquidity_filter,
        liquidity_config=liquidity_config,
        verbose=not args.quiet
    )
    
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
