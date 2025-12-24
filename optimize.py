#!/usr/bin/env python3
"""
Walk-Forward Optimization for Double Bottom Pattern Scanner.

This module implements rolling train/test window optimization to find
robust parameters that work out-of-sample, not just in-sample.

Usage:
    python optimize.py --symbols AAPL NVDA AMD
    python optimize.py --symbols AAPL NVDA AMD --train-bars 252 --test-bars 63
"""

import argparse
import json
import os
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from double_bottom_scanner import (
    DEFAULT_CONFIG,
    download_stock_data,
    detect_double_bottom,
)
from universe import (
    get_nasdaq_symbols_cached,
    get_demo_symbols,
    passes_liquidity_filter,
)
from strategy import generate_signals
from backtester import run_backtest, BacktestConfig, group_signals_by_symbol
from metrics import compute_trade_metrics, compute_equity_metrics, compute_split_metrics


def build_walkforward_windows(
    dates: pd.DatetimeIndex,
    train_bars: int = 504,
    test_bars: int = 126,
    step_bars: int = 126
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Build rolling train/test windows from a date index.
    
    Args:
        dates: DatetimeIndex (master calendar)
        train_bars: Number of bars for training (~504 = 2 years)
        test_bars: Number of bars for testing (~126 = 6 months)
        step_bars: Number of bars to advance each iteration
        
    Returns:
        List of tuples: (train_start, train_end, test_start, test_end)
    """
    windows = []
    total_bars = len(dates)
    required_bars = train_bars + test_bars
    
    if total_bars < required_bars:
        return windows
    
    start_idx = 0
    while start_idx + required_bars <= total_bars:
        train_start_idx = start_idx
        train_end_idx = start_idx + train_bars - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_bars - 1
        
        if test_end_idx >= total_bars:
            break
        
        windows.append((
            dates[train_start_idx],
            dates[train_end_idx],
            dates[test_start_idx],
            dates[test_end_idx]
        ))
        
        start_idx += step_bars
    
    return windows


def slice_price_data(
    price_data_by_symbol: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> Dict[str, pd.DataFrame]:
    """
    Slice price data to a specific date range.
    
    Args:
        price_data_by_symbol: Dict of symbol -> OHLCV DataFrame
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)
        
    Returns:
        Dict of symbol -> sliced DataFrame
    """
    sliced = {}
    for symbol, df in price_data_by_symbol.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        sliced_df = df.loc[mask].copy()
        if len(sliced_df) > 0:
            sliced[symbol] = sliced_df
    return sliced


def build_param_grid(grid_size_limit: int = 250, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Build parameter grid for optimization including portfolio/exit knobs.
    
    Includes:
    - Detection params (reduced to avoid explosion)
    - Portfolio/risk rules (Task 7)
    - Exit management rules (Task 8)
    
    Returns:
        List of parameter dictionaries (randomly sampled if exceeds limit)
    """
    import random
    
    detection_params = {
        'low_tolerance': [0.04, 0.05],
        'neckline_min_rise': [0.08, 0.10],
        'min_sep': [20, 30],
        'stop_loss_buffer': [0.02],
        'breakout_buffer': [0.002],
    }
    
    portfolio_params = {
        'risk_fraction_forming': [0.004, 0.006, 0.008],
        'max_positions_forming': [2, 3, 4],
    }
    
    forming_exit_params = {
        'forming_no_progress_days': [15, 20, 25],
        'forming_no_progress_r': [0.25, 0.50, 0.75],
        'forming_max_hold_days': [45, 60, 75],
    }
    
    confirmed_exit_params = {
        'confirmed_partial_tp_enabled': [True, False],
        'confirmed_partial_tp_at_r': [0.50, 0.75, 1.00],
        'confirmed_partial_tp_fraction': [0.33, 0.50],
        'confirmed_move_stop_to_be_at_r': [0.25, 0.50, 0.75],
        'confirmed_trailing_enabled': [False, True],
        'confirmed_trailing_start_r': [0.50, 0.75, 1.00],
        'confirmed_trailing_atr_mult': [1.5, 2.0, 2.5],
    }
    
    all_params = {}
    all_params.update(detection_params)
    all_params.update(portfolio_params)
    all_params.update(forming_exit_params)
    all_params.update(confirmed_exit_params)
    
    keys = list(all_params.keys())
    values = [all_params[k] for k in keys]
    
    all_combos = list(product(*values))
    total_combos = len(all_combos)
    
    if total_combos > grid_size_limit:
        random.seed(seed)
        all_combos = random.sample(all_combos, grid_size_limit)
    
    grid = []
    for combo in all_combos:
        params = dict(zip(keys, combo))
        params['max_sep'] = 260
        params['forming_no_progress_action'] = 'EXIT'
        grid.append(params)
    
    return grid


def params_to_config(params: Dict[str, Any], base_config: Dict = None) -> Dict:
    """
    Convert optimization params to scanner config.
    
    Args:
        params: Optimization parameter dict
        base_config: Base configuration to modify
        
    Returns:
        Scanner configuration dict
    """
    config = (base_config or DEFAULT_CONFIG).copy()
    
    if 'low_tolerance' in params:
        config['price_tolerance'] = params['low_tolerance']
    if 'neckline_min_rise' in params:
        config['min_peak_height'] = params['neckline_min_rise']
    if 'min_sep' in params:
        config['min_separation'] = params['min_sep']
    if 'max_sep' in params:
        config['max_separation'] = params['max_sep']
    
    return config


def params_to_backtest_config(params: Dict[str, Any], base_cfg: BacktestConfig) -> BacktestConfig:
    """
    Create a new BacktestConfig with portfolio/exit params from optimization grid.
    
    Args:
        params: Optimization parameter dict
        base_cfg: Base backtest configuration
        
    Returns:
        BacktestConfig with updated portfolio/exit params
    """
    from dataclasses import replace
    
    updates = {}
    
    if 'risk_fraction_forming' in params:
        updates['risk_fraction_forming'] = params['risk_fraction_forming']
    if 'max_positions_forming' in params:
        updates['max_positions_forming'] = params['max_positions_forming']
    if 'forming_no_progress_days' in params:
        updates['forming_no_progress_days'] = params['forming_no_progress_days']
    if 'forming_no_progress_r' in params:
        updates['forming_no_progress_r'] = params['forming_no_progress_r']
    if 'forming_no_progress_action' in params:
        updates['forming_no_progress_action'] = params['forming_no_progress_action']
    if 'forming_max_hold_days' in params:
        updates['forming_max_hold_days'] = params['forming_max_hold_days']
    if 'confirmed_partial_tp_enabled' in params:
        updates['confirmed_partial_tp_enabled'] = params['confirmed_partial_tp_enabled']
    if 'confirmed_partial_tp_at_r' in params:
        updates['confirmed_partial_tp_at_r'] = params['confirmed_partial_tp_at_r']
    if 'confirmed_partial_tp_fraction' in params:
        updates['confirmed_partial_tp_fraction'] = params['confirmed_partial_tp_fraction']
    if 'confirmed_move_stop_to_be_at_r' in params:
        updates['confirmed_move_stop_to_be_at_r'] = params['confirmed_move_stop_to_be_at_r']
    if 'confirmed_trailing_enabled' in params:
        updates['confirmed_trailing_enabled'] = params['confirmed_trailing_enabled']
    if 'confirmed_trailing_start_r' in params:
        updates['confirmed_trailing_start_r'] = params['confirmed_trailing_start_r']
    if 'confirmed_trailing_atr_mult' in params:
        updates['confirmed_trailing_atr_mult'] = params['confirmed_trailing_atr_mult']
    
    if updates:
        return replace(base_cfg, **updates)
    return base_cfg


def run_scan_and_backtest(
    price_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    backtest_cfg: BacktestConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run pattern detection, signal generation, and backtest for given params.
    
    Args:
        price_data: Dict of symbol -> OHLCV DataFrame
        params: Optimization parameters (detection + portfolio/exit)
        backtest_cfg: Base backtest configuration
        
    Returns:
        Tuple of (trades_df, equity_df)
    """
    config = params_to_config(params)
    
    cfg = params_to_backtest_config(params, backtest_cfg)
    
    all_signals = []
    
    for symbol, df in price_data.items():
        if len(df) < 50:
            continue
        
        patterns = detect_double_bottom(df, config)
        
        for pattern in patterns:
            pattern['symbol'] = symbol
        
        if patterns:
            signals = generate_signals(df, patterns)
            all_signals.extend(signals)
    
    if not all_signals:
        return pd.DataFrame(), pd.DataFrame()
    
    signals_by_symbol = group_signals_by_symbol(all_signals)
    
    trades_df, equity_df = run_backtest(signals_by_symbol, price_data, cfg)
    
    return trades_df, equity_df


def compute_exposure_days(trades_df: pd.DataFrame) -> int:
    """Estimate number of days with at least one open position."""
    if trades_df.empty:
        return 0
    
    all_days = set()
    for _, trade in trades_df.iterrows():
        entry = pd.to_datetime(trade['entry_date'])
        exit_d = pd.to_datetime(trade['exit_date'])
        days = pd.date_range(entry, exit_d, freq='B')
        all_days.update(days)
    
    return len(all_days)


def score_oos_composite(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    max_dd_limit: float = 0.12,
    min_trades: int = 20,
    min_exposure_days: int = 20
) -> Tuple[float, Dict[str, Any]]:
    """
    Composite scoring that rewards return but penalizes:
    - Too few trades
    - Too little exposure
    - Large drawdowns
    
    Args:
        trades_df: Trades DataFrame
        equity_df: Equity DataFrame
        max_dd_limit: Maximum allowed drawdown (e.g., 0.12 = 12%)
        min_trades: Minimum trades threshold
        min_exposure_days: Minimum exposure days threshold
        
    Returns:
        Tuple of (score, diagnostics_dict)
    """
    diag = {
        'trade_count': 0,
        'exposure_days': 0,
        'total_return_pct': 0.0,
        'max_drawdown_pct': 0.0,
        'score': float('-inf'),
        'fail_reason': None,
    }
    
    if trades_df.empty or equity_df.empty:
        diag['fail_reason'] = 'no_trades'
        return float('-inf'), diag
    
    tr = compute_trade_metrics(trades_df)
    eq = compute_equity_metrics(equity_df)
    
    trade_count = tr['trade_count']
    exposure_days = compute_exposure_days(trades_df)
    total_return = eq.get('total_return_pct', 0)
    max_dd_pct = abs(eq.get('max_drawdown_pct', 0))
    
    diag['trade_count'] = trade_count
    diag['exposure_days'] = exposure_days
    diag['total_return_pct'] = total_return
    diag['max_drawdown_pct'] = -max_dd_pct
    
    if trade_count < min_trades:
        diag['fail_reason'] = 'too_few_trades'
        return float('-inf'), diag
    
    if max_dd_pct > max_dd_limit * 100:
        diag['fail_reason'] = 'max_dd_exceeded'
        return float('-inf'), diag
    
    trade_penalty = min(0.0, (trade_count - min_trades) / min_trades)
    exposure_penalty = 0.0
    if exposure_days < min_exposure_days:
        exposure_penalty = (min_exposure_days - exposure_days) / min_exposure_days
    
    base = total_return
    score = base * (1 + 0.25 * trade_penalty) - 0.50 * max_dd_pct - 0.25 * exposure_penalty
    
    diag['score'] = score
    return score, diag


def score_oos(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    objective: str = 'total_return',
    max_dd_limit: float = 0.12,
    min_trades: int = 2,
    require_min_trades: bool = True,
    min_exposure_days: int = 20
) -> Tuple[float, bool]:
    """
    Score out-of-sample performance with constraints.
    
    Args:
        trades_df: Trades DataFrame
        equity_df: Equity DataFrame
        objective: Scoring objective ('total_return', 'sharpe', 'expectancy_r', 'composite')
        max_dd_limit: Maximum allowed drawdown (positive, e.g., 0.12 = 12%)
        min_trades: Minimum trades for quality threshold
        require_min_trades: If True, return -inf for < min_trades; if False, return score anyway
        min_exposure_days: Minimum exposure days for composite scoring
        
    Returns:
        Tuple of (score, meets_quality) where meets_quality indicates trade count threshold met
    """
    if trades_df.empty or equity_df.empty:
        return float('-inf'), False
    
    if objective == 'composite':
        score, diag = score_oos_composite(
            trades_df, equity_df, max_dd_limit, min_trades, min_exposure_days
        )
        meets_quality = diag['trade_count'] >= min_trades
        return score, meets_quality
    
    tr = compute_trade_metrics(trades_df)
    eq = compute_equity_metrics(equity_df)
    
    trade_count = tr['trade_count']
    meets_quality = trade_count >= min_trades
    
    max_dd_pct = abs(eq.get('max_drawdown_pct', 0))
    if max_dd_pct > max_dd_limit * 100:
        return float('-inf'), False
    
    if objective == 'total_return':
        score = eq.get('total_return_pct', 0)
    elif objective == 'sharpe':
        score = eq.get('sharpe_ratio', 0)
        if np.isnan(score):
            score = 0.0
    elif objective == 'expectancy_r':
        score = tr.get('expectancy_r', 0)
        if np.isnan(score):
            score = 0.0
    else:
        score = eq.get('total_return_pct', 0)
    
    if require_min_trades and not meets_quality:
        return float('-inf'), False
    
    return score, meets_quality


def run_window_optimization(
    price_data: Dict[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    param_grid: List[Dict[str, Any]],
    backtest_cfg: BacktestConfig,
    objective: str = 'total_return',
    max_dd_train: float = 0.18,
    max_dd_test: float = 0.12,
    min_trades_test: int = 20,
    min_exposure_days_test: int = 20,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run optimization for a single walk-forward window.
    
    Args:
        price_data: Full price data
        train_start/end: Training period
        test_start/end: Testing period
        param_grid: List of parameter combinations
        backtest_cfg: Backtest config
        objective: Scoring objective
        max_dd_train: Max drawdown for training
        max_dd_test: Max drawdown for testing
        min_trades_test: Min trades for test
        min_exposure_days_test: Min exposure days for composite scoring
        verbose: Show progress
        
    Returns:
        Dict with window results
    """
    train_data = slice_price_data(price_data, train_start, train_end)
    test_data = slice_price_data(price_data, test_start, test_end)
    
    if not train_data or not test_data:
        return {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': None,
            'train_score': float('-inf'),
            'test_score': float('-inf'),
            'error': 'No data in period'
        }
    
    best_train_score = float('-inf')
    best_params = None
    best_train_trades_df = pd.DataFrame()
    best_train_equity_df = pd.DataFrame()
    
    iterator = param_grid if not verbose else tqdm(param_grid, desc="  Grid search", leave=False)
    
    for params in iterator:
        trades_df, equity_df = run_scan_and_backtest(train_data, params, backtest_cfg)
        
        score, _ = score_oos(
            trades_df, equity_df,
            objective=objective,
            max_dd_limit=max_dd_train,
            min_trades=2,
            require_min_trades=True,
            min_exposure_days=min_exposure_days_test
        )
        
        if score > best_train_score:
            best_train_score = score
            best_params = params.copy()
            best_train_trades_df = trades_df
            best_train_equity_df = equity_df
    
    if best_params is None:
        return {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': None,
            'train_score': float('-inf'),
            'test_score': float('-inf'),
            'error': 'No valid params found'
        }
    
    train_tr = compute_trade_metrics(best_train_trades_df) if not best_train_trades_df.empty else {}
    train_eq = compute_equity_metrics(best_train_equity_df) if not best_train_equity_df.empty else {}
    
    test_trades, test_equity = run_scan_and_backtest(test_data, best_params, backtest_cfg)
    
    test_score, test_quality = score_oos(
        test_trades, test_equity,
        objective=objective,
        max_dd_limit=max_dd_test,
        min_trades=min_trades_test,
        require_min_trades=False,
        min_exposure_days=min_exposure_days_test
    )
    
    tr = compute_trade_metrics(test_trades) if not test_trades.empty else {}
    eq = compute_equity_metrics(test_equity) if not test_equity.empty else {}
    split = compute_split_metrics(test_trades, test_equity) if not test_trades.empty else {}
    test_exposure_days = compute_exposure_days(test_trades)
    
    return {
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'best_params': best_params,
        'best_params_json': json.dumps(best_params, default=str),
        'train_score': best_train_score,
        'train_trade_count': train_tr.get('trade_count', 0),
        'train_total_return_pct': train_eq.get('total_return_pct', np.nan),
        'test_score': test_score,
        'test_quality': test_quality,
        'test_total_return_pct': eq.get('total_return_pct', np.nan),
        'test_max_dd_pct': eq.get('max_drawdown_pct', np.nan),
        'test_sharpe': eq.get('sharpe_ratio', np.nan),
        'test_trade_count': tr.get('trade_count', 0),
        'test_exposure_days': test_exposure_days,
        'test_expectancy_r': tr.get('expectancy_r', np.nan),
        'test_win_rate': tr.get('win_rate', np.nan),
        'test_profit_factor': tr.get('profit_factor', np.nan),
        'test_forming_expectancy_r': split.get('FORMING', {}).get('expectancy_r', np.nan),
        'test_confirmed_expectancy_r': split.get('CONFIRMED', {}).get('expectancy_r', np.nan),
    }


def compute_stability_summary(window_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Count how often each parameter value was selected.
    
    Args:
        window_results: List of window result dicts
        
    Returns:
        Dict of param_name -> {value: count}
    """
    counts = defaultdict(lambda: defaultdict(int))
    
    for result in window_results:
        params = result.get('best_params')
        if params is None:
            continue
        
        for key, value in params.items():
            counts[key][value] += 1
    
    return {k: dict(v) for k, v in counts.items()}


def format_stability_summary_v2(
    stability: Dict[str, Dict[str, int]], 
    window_results: List[Dict[str, Any]]
) -> str:
    """Format stability summary v2 with aggregate stats."""
    lines = ["WALK-FORWARD STABILITY SUMMARY v2", "=" * 50]
    
    priority_params = [
        'risk_fraction_forming',
        'max_positions_forming',
        'forming_no_progress_days',
        'forming_no_progress_r',
        'forming_max_hold_days',
        'confirmed_partial_tp_at_r',
        'confirmed_partial_tp_enabled',
        'confirmed_move_stop_to_be_at_r',
        'confirmed_trailing_enabled',
        'confirmed_trailing_atr_mult',
        'low_tolerance',
        'neckline_min_rise',
        'min_sep',
    ]
    
    lines.append("\n--- Parameter Win Counts ---")
    for param in priority_params:
        if param in stability:
            value_counts = stability[param]
            sorted_counts = sorted(value_counts.items(), key=lambda x: -x[1])
            counts_str = ", ".join(f"{v}({c})" for v, c in sorted_counts)
            lines.append(f"  {param}: {counts_str}")
    
    for param, value_counts in sorted(stability.items()):
        if param not in priority_params:
            sorted_counts = sorted(value_counts.items(), key=lambda x: -x[1])
            counts_str = ", ".join(f"{v}({c})" for v, c in sorted_counts)
            lines.append(f"  {param}: {counts_str}")
    
    test_returns = [r.get('test_total_return_pct', np.nan) for r in window_results]
    test_dds = [r.get('test_max_dd_pct', np.nan) for r in window_results]
    valid_returns = [r for r in test_returns if not np.isnan(r)]
    valid_dds = [d for d in test_dds if not np.isnan(d)]
    
    passing_windows = sum(
        1 for r in window_results 
        if r.get('test_score', float('-inf')) != float('-inf')
    )
    total_windows = len(window_results)
    
    lines.append("\n--- Aggregate Statistics ---")
    if valid_returns:
        lines.append(f"  Mean test return: {np.mean(valid_returns):.2f}%")
        lines.append(f"  Median test return: {np.median(valid_returns):.2f}%")
    if valid_dds:
        lines.append(f"  Mean test max DD: {np.mean(valid_dds):.2f}%")
        lines.append(f"  Median test max DD: {np.median(valid_dds):.2f}%")
    lines.append(f"  Windows passing constraints: {passing_windows}/{total_windows} ({100*passing_windows/total_windows if total_windows else 0:.0f}%)")
    
    return "\n".join(lines)


def format_stability_summary(stability: Dict[str, Dict[str, int]]) -> str:
    """Format stability summary as readable text (legacy)."""
    lines = ["PARAMETER STABILITY SUMMARY", "=" * 40]
    
    for param, value_counts in sorted(stability.items()):
        sorted_counts = sorted(value_counts.items(), key=lambda x: -x[1])
        counts_str = ", ".join(f"{v} ({c})" for v, c in sorted_counts)
        lines.append(f"{param}: {counts_str}")
    
    return "\n".join(lines)


def find_overall_best_params(window_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    """
    Find the parameter set with best mean test score.
    
    Args:
        window_results: List of window results
        
    Returns:
        Tuple of (best_params, mean_score)
    """
    param_scores = defaultdict(list)
    
    for result in window_results:
        params = result.get('best_params')
        test_score = result.get('test_score', float('-inf'))
        
        if params is None or test_score == float('-inf'):
            continue
        
        params_key = json.dumps(params, sort_keys=True)
        param_scores[params_key].append(test_score)
    
    if not param_scores:
        return {}, float('-inf')
    
    best_key = None
    best_mean = float('-inf')
    
    for key, scores in param_scores.items():
        valid_scores = [s for s in scores if s != float('-inf')]
        if valid_scores:
            mean_score = np.mean(valid_scores)
            if mean_score > best_mean:
                best_mean = mean_score
                best_key = key
    
    if best_key is None:
        return {}, float('-inf')
    
    return json.loads(best_key), best_mean


def run_walkforward_optimization(
    symbols: List[str],
    train_bars: int = 504,
    test_bars: int = 126,
    step_bars: int = 126,
    objective: str = 'composite',
    max_dd_train: float = 0.18,
    max_dd_test: float = 0.12,
    min_trades_test: int = 20,
    min_exposure_days_test: int = 20,
    grid_size_limit: int = 250,
    initial_capital: float = 100000.0,
    verbose: bool = True,
    liquidity_config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Run complete walk-forward optimization v2.
    
    Args:
        symbols: List of symbols to optimize
        train_bars: Training period bars
        test_bars: Testing period bars
        step_bars: Step size between windows
        objective: Scoring objective ('total_return', 'sharpe', 'expectancy_r', 'composite')
        max_dd_train: Max drawdown for training
        max_dd_test: Max drawdown for testing
        min_trades_test: Min trades for test
        min_exposure_days_test: Min exposure days for composite scoring
        grid_size_limit: Max grid combinations
        initial_capital: Starting capital
        verbose: Show progress
        liquidity_config: Liquidity filter settings
        
    Returns:
        Tuple of (results_df, best_params, summary_text)
    """
    print("\n" + "=" * 60)
    print("WALK-FORWARD OPTIMIZATION")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Train/Test/Step: {train_bars}/{test_bars}/{step_bars} bars")
    print(f"Objective: {objective}")
    print(f"Max DD Train/Test: {max_dd_train*100:.0f}%/{max_dd_test*100:.0f}%")
    print("=" * 60)
    
    if liquidity_config is None:
        liquidity_config = {
            'min_price': 5.0,
            'min_avg_dollar_vol': 20_000_000,
            'use_filter': True,
        }
    
    use_liquidity = liquidity_config.get('use_filter', True)
    
    print("\nStep 1: Downloading price data...")
    price_data = {}
    download_years = max(5, (train_bars + test_bars * 3) // 252 + 1)
    
    n_total = len(symbols)
    n_no_data = 0
    n_failed_liq = 0
    
    iterator = tqdm(symbols, desc="Downloading & filtering") if verbose else symbols
    for symbol in iterator:
        df = download_stock_data(symbol, years=download_years)
        if df is None or len(df) <= train_bars:
            n_no_data += 1
            continue
        
        if use_liquidity:
            passed, diag = passes_liquidity_filter(
                df,
                min_price=liquidity_config.get('min_price', 5.0),
                min_avg_dollar_vol=liquidity_config.get('min_avg_dollar_vol', 20_000_000),
                window=liquidity_config.get('window', 20)
            )
            if not passed:
                n_failed_liq += 1
                continue
        
        price_data[symbol] = df
    
    print(f"Universe: {n_total} | Data OK: {n_total - n_no_data} | Liquidity pass: {len(price_data)}")
    
    if not price_data:
        print("ERROR: No valid price data downloaded")
        return pd.DataFrame(), {}, "No data available"
    
    print("\nStep 2: Building walk-forward windows...")
    all_dates = [set(df.index) for df in price_data.values()]
    common_dates = set.intersection(*all_dates) if all_dates else set()
    master_dates = pd.DatetimeIndex(sorted(common_dates))
    
    windows = build_walkforward_windows(master_dates, train_bars, test_bars, step_bars)
    print(f"Created {len(windows)} windows")
    
    if not windows:
        print("ERROR: Not enough data for walk-forward windows")
        return pd.DataFrame(), {}, "Insufficient data for windowing"
    
    print("\nStep 3: Building parameter grid...")
    param_grid = build_param_grid(grid_size_limit)
    print(f"Grid size: {len(param_grid)} combinations")
    
    backtest_cfg = BacktestConfig(
        initial_capital=initial_capital,
        risk_fraction_per_trade=0.01,
        max_positions=10,
        one_position_per_symbol=True,
        slippage_bps=5.0,
        commission_per_trade=1.0
    )
    
    print("\nStep 4: Running walk-forward optimization...")
    window_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        if verbose:
            print(f"\nWindow {i+1}/{len(windows)}: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
        
        result = run_window_optimization(
            price_data=price_data,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            param_grid=param_grid,
            backtest_cfg=backtest_cfg,
            objective=objective,
            max_dd_train=max_dd_train,
            max_dd_test=max_dd_test,
            min_trades_test=min_trades_test,
            min_exposure_days_test=min_exposure_days_test,
            verbose=verbose
        )
        
        result['window_id'] = i + 1
        window_results.append(result)
        
        if verbose:
            print(f"  Train score: {result['train_score']:.2f}, Test score: {result['test_score']:.2f}")
    
    print("\nStep 5: Computing results...")
    
    results_df = pd.DataFrame(window_results)
    
    cols = ['window_id', 'train_start', 'train_end', 'test_start', 'test_end',
            'train_score', 'train_trade_count', 'train_total_return_pct',
            'test_score', 'test_total_return_pct', 'test_max_dd_pct',
            'test_sharpe', 'test_trade_count', 'test_exposure_days', 
            'test_expectancy_r', 'test_win_rate', 'test_profit_factor', 
            'test_forming_expectancy_r', 'test_confirmed_expectancy_r',
            'best_params_json']
    
    available_cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[available_cols + [c for c in results_df.columns if c not in available_cols]]
    
    stability = compute_stability_summary(window_results)
    stability_text = format_stability_summary_v2(stability, window_results)
    
    best_params, best_mean_score = find_overall_best_params(window_results)
    
    valid_test_scores = [r['test_score'] for r in window_results if r['test_score'] != float('-inf')]
    quality_test_scores = [r['test_score'] for r in window_results if r.get('test_quality', False)]
    mean_test_score = np.mean(valid_test_scores) if valid_test_scores else float('nan')
    median_test_score = np.median(valid_test_scores) if valid_test_scores else float('nan')
    
    summary_lines = [
        "WALK-FORWARD OPTIMIZATION SUMMARY",
        "=" * 40,
        f"Windows: {len(windows)}",
        f"Valid test windows (any trades): {len(valid_test_scores)}",
        f"Quality test windows (min trades met): {len(quality_test_scores)}",
        f"Mean test score: {mean_test_score:.2f}",
        f"Median test score: {median_test_score:.2f}",
        "",
        "OVERALL BEST PARAMETERS (by mean OOS score):",
        json.dumps(best_params, indent=2),
        f"Mean OOS score: {best_mean_score:.2f}",
        "",
        stability_text
    ]
    summary_text = "\n".join(summary_lines)
    
    print("\n" + summary_text)
    
    return results_df, best_params, summary_text


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Walk-Forward Optimization for Double Bottom Scanner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'NVDA', 'AMD'],
                       help='Stock symbols to optimize')
    parser.add_argument('--train-bars', type=int, default=504,
                       help='Training period in bars (~504 = 2 years)')
    parser.add_argument('--test-bars', type=int, default=126,
                       help='Testing period in bars (~126 = 6 months)')
    parser.add_argument('--step-bars', type=int, default=126,
                       help='Step size between windows')
    parser.add_argument('--objective', type=str, default='composite',
                       choices=['total_return', 'sharpe', 'expectancy_r', 'composite'],
                       help='Optimization objective (composite includes trade/exposure penalties)')
    parser.add_argument('--max-dd-train', type=float, default=0.18,
                       help='Max drawdown for training (0.18 = 18%%)')
    parser.add_argument('--max-dd-test', type=float, default=0.12,
                       help='Max drawdown for testing (0.12 = 12%%)')
    parser.add_argument('--min-trades-test', type=int, default=20,
                       help='Minimum trades required in test period')
    parser.add_argument('--min-exposure-days-test', type=int, default=20,
                       help='Minimum exposure days for composite scoring')
    parser.add_argument('--grid-size-limit', type=int, default=250,
                       help='Maximum parameter combinations')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Starting capital for backtest')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    parser.add_argument('--universe', type=str, default='custom',
                       choices=['demo', 'nasdaq', 'custom'],
                       help='Universe selection: demo (20 stocks), nasdaq (full cached list), custom (use --symbols)')
    parser.add_argument('--max-stocks', type=int, default=None,
                       help='Maximum number of stocks to scan')
    parser.add_argument('--min-price', type=float, default=5.0,
                       help='Minimum stock price for liquidity filter')
    parser.add_argument('--min-dollar-vol', type=float, default=20_000_000,
                       help='Minimum average dollar volume (20M default)')
    parser.add_argument('--disable-liquidity-filter', action='store_true',
                       help='Disable liquidity filtering')
    parser.add_argument('--liquidity-window', type=int, default=20,
                       help='Lookback window for avg dollar volume calculation')
    
    args = parser.parse_args()
    
    if args.universe == 'nasdaq':
        symbols = get_nasdaq_symbols_cached(limit=args.max_stocks)
    elif args.universe == 'demo':
        symbols = get_demo_symbols()
        if args.max_stocks:
            symbols = symbols[:args.max_stocks]
    else:
        symbols = args.symbols
    
    liquidity_config = {
        'min_price': args.min_price,
        'min_avg_dollar_vol': args.min_dollar_vol,
        'use_filter': not args.disable_liquidity_filter,
        'window': args.liquidity_window,
    }
    
    results_df, best_params, summary_text = run_walkforward_optimization(
        symbols=symbols,
        liquidity_config=liquidity_config,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        objective=args.objective,
        max_dd_train=args.max_dd_train,
        max_dd_test=args.max_dd_test,
        min_trades_test=args.min_trades_test,
        min_exposure_days_test=args.min_exposure_days_test,
        grid_size_limit=args.grid_size_limit,
        initial_capital=args.initial_capital,
        verbose=not args.quiet
    )
    
    os.makedirs('outputs', exist_ok=True)
    
    results_df.to_csv('outputs/walkforward_results.csv', index=False)
    print(f"\nResults saved to: outputs/walkforward_results.csv")
    
    with open('outputs/walkforward_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best params saved to: outputs/walkforward_best_params.json")
    
    with open('outputs/walkforward_summary.txt', 'w') as f:
        f.write(summary_text)
    print(f"Summary saved to: outputs/walkforward_summary.txt")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
