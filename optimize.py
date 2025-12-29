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
import sys
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
    refresh_symbol_cache,
)
from strategy import generate_signals
from backtester import run_backtest, BacktestConfig, group_signals_by_symbol, BacktestContext, build_backtest_context
from metrics import compute_trade_metrics, compute_equity_metrics, compute_split_metrics
from price_cache import (
    preload_price_data,
    compute_required_date_range,
    clear_price_cache,
    get_cache_stats,
)


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
    Build parameter grid for optimization including portfolio/exit knobs and regime.
    
    Uses random sampling to avoid combinatorial explosion.
    
    Returns:
        List of parameter dictionaries (randomly sampled)
    """
    import random
    random.seed(seed)
    
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
    
    scoring_params = {
        'score_policy': ['RAW', 'INVERT'],
        'min_pattern_score': [0, 40, 50, 60],
        'top_k_per_day': [0, 2, 3, 5],
        'trend_score_mode': ['NEUTRAL', 'BELOW_MA200', 'ABOVE_MA200'],
    }
    
    regime_modes = ['OFF', 'HARD_SKIP', 'SOFT_GATE']
    
    regime_params_when_on = {
        'regime_trend_fast_ma': [20, 50],
        'regime_trend_slow_ma': [150, 200],
        'regime_vol_lookback': [14, 20],
        'regime_vol_high_threshold': [0.025, 0.030, 0.035],
    }
    
    soft_gate_multipliers = {
        'regime_downtrend_forming_mult': [0.80, 0.85, 1.00],
        'regime_highvol_forming_mult': [0.70, 0.85, 1.00],
        'regime_highvol_confirmed_mult': [0.80, 0.90, 1.00],
    }
    
    score_risk_params = {
        'use_score_risk_scaling': [True],
        'score_risk_alpha': [0.8, 1.0, 1.4],
        'score_risk_min_mult': [0.6, 0.7],
        'score_risk_max_mult': [1.1, 1.2, 1.3],
        'score_risk_apply_to': ['BOTH', 'FORMING'],
        'max_risk_fraction_per_trade': [0.015, 0.02],
    }
    
    portfolio_allocator_params = {
        'use_portfolio_allocator': [True],
        'daily_risk_budget': [0.03, 0.04, 0.05],
        'daily_risk_budget_forming': [0.01, 0.015, 0.02],
        'max_signals_per_day': [None, 3, 5],
        'allocation_scaling_mode': ['PROPORTIONAL'],
        'min_allocation_scale': [0.25, 0.40],
    }
    
    all_base_params = {}
    all_base_params.update(detection_params)
    all_base_params.update(portfolio_params)
    all_base_params.update(forming_exit_params)
    all_base_params.update(confirmed_exit_params)
    all_base_params.update(scoring_params)
    all_base_params.update(score_risk_params)
    all_base_params.update(portfolio_allocator_params)
    
    grid = []
    for _ in range(grid_size_limit * 2):
        params = {}
        for key, values_list in all_base_params.items():
            params[key] = random.choice(values_list)
        
        params['max_sep'] = 260
        params['forming_no_progress_action'] = 'EXIT'
        
        regime_mode = random.choice(regime_modes)
        params['regime_mode'] = regime_mode
        
        if regime_mode in ['HARD_SKIP', 'SOFT_GATE']:
            for key, values_list in regime_params_when_on.items():
                params[key] = random.choice(values_list)
        
        if regime_mode == 'SOFT_GATE':
            for key, values_list in soft_gate_multipliers.items():
                params[key] = random.choice(values_list)
        
        param_tuple = tuple(sorted(params.items()))
        if param_tuple not in [tuple(sorted(p.items())) for p in grid]:
            grid.append(params)
        
        if len(grid) >= grid_size_limit:
            break
    
    return grid[:grid_size_limit]


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
    
    regime_mode = params.get('regime_mode', 'SOFT_GATE')
    
    if regime_mode == 'OFF':
        updates['use_regime_filter'] = False
    elif regime_mode == 'HARD_SKIP':
        updates['use_regime_filter'] = True
        updates['regime_soft_gate'] = False
        if 'regime_trend_fast_ma' in params:
            updates['regime_trend_fast_ma'] = params['regime_trend_fast_ma']
        if 'regime_trend_slow_ma' in params:
            updates['regime_trend_slow_ma'] = params['regime_trend_slow_ma']
        if 'regime_vol_lookback' in params:
            updates['regime_vol_lookback'] = params['regime_vol_lookback']
        if 'regime_vol_high_threshold' in params:
            updates['regime_vol_high_threshold'] = params['regime_vol_high_threshold']
    elif regime_mode == 'SOFT_GATE':
        updates['use_regime_filter'] = True
        updates['regime_soft_gate'] = True
        if 'regime_trend_fast_ma' in params:
            updates['regime_trend_fast_ma'] = params['regime_trend_fast_ma']
        if 'regime_trend_slow_ma' in params:
            updates['regime_trend_slow_ma'] = params['regime_trend_slow_ma']
        if 'regime_vol_lookback' in params:
            updates['regime_vol_lookback'] = params['regime_vol_lookback']
        if 'regime_vol_high_threshold' in params:
            updates['regime_vol_high_threshold'] = params['regime_vol_high_threshold']
        if 'regime_downtrend_forming_mult' in params:
            updates['regime_downtrend_forming_risk_mult'] = params['regime_downtrend_forming_mult']
        if 'regime_highvol_forming_mult' in params:
            updates['regime_highvol_forming_risk_mult'] = params['regime_highvol_forming_mult']
        if 'regime_highvol_confirmed_mult' in params:
            updates['regime_highvol_confirmed_risk_mult'] = params['regime_highvol_confirmed_mult']
    
    if 'use_score_risk_scaling' in params:
        updates['use_score_risk_scaling'] = params['use_score_risk_scaling']
    if 'score_risk_alpha' in params:
        updates['score_risk_alpha'] = params['score_risk_alpha']
    if 'score_risk_min_mult' in params:
        updates['score_risk_min_mult'] = params['score_risk_min_mult']
    if 'score_risk_max_mult' in params:
        updates['score_risk_max_mult'] = params['score_risk_max_mult']
    if 'score_risk_apply_to' in params:
        updates['score_risk_apply_to'] = params['score_risk_apply_to']
    if 'score_risk_missing_policy' in params:
        updates['score_risk_missing_policy'] = params['score_risk_missing_policy']
    if 'max_risk_fraction_per_trade' in params:
        updates['max_risk_fraction_per_trade'] = params['max_risk_fraction_per_trade']
    
    # Portfolio allocation params (Task 20)
    if 'use_portfolio_allocator' in params:
        updates['use_portfolio_allocator'] = params['use_portfolio_allocator']
    if 'daily_risk_budget' in params:
        updates['daily_risk_budget'] = params['daily_risk_budget']
    if 'weekly_risk_budget' in params:
        updates['weekly_risk_budget'] = params['weekly_risk_budget']
    if 'daily_risk_budget_forming' in params:
        updates['daily_risk_budget_forming'] = params['daily_risk_budget_forming']
    if 'daily_risk_budget_confirmed' in params:
        updates['daily_risk_budget_confirmed'] = params['daily_risk_budget_confirmed']
    if 'allocation_rank_metric' in params:
        updates['allocation_rank_metric'] = params['allocation_rank_metric']
    if 'max_signals_per_day' in params:
        updates['max_signals_per_day'] = params['max_signals_per_day']
    if 'allocation_scaling_mode' in params:
        updates['allocation_scaling_mode'] = params['allocation_scaling_mode']
    if 'min_allocation_scale' in params:
        updates['min_allocation_scale'] = params['min_allocation_scale']
    
    if updates:
        return replace(base_cfg, **updates)
    return base_cfg


def run_scan_and_backtest(
    price_data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    backtest_cfg: BacktestConfig,
    return_diagnostics: bool = False,
    ctx: Optional[BacktestContext] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run pattern detection, signal generation, and backtest for given params.
    
    Args:
        price_data: Dict of symbol -> OHLCV DataFrame
        params: Optimization parameters (detection + portfolio/exit)
        backtest_cfg: Base backtest configuration
        return_diagnostics: If True, return (trades_df, equity_df, diagnostics)
        ctx: Optional BacktestContext with precomputed artifacts for speedup
        
    Returns:
        Tuple of (trades_df, equity_df) or (trades_df, equity_df, diagnostics)
    """
    config = params_to_config(params)
    
    cfg = params_to_backtest_config(params, backtest_cfg)
    
    regime_symbol = cfg.regime_symbol
    if cfg.use_regime_filter and regime_symbol not in price_data:
        pass
    
    all_signals = []
    
    for symbol, df in price_data.items():
        if len(df) < 50:
            continue
        
        if cfg.use_regime_filter and symbol == cfg.regime_symbol:
            continue
        
        patterns = detect_double_bottom(df, config)
        
        for pattern in patterns:
            pattern['symbol'] = symbol
        
        if patterns:
            signals = generate_signals(
                df, patterns,
                compute_scores=True,
                score_policy=params.get('score_policy', 'RAW'),
                trend_mode=params.get('trend_score_mode', 'NEUTRAL'),
                min_pattern_score=params.get('min_pattern_score', 0),
                top_k_per_day=params.get('top_k_per_day', 0),
                top_k_per_week=0,
            )
            all_signals.extend(signals)
    
    if not all_signals:
        if return_diagnostics:
            return pd.DataFrame(), pd.DataFrame(), {}
        return pd.DataFrame(), pd.DataFrame()
    
    signals_by_symbol = group_signals_by_symbol(all_signals)
    
    result = run_backtest(signals_by_symbol, price_data, cfg, return_diagnostics=return_diagnostics, ctx=ctx)
    
    if return_diagnostics:
        return result
    return result


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
    
    train_ctx = build_backtest_context(train_data, backtest_cfg)
    test_ctx = build_backtest_context(test_data, backtest_cfg)
    
    best_train_score = float('-inf')
    best_params = None
    best_train_trades_df = pd.DataFrame()
    best_train_equity_df = pd.DataFrame()
    
    iterator = param_grid if not verbose else tqdm(param_grid, desc="  Grid search", leave=False)
    
    for params in iterator:
        trades_df, equity_df = run_scan_and_backtest(train_data, params, backtest_cfg, ctx=train_ctx)
        
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
    
    test_result = run_scan_and_backtest(test_data, best_params, backtest_cfg, return_diagnostics=True, ctx=test_ctx)
    if len(test_result) == 3:
        test_trades, test_equity, test_diag = test_result
    else:
        test_trades, test_equity = test_result
        test_diag = {}
    
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
    
    regime_mode = best_params.get('regime_mode', 'SOFT_GATE')
    
    result = {
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
        'test_regime_mode': regime_mode,
        'test_downtrend_days': test_diag.get('regime_downtrend_days', 0),
        'test_highvol_days': test_diag.get('regime_highvol_days', 0),
        'test_regime_hard_skipped': test_diag.get('skipped_regime_hard', 0),
        'test_forming_reduced_downtrend': test_diag.get('reduced_regime_forming_downtrend', 0),
        'test_forming_reduced_highvol': test_diag.get('reduced_regime_forming_highvol', 0),
        'test_confirmed_reduced_highvol': test_diag.get('reduced_regime_confirmed_highvol', 0),
        'test_avg_risk_forming': test_diag.get('avg_effective_risk_forming', 0.0),
        'test_avg_risk_confirmed': test_diag.get('avg_effective_risk_confirmed', 0.0),
    }
    
    return result


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


def find_overall_best_params(
    window_results: List[Dict[str, Any]],
    min_pass_rate: float = 0.60,
    min_median_test_score: float = 0.0
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Find the parameter set with best mean test score, applying robustness gating.
    
    Robustness criteria:
    1. Minimum pass rate (windows with valid scores / total windows)
    2. Minimum median test score > threshold
    3. Prefer lower variance (stability) as tie-breaker
    
    Args:
        window_results: List of window results
        min_pass_rate: Minimum fraction of windows that must pass (0.6 = 60%)
        min_median_test_score: Minimum median test score required
        
    Returns:
        Tuple of (best_params, mean_score, robustness_stats)
    """
    param_stats = defaultdict(lambda: {
        'scores': [],
        'returns': [],
        'dds': [],
        'trades': [],
        'exposure_days': []
    })
    
    total_windows = len(window_results)
    
    for result in window_results:
        params = result.get('best_params')
        test_score = result.get('test_score', float('-inf'))
        
        if params is None:
            continue
        
        params_key = json.dumps(params, sort_keys=True)
        
        if test_score != float('-inf'):
            param_stats[params_key]['scores'].append(test_score)
            param_stats[params_key]['returns'].append(result.get('test_total_return_pct', 0))
            param_stats[params_key]['dds'].append(result.get('test_max_dd_pct', 0))
            param_stats[params_key]['trades'].append(result.get('test_trade_count', 0))
            param_stats[params_key]['exposure_days'].append(result.get('test_exposure_days', 0))
    
    if not param_stats:
        return {}, float('-inf'), {'candidates': 0, 'passed_gating': 0}
    
    candidates = []
    for params_key, stats in param_stats.items():
        scores = stats['scores']
        if not scores:
            continue
        
        pass_rate = len(scores) / total_windows if total_windows > 0 else 0
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        score_variance = np.var(scores) if len(scores) > 1 else 0
        mean_dd = np.mean(stats['dds']) if stats['dds'] else 0
        dd_variance = np.var(stats['dds']) if len(stats['dds']) > 1 else 0
        mean_trades = np.mean(stats['trades']) if stats['trades'] else 0
        mean_exposure = np.mean(stats['exposure_days']) if stats['exposure_days'] else 0
        
        candidates.append({
            'params_key': params_key,
            'pass_rate': pass_rate,
            'mean_score': mean_score,
            'median_score': median_score,
            'score_variance': score_variance,
            'mean_dd': mean_dd,
            'dd_variance': dd_variance,
            'mean_trades': mean_trades,
            'mean_exposure': mean_exposure,
            'window_count': len(scores)
        })
    
    robustness_stats = {
        'total_candidates': len(candidates),
        'total_windows': total_windows,
        'min_pass_rate_required': min_pass_rate,
        'min_median_score_required': min_median_test_score,
    }
    
    passing_candidates = [
        c for c in candidates 
        if c['pass_rate'] >= min_pass_rate and c['median_score'] >= min_median_test_score
    ]
    
    robustness_stats['candidates_passing_gating'] = len(passing_candidates)
    
    if not passing_candidates:
        if candidates:
            candidates.sort(key=lambda x: (-x['median_score'], x['dd_variance']))
            best = candidates[0]
            robustness_stats['relaxed_gating'] = True
            robustness_stats['warning'] = 'No candidates met robustness criteria, using best available'
        else:
            return {}, float('-inf'), robustness_stats
    else:
        passing_candidates.sort(key=lambda x: (-x['median_score'], x['dd_variance']))
        best = passing_candidates[0]
        robustness_stats['relaxed_gating'] = False
    
    robustness_stats['selected'] = {
        'pass_rate': best['pass_rate'],
        'mean_score': best['mean_score'],
        'median_score': best['median_score'],
        'score_variance': best['score_variance'],
        'mean_dd': best['mean_dd'],
        'dd_variance': best['dd_variance'],
        'mean_trades': best['mean_trades'],
        'mean_exposure': best['mean_exposure'],
        'windows_passing': best['window_count']
    }
    
    return json.loads(best['params_key']), best['mean_score'], robustness_stats


def compute_score_policy_wf_summary(
    window_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Aggregate walk-forward results by scoring knobs.
    
    Groups by: score_policy, trend_score_mode, min_pattern_score, top_k_per_day
    
    Args:
        window_results: List of window result dicts from walk-forward
        
    Returns:
        DataFrame with aggregated statistics by scoring configuration
    """
    scoring_knobs = ['score_policy', 'trend_score_mode', 'min_pattern_score', 'top_k_per_day']
    
    grouped_data = defaultdict(lambda: {
        'scores': [],
        'returns': [],
        'dds': [],
        'trades': [],
        'exposure_days': []
    })
    
    for result in window_results:
        params = result.get('best_params')
        if params is None:
            continue
        
        key = tuple(params.get(k, 'N/A') for k in scoring_knobs)
        test_score = result.get('test_score', float('-inf'))
        
        grouped_data[key]['scores'].append(test_score)
        grouped_data[key]['returns'].append(result.get('test_total_return_pct', np.nan))
        grouped_data[key]['dds'].append(result.get('test_max_dd_pct', np.nan))
        grouped_data[key]['trades'].append(result.get('test_trade_count', 0))
        grouped_data[key]['exposure_days'].append(result.get('test_exposure_days', 0))
    
    total_windows = len(window_results)
    rows = []
    
    for key, data in grouped_data.items():
        valid_scores = [s for s in data['scores'] if s != float('-inf') and not np.isnan(s)]
        valid_returns = [r for r in data['returns'] if not np.isnan(r)]
        valid_dds = [d for d in data['dds'] if not np.isnan(d)]
        
        windows_passing = len(valid_scores)
        pass_rate = windows_passing / total_windows if total_windows > 0 else 0
        
        row = {
            'score_policy': key[0],
            'trend_score_mode': key[1],
            'min_pattern_score': key[2],
            'top_k_per_day': key[3],
            'windows_total': total_windows,
            'windows_passing': windows_passing,
            'pass_rate': round(pass_rate, 4),
            'mean_test_score': round(np.mean(valid_scores), 4) if valid_scores else np.nan,
            'median_test_score': round(np.median(valid_scores), 4) if valid_scores else np.nan,
            'mean_test_return_pct': round(np.mean(valid_returns), 2) if valid_returns else np.nan,
            'median_test_return_pct': round(np.median(valid_returns), 2) if valid_returns else np.nan,
            'mean_test_max_dd_pct': round(np.mean(valid_dds), 2) if valid_dds else np.nan,
            'median_test_max_dd_pct': round(np.median(valid_dds), 2) if valid_dds else np.nan,
            'mean_test_trade_count': round(np.mean(data['trades']), 1) if data['trades'] else 0,
            'mean_test_exposure_days': round(np.mean(data['exposure_days']), 1) if data['exposure_days'] else 0,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['pass_rate', 'median_test_score'], ascending=[False, False])
    
    return df


def select_best_scoring_defaults(
    summary_df: pd.DataFrame,
    min_pass_rate: float = 0.60,
    min_median_test_score: float = 0.0
) -> Dict[str, Any]:
    """
    Select best scoring defaults using stability-first rules.
    
    Rules:
    1. require pass_rate >= min_pass_rate
    2. require median_test_score >= min_median_test_score
    3. maximize median_test_score
    4. tie-breakers:
       a) minimize median_test_max_dd_pct
       b) maximize median_test_return_pct
       c) prefer simpler rules (min_score=0 over 60 if similar)
    
    Args:
        summary_df: DataFrame from compute_score_policy_wf_summary
        min_pass_rate: Minimum pass rate required
        min_median_test_score: Minimum median test score required
        
    Returns:
        Dict with chosen defaults and reasoning
    """
    result = {
        'chosen_defaults': {
            'score_policy': 'RAW',
            'trend_score_mode': 'NEUTRAL',
            'min_pattern_score': 0,
            'top_k_per_day': 0,
        },
        'reasoning': {
            'pass_rate': 0,
            'median_score': 0,
            'median_dd': 0,
            'median_return': 0,
            'trade_count': 0,
            'selection_method': 'default_fallback'
        },
        'warning': None
    }
    
    if summary_df.empty:
        result['warning'] = 'No walk-forward results available'
        return result
    
    candidates = summary_df[
        (summary_df['pass_rate'] >= min_pass_rate) &
        (summary_df['median_test_score'] >= min_median_test_score)
    ].copy()
    
    if candidates.empty:
        candidates = summary_df.copy()
        result['warning'] = 'No candidates met stability criteria, using best available'
        result['reasoning']['selection_method'] = 'relaxed_gating'
    else:
        result['reasoning']['selection_method'] = 'stability_first'
    
    candidates = candidates.sort_values(
        ['median_test_score', 'median_test_max_dd_pct', 'median_test_return_pct', 'min_pattern_score'],
        ascending=[False, True, False, True]
    )
    
    best = candidates.iloc[0]
    
    result['chosen_defaults'] = {
        'score_policy': best['score_policy'],
        'trend_score_mode': best['trend_score_mode'],
        'min_pattern_score': int(best['min_pattern_score']) if pd.notna(best['min_pattern_score']) else 0,
        'top_k_per_day': int(best['top_k_per_day']) if pd.notna(best['top_k_per_day']) else 0,
    }
    
    result['reasoning'] = {
        'pass_rate': round(best['pass_rate'], 4),
        'median_score': round(best['median_test_score'], 4) if pd.notna(best['median_test_score']) else 0,
        'median_dd': round(best['median_test_max_dd_pct'], 2) if pd.notna(best['median_test_max_dd_pct']) else 0,
        'median_return': round(best['median_test_return_pct'], 2) if pd.notna(best['median_test_return_pct']) else 0,
        'trade_count': round(best['mean_test_trade_count'], 1) if pd.notna(best['mean_test_trade_count']) else 0,
        'selection_method': result['reasoning']['selection_method']
    }
    
    return result


def build_strategy_config(
    best_params: Dict[str, Any],
    args,
    robustness_stats: Dict[str, Any],
    window_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a comprehensive strategy configuration for production use.
    
    Args:
        best_params: Best parameters from optimization
        args: CLI arguments namespace
        robustness_stats: Robustness gating statistics
        window_results: List of window results for metadata
        
    Returns:
        Complete strategy configuration dict
    """
    config = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'generator': 'optimize.py walk-forward v2',
            'version': '1.0.0',
        },
        
        'optimization': {
            'universe_used': args.universe,
            'max_stocks_used': args.max_stocks,
            'train_bars': args.train_bars,
            'test_bars': args.test_bars,
            'step_bars': args.step_bars,
            'objective': args.objective,
            'grid_size_limit': args.grid_size_limit,
            'min_trades_test': args.min_trades_test,
            'min_exposure_days_test': args.min_exposure_days_test,
            'max_dd_train': args.max_dd_train,
            'max_dd_test': args.max_dd_test,
            'min_pass_rate': getattr(args, 'min_pass_rate', 0.60),
            'min_median_test_score': getattr(args, 'min_median_test_score', 0.0),
            'windows_completed': len(window_results),
        },
        
        'robustness': robustness_stats,
        
        'detection': {
            'price_tolerance': best_params.get('low_tolerance', 0.04),
            'min_peak_height': best_params.get('neckline_min_rise', 0.06),
            'min_separation': best_params.get('min_sep', 20),
            'max_separation': best_params.get('max_sep', 200),
            'stop_loss_buffer': best_params.get('stop_loss_buffer', 0.02),
            'breakout_buffer': best_params.get('breakout_buffer', 0.002),
            'lookback_days': 504,
        },
        
        'portfolio': {
            'initial_capital': args.initial_capital,
            'risk_fraction_confirmed': 0.01,
            'risk_fraction_forming': best_params.get('risk_fraction_forming', 0.006),
            'max_positions_total': 10,
            'max_positions_forming': best_params.get('max_positions_forming', 3),
            'slippage_bps': 5.0,
            'commission_per_trade': 1.0,
        },
        
        'exits': {
            'confirmed': {
                'move_stop_to_be_at_r': best_params.get('confirmed_move_stop_to_be_at_r', 0.5),
                'partial_tp_enabled': best_params.get('confirmed_partial_tp_enabled', True),
                'partial_tp_at_r': best_params.get('confirmed_partial_tp_at_r', 1.0),
                'partial_tp_fraction': best_params.get('confirmed_partial_tp_fraction', 0.5),
                'trailing_enabled': best_params.get('confirmed_trailing_enabled', True),
                'trailing_start_r': best_params.get('confirmed_trailing_start_r', 1.0),
                'trailing_atr_mult': best_params.get('confirmed_trailing_atr_mult', 2.0),
            },
            'forming': {
                'max_hold_days': best_params.get('forming_max_hold_days', 60),
                'no_progress_days': best_params.get('forming_no_progress_days', 20),
                'no_progress_r': best_params.get('forming_no_progress_r', 0.5),
                'no_progress_action': best_params.get('forming_no_progress_action', 'EXIT'),
                'tighten_stop_to_r': -0.25,
            },
            'atr_length': 14,
        },
        
        'correlation_caps': {
            'enabled': True,
            'lookback_days': 60,
            'max_corr_to_existing': 0.80,
        },
        
        'cluster_caps': {
            'enabled': True,
            'n_clusters': 8,
            'max_positions_per_cluster': 2,
        },
        
        'regime': {
            'enabled': best_params.get('regime_mode', 'OFF') != 'OFF',
            'mode': best_params.get('regime_mode', 'SOFT_GATE'),
            'symbol': 'QQQ',
            'fast_ma': best_params.get('regime_trend_fast_ma', 50),
            'slow_ma': best_params.get('regime_trend_slow_ma', 200),
            'vol_lookback': best_params.get('regime_vol_lookback', 20),
            'vol_high_threshold': best_params.get('regime_vol_high_threshold', 0.03),
            'downtrend_forming_mult': best_params.get('regime_downtrend_forming_mult', 0.85),
            'highvol_forming_mult': best_params.get('regime_highvol_forming_mult', 0.70),
            'highvol_confirmed_mult': best_params.get('regime_highvol_confirmed_mult', 0.85),
        },
        
        'scoring': {
            'score_policy': best_params.get('score_policy', 'RAW'),
            'min_pattern_score': best_params.get('min_pattern_score', 0),
            'top_k_per_day': best_params.get('top_k_per_day', 0),
            'top_k_per_week': best_params.get('top_k_per_week', 0),
            'trend_score_mode': best_params.get('trend_score_mode', 'NEUTRAL'),
        },
        
        'score_risk_scaling': {
            'enabled': best_params.get('use_score_risk_scaling', True),
            'alpha': best_params.get('score_risk_alpha', 1.0),
            'min_mult': best_params.get('score_risk_min_mult', 0.60),
            'max_mult': best_params.get('score_risk_max_mult', 1.20),
            'apply_to': best_params.get('score_risk_apply_to', 'BOTH'),
            'missing_policy': best_params.get('score_risk_missing_policy', 'NEUTRAL'),
            'max_risk_fraction_per_trade': best_params.get('max_risk_fraction_per_trade', 0.02),
        },
        
        'portfolio_allocator': {
            'enabled': best_params.get('use_portfolio_allocator', True),
            'daily_risk_budget': best_params.get('daily_risk_budget', 0.04),
            'weekly_risk_budget': best_params.get('weekly_risk_budget', None),
            'daily_risk_budget_forming': best_params.get('daily_risk_budget_forming', 0.015),
            'daily_risk_budget_confirmed': best_params.get('daily_risk_budget_confirmed', None),
            'allocation_rank_metric': best_params.get('allocation_rank_metric', 'score_weighted'),
            'max_signals_per_day': best_params.get('max_signals_per_day', None),
            'allocation_scaling_mode': best_params.get('allocation_scaling_mode', 'PROPORTIONAL'),
            'min_allocation_scale': best_params.get('min_allocation_scale', 0.25),
        },
        
        'liquidity': {
            'min_price': args.min_price,
            'min_avg_dollar_vol': args.min_dollar_vol,
            'filter_window': args.liquidity_window,
            'enabled': not args.disable_liquidity_filter,
        },
        
        'raw_best_params': best_params,
    }
    
    return config


def format_robustness_summary(robustness_stats: Dict[str, Any]) -> str:
    """Format robustness statistics as readable text."""
    lines = ["ROBUSTNESS GATING SUMMARY", "=" * 40]
    
    lines.append(f"Total candidates evaluated: {robustness_stats.get('total_candidates', 0)}")
    lines.append(f"Total windows: {robustness_stats.get('total_windows', 0)}")
    lines.append(f"Min pass rate required: {robustness_stats.get('min_pass_rate_required', 0.6):.0%}")
    lines.append(f"Min median score required: {robustness_stats.get('min_median_score_required', 0.0):.2f}")
    lines.append(f"Candidates passing gating: {robustness_stats.get('candidates_passing_gating', 0)}")
    
    if robustness_stats.get('relaxed_gating'):
        lines.append(f"\nWARNING: {robustness_stats.get('warning', 'Gating relaxed')}")
    
    selected = robustness_stats.get('selected', {})
    if selected:
        lines.append("\nSELECTED CONFIG STATS:")
        lines.append(f"  Pass rate: {selected.get('pass_rate', 0):.1%}")
        lines.append(f"  Mean score: {selected.get('mean_score', 0):.2f}")
        lines.append(f"  Median score: {selected.get('median_score', 0):.2f}")
        lines.append(f"  Score variance: {selected.get('score_variance', 0):.4f}")
        lines.append(f"  Mean max DD: {selected.get('mean_dd', 0):.2f}%")
        lines.append(f"  DD variance: {selected.get('dd_variance', 0):.4f}")
        lines.append(f"  Mean trades: {selected.get('mean_trades', 0):.1f}")
        lines.append(f"  Mean exposure days: {selected.get('mean_exposure', 0):.1f}")
        lines.append(f"  Windows passing: {selected.get('windows_passing', 0)}")
    
    return "\n".join(lines)


def compute_regime_mode_stats(window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics for regime mode across windows.
    
    Args:
        window_results: List of window result dicts
        
    Returns:
        Dict with regime stats
    """
    mode_counts = defaultdict(int)
    mode_scores = defaultdict(list)
    total_downtrend_days = 0
    total_highvol_days = 0
    total_hard_skipped = 0
    total_forming_reduced_downtrend = 0
    total_forming_reduced_highvol = 0
    total_confirmed_reduced_highvol = 0
    avg_risk_forming_list = []
    avg_risk_confirmed_list = []
    
    for r in window_results:
        mode = r.get('test_regime_mode', 'OFF')
        mode_counts[mode] += 1
        
        test_score = r.get('test_score', float('-inf'))
        if test_score != float('-inf'):
            mode_scores[mode].append(test_score)
        
        total_downtrend_days += r.get('test_downtrend_days', 0)
        total_highvol_days += r.get('test_highvol_days', 0)
        total_hard_skipped += r.get('test_regime_hard_skipped', 0)
        total_forming_reduced_downtrend += r.get('test_forming_reduced_downtrend', 0)
        total_forming_reduced_highvol += r.get('test_forming_reduced_highvol', 0)
        total_confirmed_reduced_highvol += r.get('test_confirmed_reduced_highvol', 0)
        
        if r.get('test_avg_risk_forming', 0) > 0:
            avg_risk_forming_list.append(r['test_avg_risk_forming'])
        if r.get('test_avg_risk_confirmed', 0) > 0:
            avg_risk_confirmed_list.append(r['test_avg_risk_confirmed'])
    
    mode_mean_scores = {}
    for mode, scores in mode_scores.items():
        mode_mean_scores[mode] = np.mean(scores) if scores else float('nan')
    
    return {
        'mode_counts': dict(mode_counts),
        'mode_mean_scores': mode_mean_scores,
        'total_downtrend_days': total_downtrend_days,
        'total_highvol_days': total_highvol_days,
        'total_hard_skipped': total_hard_skipped,
        'total_forming_reduced_downtrend': total_forming_reduced_downtrend,
        'total_forming_reduced_highvol': total_forming_reduced_highvol,
        'total_confirmed_reduced_highvol': total_confirmed_reduced_highvol,
        'avg_risk_forming': np.mean(avg_risk_forming_list) if avg_risk_forming_list else 0.0,
        'avg_risk_confirmed': np.mean(avg_risk_confirmed_list) if avg_risk_confirmed_list else 0.0,
    }


def format_regime_stats(stats: Dict[str, Any]) -> str:
    """Format regime stats for summary output."""
    lines = [
        "REGIME FILTER STATISTICS",
        "-" * 30,
    ]
    
    mode_counts = stats.get('mode_counts', {})
    mode_scores = stats.get('mode_mean_scores', {})
    
    for mode in ['OFF', 'HARD_SKIP', 'SOFT_GATE']:
        count = mode_counts.get(mode, 0)
        mean_score = mode_scores.get(mode, float('nan'))
        if count > 0:
            lines.append(f"  {mode}: {count} windows, mean score={mean_score:.2f}")
    
    total_reduced = (stats.get('total_forming_reduced_downtrend', 0) +
                     stats.get('total_forming_reduced_highvol', 0) +
                     stats.get('total_confirmed_reduced_highvol', 0))
    
    lines.append(f"  Hard skipped (HARD_SKIP mode): {stats.get('total_hard_skipped', 0)}")
    lines.append(f"  Soft gated (reduced risk): {total_reduced}")
    lines.append(f"  Downtrend days (total): {stats.get('total_downtrend_days', 0)}")
    lines.append(f"  High-vol days (total): {stats.get('total_highvol_days', 0)}")
    
    avg_risk_f = stats.get('avg_risk_forming', 0)
    avg_risk_c = stats.get('avg_risk_confirmed', 0)
    if avg_risk_f > 0 or avg_risk_c > 0:
        lines.append(f"  Avg effective risk: FORMING={avg_risk_f:.3f}, CONFIRMED={avg_risk_c:.3f}")
    
    return "\n".join(lines)


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
    liquidity_config: Optional[Dict[str, Any]] = None,
    use_price_cache: bool = True,
    price_cache_dir: str = "data/price_cache",
    download_batch_size: int = 50,
    min_pass_rate: float = 0.60,
    min_median_test_score: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any], str, List[Dict[str, Any]], Dict[str, Any]]:
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
    
    print("\nStep 1: Downloading price data (batch mode)...")
    
    all_symbols = list(symbols) + ['QQQ', 'SPY']
    all_symbols = list(dict.fromkeys(all_symbols))
    
    start_date, end_date = compute_required_date_range(
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        n_windows=10,
        buffer_bars=50
    )
    
    if use_price_cache:
        cache_stats = get_cache_stats(price_cache_dir)
        if cache_stats.get('count', 0) > 0:
            print(f"Price cache: {cache_stats['count']} files, {cache_stats['size_mb']:.1f} MB")
    
    raw_price_data = preload_price_data(
        symbols=all_symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_price_cache,
        cache_dir=price_cache_dir,
        batch_size=download_batch_size,
        verbose=verbose
    )
    
    n_total = len(symbols)
    n_no_data = 0
    n_failed_liq = 0
    price_data = {}
    
    for symbol in symbols:
        if symbol not in raw_price_data:
            n_no_data += 1
            continue
        
        df = raw_price_data[symbol]
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
    
    for regime_sym in ['QQQ', 'SPY']:
        if regime_sym in raw_price_data and regime_sym not in price_data:
            df = raw_price_data[regime_sym]
            if df is not None and len(df) > train_bars:
                price_data[regime_sym] = df
                print(f"Added regime symbol {regime_sym} for regime filter")
    
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
            'test_regime_mode', 'test_downtrend_days', 'test_highvol_days',
            'test_regime_hard_skipped', 'test_forming_reduced_downtrend',
            'test_forming_reduced_highvol', 'test_confirmed_reduced_highvol',
            'test_avg_risk_forming', 'test_avg_risk_confirmed',
            'best_params_json']
    
    available_cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[available_cols + [c for c in results_df.columns if c not in available_cols]]
    
    stability = compute_stability_summary(window_results)
    stability_text = format_stability_summary_v2(stability, window_results)
    
    best_params, best_mean_score, robustness_stats = find_overall_best_params(
        window_results,
        min_pass_rate=min_pass_rate,
        min_median_test_score=min_median_test_score
    )
    
    robustness_text = format_robustness_summary(robustness_stats)
    
    valid_test_scores = [r['test_score'] for r in window_results if r['test_score'] != float('-inf')]
    quality_test_scores = [r['test_score'] for r in window_results if r.get('test_quality', False)]
    mean_test_score = np.mean(valid_test_scores) if valid_test_scores else float('nan')
    median_test_score = np.median(valid_test_scores) if valid_test_scores else float('nan')
    
    regime_stats = compute_regime_mode_stats(window_results)
    regime_stats_text = format_regime_stats(regime_stats)
    
    summary_lines = [
        "WALK-FORWARD OPTIMIZATION SUMMARY",
        "=" * 40,
        f"Windows: {len(windows)}",
        f"Valid test windows (any trades): {len(valid_test_scores)}",
        f"Quality test windows (min trades met): {len(quality_test_scores)}",
        f"Mean test score: {mean_test_score:.2f}",
        f"Median test score: {median_test_score:.2f}",
        "",
        robustness_text,
        "",
        "OVERALL BEST PARAMETERS (by median OOS score + stability):",
        json.dumps(best_params, indent=2),
        f"Mean OOS score: {best_mean_score:.2f}",
        "",
        regime_stats_text,
        "",
        stability_text
    ]
    summary_text = "\n".join(summary_lines)
    
    print("\n" + summary_text)
    
    return results_df, best_params, summary_text, window_results, robustness_stats


def run_validation(args):
    """
    Run enhanced validation on best config file.
    
    Outputs:
    - validation_score_bucket_report.csv
    - validation_feature_attribution_report.csv
    - validation_scoring_recommendation.txt
    - Comparison of scoring-enabled vs disabled
    """
    import random as _random
    from metrics import (
        compute_trade_metrics, compute_equity_metrics, compute_split_metrics,
        compute_score_bucket_metrics, compute_threshold_performance,
        compute_feature_attribution_report, check_score_monotonicity
    )
    
    config_path = args.validate_config
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        print("Run optimization first: python optimize.py --universe nasdaq --max-stocks 200")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\n" + "=" * 60)
    print("ENHANCED VALIDATION MODE")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Generated: {config.get('meta', {}).get('generated_at', 'unknown')}")
    print(f"Stocks: {args.validate_stocks}")
    print("=" * 60)
    
    all_symbols = get_nasdaq_symbols_cached(limit=None)
    all_symbols = sorted(all_symbols)
    
    if args.symbols_seed is not None:
        _random.seed(args.symbols_seed)
        symbols = sorted(_random.sample(all_symbols, min(args.validate_stocks, len(all_symbols))))
    else:
        symbols = all_symbols[:args.validate_stocks]
    
    print(f"\nDownloading data for {len(symbols)} symbols...")
    
    start_date, end_date = compute_required_date_range(504 + 126)
    raw_price_data = preload_price_data(
        symbols + ['QQQ', 'SPY'],
        start_date,
        end_date,
        cache_dir=args.price_cache_dir,
        batch_size=args.download_batch_size
    )
    
    price_data = {}
    liquidity_config = {
        'min_price': args.min_price,
        'min_avg_dollar_vol': args.min_dollar_vol,
        'use_filter': not args.disable_liquidity_filter,
        'window': args.liquidity_window,
    }
    
    for symbol in symbols:
        if symbol not in raw_price_data or raw_price_data[symbol] is None:
            continue
        df = raw_price_data[symbol]
        if len(df) < 200:
            continue
        if not args.disable_liquidity_filter:
            passed, _ = passes_liquidity_filter(df, **{k: v for k, v in liquidity_config.items() if k != 'use_filter'})
            if not passed:
                continue
        price_data[symbol] = df
    
    for regime_sym in ['QQQ', 'SPY']:
        if regime_sym in raw_price_data and raw_price_data[regime_sym] is not None:
            price_data[regime_sym] = raw_price_data[regime_sym]
    
    print(f"Valid symbols after filtering: {len([s for s in price_data if s not in ['QQQ', 'SPY']])}")
    
    scoring_config = config.get('scoring', {})
    detection_config = config.get('detection', {})
    portfolio_config = config.get('portfolio', {})
    exits_config = config.get('exits', {})
    regime_config = config.get('regime', {})
    
    def run_backtest_with_scoring(use_scoring: bool) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Run backtest with or without scoring filters."""
        if use_scoring:
            min_score = scoring_config.get('min_pattern_score', 0)
            top_k = scoring_config.get('top_k_per_day', 0)
            score_policy = scoring_config.get('score_policy', 'RAW')
            trend_mode = scoring_config.get('trend_score_mode', 'NEUTRAL')
        else:
            min_score = 0
            top_k = 0
            score_policy = 'RAW'
            trend_mode = 'NEUTRAL'
        
        detection_cfg = {
            'low_tolerance': detection_config.get('price_tolerance', 0.04),
            'neckline_min_rise': detection_config.get('min_peak_height', 0.06),
            'min_sep': detection_config.get('min_separation', 20),
            'max_sep': detection_config.get('max_separation', 200),
            'stop_loss_buffer': detection_config.get('stop_loss_buffer', 0.02),
            'breakout_buffer': detection_config.get('breakout_buffer', 0.002),
        }
        
        all_signals = []
        for symbol, df in price_data.items():
            if symbol in ['QQQ', 'SPY']:
                continue
            
            patterns = detect_double_bottom(df, detection_cfg)
            if not patterns:
                continue
            
            signals = generate_signals(
                df=df,
                patterns=patterns,
                stop_loss_buffer=detection_cfg.get('stop_loss_buffer', 0.02),
                price_tolerance=detection_cfg.get('low_tolerance', 0.04),
                min_peak_height=detection_cfg.get('neckline_min_rise', 0.06),
                score_policy=score_policy,
                trend_mode=trend_mode,
                min_pattern_score=min_score,
                top_k_per_day=top_k,
                top_k_per_week=0
            )
            
            for sig in signals:
                sig.symbol = symbol
            all_signals.extend(signals)
        
        if not all_signals:
            return pd.DataFrame(), {}, {}
        
        backtest_cfg = BacktestConfig(
            initial_capital=portfolio_config.get('initial_capital', 100000),
            risk_fraction_per_trade=portfolio_config.get('risk_fraction_confirmed', 0.01),
            max_positions=portfolio_config.get('max_positions_total', 10),
            one_position_per_symbol=True,
            slippage_bps=portfolio_config.get('slippage_bps', 5.0),
            commission_per_trade=portfolio_config.get('commission_per_trade', 1.0)
        )
        
        signals_by_symbol = group_signals_by_symbol(all_signals)
        
        trades_df, equity_df = run_backtest(signals_by_symbol, price_data, backtest_cfg)
        
        if trades_df.empty:
            return pd.DataFrame(), {}, {}
        
        trades_df['pnl_dollars'] = trades_df.get('pnl', 0)
        trades_df['pnl_r_multiple'] = trades_df.get('pnl_r', 0)
        
        class MockResult:
            def __init__(self, trades_df, equity_df, cfg):
                self.trades = []
                self.equity_curve = equity_df
                self.initial_capital = cfg.initial_capital
                self.final_capital = equity_df['equity'].iloc[-1] if not equity_df.empty else cfg.initial_capital
                
                for _, row in trades_df.iterrows():
                    class MockTrade:
                        pass
                    t = MockTrade()
                    t.pnl = row.get('pnl', 0)
                    t.pnl_r = row.get('pnl_r', 0)
                    t.entry_date = row.get('entry_date')
                    t.exit_date = row.get('exit_date')
                    t.symbol = row.get('symbol', '')
                    t.entry_price = row.get('entry_price', 0)
                    t.exit_price = row.get('exit_price', 0)
                    self.trades.append(t)
        
        mock_result = MockResult(trades_df, equity_df, backtest_cfg)
        trade_metrics = compute_trade_metrics(mock_result)
        equity_metrics = compute_equity_metrics(mock_result)
        
        return trades_df, trade_metrics, equity_metrics
    
    print("\nRunning backtest with scoring enabled...")
    trades_scoring, tr_scoring, eq_scoring = run_backtest_with_scoring(use_scoring=True)
    
    print("Running backtest without scoring (baseline)...")
    trades_baseline, tr_baseline, eq_baseline = run_backtest_with_scoring(use_scoring=False)
    
    os.makedirs('outputs', exist_ok=True)
    
    bucket_report = pd.DataFrame()
    monotonicity = {}
    
    if not trades_scoring.empty:
        bucket_report = compute_score_bucket_metrics(trades_scoring)
        if not bucket_report.empty:
            bucket_report.to_csv('outputs/validation_score_bucket_report.csv', index=False)
            print("Saved: outputs/validation_score_bucket_report.csv")
            
            monotonicity = check_score_monotonicity(bucket_report)
            if monotonicity.get('warning'):
                print(f"\n*** WARNING: {monotonicity['warning']} ***\n")
        
        feature_report = compute_feature_attribution_report(trades_scoring)
        if not feature_report.empty:
            feature_report.to_csv('outputs/validation_feature_attribution_report.csv', index=False)
            print("Saved: outputs/validation_feature_attribution_report.csv")
    
    recommendation_lines = [
        "=" * 60,
        "VALIDATION SCORING RECOMMENDATION",
        "=" * 60,
        "",
        "CHOSEN SCORING DEFAULTS:",
        f"  score_policy: {scoring_config.get('score_policy', 'RAW')}",
        f"  trend_score_mode: {scoring_config.get('trend_score_mode', 'NEUTRAL')}",
        f"  min_pattern_score: {scoring_config.get('min_pattern_score', 0)}",
        f"  top_k_per_day: {scoring_config.get('top_k_per_day', 0)}",
        "",
        "SCORING ENABLED RESULTS:",
        f"  Trades: {tr_scoring.get('trade_count', 0)}",
        f"  Win Rate: {tr_scoring.get('win_rate', 0):.1f}%",
        f"  Expectancy R: {tr_scoring.get('expectancy_r', 0):.3f}",
        f"  Total Return: {eq_scoring.get('total_return_pct', 0):.2f}%",
        f"  Max Drawdown: {eq_scoring.get('max_drawdown_pct', 0):.2f}%",
        f"  Profit Factor: {tr_scoring.get('profit_factor', 0):.2f}",
        "",
        "BASELINE (NO SCORING FILTER) RESULTS:",
        f"  Trades: {tr_baseline.get('trade_count', 0)}",
        f"  Win Rate: {tr_baseline.get('win_rate', 0):.1f}%",
        f"  Expectancy R: {tr_baseline.get('expectancy_r', 0):.3f}",
        f"  Total Return: {eq_baseline.get('total_return_pct', 0):.2f}%",
        f"  Max Drawdown: {eq_baseline.get('max_drawdown_pct', 0):.2f}%",
        f"  Profit Factor: {tr_baseline.get('profit_factor', 0):.2f}",
        "",
        "DELTA (SCORING vs BASELINE):",
    ]
    
    delta_return = eq_scoring.get('total_return_pct', 0) - eq_baseline.get('total_return_pct', 0)
    delta_dd = eq_scoring.get('max_drawdown_pct', 0) - eq_baseline.get('max_drawdown_pct', 0)
    delta_expectancy = tr_scoring.get('expectancy_r', 0) - tr_baseline.get('expectancy_r', 0)
    delta_trades = tr_scoring.get('trade_count', 0) - tr_baseline.get('trade_count', 0)
    
    recommendation_lines.extend([
        f"  Return: {'+' if delta_return >= 0 else ''}{delta_return:.2f}%",
        f"  Drawdown: {'+' if delta_dd >= 0 else ''}{delta_dd:.2f}% {'(better)' if delta_dd < 0 else '(worse)'}",
        f"  Expectancy R: {'+' if delta_expectancy >= 0 else ''}{delta_expectancy:.3f}",
        f"  Trade Count: {'+' if delta_trades >= 0 else ''}{delta_trades}",
        "",
    ])
    
    if not trades_scoring.empty:
        monotonicity = check_score_monotonicity(bucket_report) if not bucket_report.empty else {}
        recommendation_lines.extend([
            "MONOTONICITY CHECK:",
            f"  Inverted: {monotonicity.get('inverted_top_decile', 'N/A')}",
            f"  Spearman Correlation: {monotonicity.get('monotonicity_score', 'N/A')}",
            f"  Top Bucket Avg R: {monotonicity.get('top_bucket_avg_r', 'N/A')}",
            f"  Bottom Bucket Avg R: {monotonicity.get('bottom_bucket_avg_r', 'N/A')}",
            "",
        ])
        
        if monotonicity.get('inverted_top_decile'):
            recommendation_lines.append("*** WARNING: Score inversion detected! Consider using --score-policy INVERT ***")
        else:
            recommendation_lines.append("Score monotonicity OK - higher scores correlate with better performance.")
    
    recommendation_lines.extend([
        "",
        "=" * 60,
        "RECOMMENDATION:",
        "=" * 60,
    ])
    
    if delta_expectancy > 0 and not monotonicity.get('inverted_top_decile', False):
        recommendation_lines.append("Use scoring filters - they improve expectancy without inversion.")
        recommendation_lines.append(f"Run: python main.py --config {config_path} --backtest-v2")
    elif monotonicity.get('inverted_top_decile', False):
        recommendation_lines.append("Consider disabling scoring or using INVERT policy due to detected inversion.")
        recommendation_lines.append("Run: python main.py --score-policy INVERT --backtest-v2")
    else:
        recommendation_lines.append("Scoring filter may not provide significant benefit. Test both approaches.")
        recommendation_lines.append(f"Run: python main.py --config {config_path} --backtest-v2")
    
    recommendation_text = "\n".join(recommendation_lines)
    
    with open('outputs/validation_scoring_recommendation.txt', 'w') as f:
        f.write(recommendation_text)
    print("\nSaved: outputs/validation_scoring_recommendation.txt")
    
    print("\n" + recommendation_text)
    
    validation_summary = {
        'config_path': config_path,
        'symbols_validated': len([s for s in price_data if s not in ['QQQ', 'SPY']]),
        'scoring_defaults': scoring_config,
        'results_with_scoring': {
            'trades': tr_scoring.get('trade_count', 0),
            'win_rate': tr_scoring.get('win_rate', 0),
            'expectancy_r': tr_scoring.get('expectancy_r', 0),
            'total_return_pct': eq_scoring.get('total_return_pct', 0),
            'max_drawdown_pct': eq_scoring.get('max_drawdown_pct', 0),
        },
        'results_baseline': {
            'trades': tr_baseline.get('trade_count', 0),
            'win_rate': tr_baseline.get('win_rate', 0),
            'expectancy_r': tr_baseline.get('expectancy_r', 0),
            'total_return_pct': eq_baseline.get('total_return_pct', 0),
            'max_drawdown_pct': eq_baseline.get('max_drawdown_pct', 0),
        },
        'monotonicity_check': monotonicity if not trades_scoring.empty else {},
        'delta': {
            'return_pct': delta_return,
            'drawdown_pct': delta_dd,
            'expectancy_r': delta_expectancy,
        }
    }
    
    with open('outputs/validation_summary.json', 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    print("Saved: outputs/validation_summary.json")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE!")
    print("=" * 60)


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
    
    parser.add_argument('--refresh-symbol-cache', action='store_true',
                       help='Force refresh of NASDAQ symbol cache before running')
    parser.add_argument('--price-cache-dir', type=str, default='data/price_cache',
                       help='Directory for parquet price cache')
    parser.add_argument('--download-batch-size', type=int, default=50,
                       help='Number of symbols per yfinance batch download (50-75 recommended)')
    parser.add_argument('--disable-price-cache', action='store_true',
                       help='Disable price data caching (download fresh each run)')
    parser.add_argument('--clear-price-cache', action='store_true',
                       help='Clear price cache before running')
    
    parser.add_argument('--min-pass-rate', type=float, default=0.60,
                       help='Minimum window pass rate for robustness gating (0.6 = 60%%)')
    parser.add_argument('--min-median-test-score', type=float, default=0.0,
                       help='Minimum median test score required')
    
    parser.add_argument('--validate-best', action='store_true',
                       help='Run validation on best config (load outputs/strategy_best_config.json)')
    parser.add_argument('--validate-config', type=str, default='outputs/strategy_best_config.json',
                       help='Path to config file for validation')
    parser.add_argument('--validate-stocks', type=int, default=300,
                       help='Number of stocks for validation (default 300)')
    
    parser.add_argument('--symbols-seed', type=int, default=None,
                       help='Random seed for deterministic symbol subset selection')
    parser.add_argument('--grid-seed', type=int, default=42,
                       help='Random seed for parameter grid sampling')
    
    args = parser.parse_args()
    
    if args.refresh_symbol_cache:
        refresh_symbol_cache()
    
    if args.clear_price_cache:
        clear_price_cache(args.price_cache_dir)
    
    if args.validate_best:
        run_validation(args)
        return
    
    import random as _random
    from manifest import generate_run_id, create_manifest, save_manifest, get_version_info
    from price_cache import get_cache_stats
    
    symbols_requested = args.max_stocks or 0
    
    if args.universe == 'nasdaq':
        all_symbols = get_nasdaq_symbols_cached(limit=None)
        all_symbols = sorted(all_symbols)
        
        if args.max_stocks and args.max_stocks < len(all_symbols):
            if args.symbols_seed is not None:
                _random.seed(args.symbols_seed)
                symbols = sorted(_random.sample(all_symbols, args.max_stocks))
            else:
                symbols = all_symbols[:args.max_stocks]
        else:
            symbols = all_symbols
        symbols_requested = args.max_stocks or len(all_symbols)
    elif args.universe == 'demo':
        symbols = sorted(get_demo_symbols())
        if args.max_stocks:
            if args.symbols_seed is not None:
                _random.seed(args.symbols_seed)
                symbols = sorted(_random.sample(symbols, min(args.max_stocks, len(symbols))))
            else:
                symbols = symbols[:args.max_stocks]
        symbols_requested = args.max_stocks or len(symbols)
    else:
        symbols = sorted(args.symbols)
        symbols_requested = len(symbols)
    
    run_id = generate_run_id()
    
    liquidity_config = {
        'min_price': args.min_price,
        'min_avg_dollar_vol': args.min_dollar_vol,
        'use_filter': not args.disable_liquidity_filter,
        'window': args.liquidity_window,
    }
    
    results_df, best_params, summary_text, window_results, robustness_stats = run_walkforward_optimization(
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
        verbose=not args.quiet,
        use_price_cache=not args.disable_price_cache,
        price_cache_dir=args.price_cache_dir,
        download_batch_size=args.download_batch_size,
        min_pass_rate=args.min_pass_rate,
        min_median_test_score=args.min_median_test_score,
    )
    
    os.makedirs('outputs', exist_ok=True)
    
    results_df.to_csv('outputs/walkforward_results.csv', index=False)
    print(f"\nResults saved to: outputs/walkforward_results.csv")
    
    with open('outputs/walkforward_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best params saved to: outputs/walkforward_best_params.json")
    
    strategy_config = build_strategy_config(best_params, args, robustness_stats, window_results)
    with open('outputs/strategy_best_config.json', 'w') as f:
        json.dump(strategy_config, f, indent=2, default=str)
    print(f"Full strategy config saved to: outputs/strategy_best_config.json")
    
    with open('outputs/walkforward_summary.txt', 'w') as f:
        f.write(summary_text)
    print(f"Summary saved to: outputs/walkforward_summary.txt")
    
    score_policy_summary = compute_score_policy_wf_summary(window_results)
    if not score_policy_summary.empty:
        score_policy_summary.to_csv('outputs/score_policy_wf_summary.csv', index=False)
        print(f"Score policy summary saved to: outputs/score_policy_wf_summary.csv")
        
        scoring_defaults = select_best_scoring_defaults(
            score_policy_summary,
            min_pass_rate=args.min_pass_rate,
            min_median_test_score=args.min_median_test_score
        )
        
        with open('outputs/chosen_scoring_defaults.json', 'w') as f:
            json.dump(scoring_defaults, f, indent=2)
        print(f"Chosen scoring defaults saved to: outputs/chosen_scoring_defaults.json")
        
        if scoring_defaults.get('warning'):
            print(f"  Warning: {scoring_defaults['warning']}")
        print(f"  Selected: score_policy={scoring_defaults['chosen_defaults']['score_policy']}, "
              f"trend={scoring_defaults['chosen_defaults']['trend_score_mode']}, "
              f"min_score={scoring_defaults['chosen_defaults']['min_pattern_score']}, "
              f"top_k={scoring_defaults['chosen_defaults']['top_k_per_day']}")
    
    cache_stats = get_cache_stats(args.price_cache_dir)
    
    manifest = create_manifest(
        run_id=run_id,
        mode='optimize',
        command_line=' '.join(['python', 'optimize.py'] + sys.argv[1:]),
        symbols_used=symbols,
        symbols_requested=symbols_requested,
        config=strategy_config,
        cache_stats=cache_stats,
        date_range=None,
        universe_type=args.universe,
        liquidity_config=liquidity_config,
        extra_info={
            'symbols_seed': args.symbols_seed,
            'grid_seed': args.grid_seed,
            'train_bars': args.train_bars,
            'test_bars': args.test_bars,
            'step_bars': args.step_bars,
            'objective': args.objective,
            'robustness_stats': robustness_stats,
        }
    )
    
    manifest_path, symbols_path = save_manifest(manifest, symbols)
    print(f"Run manifest saved to: {manifest_path}")
    print(f"Symbols list saved to: {symbols_path}")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print(f"Run ID: {run_id}")
    print("=" * 60)


if __name__ == '__main__':
    main()
