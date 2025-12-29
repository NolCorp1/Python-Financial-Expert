#!/usr/bin/env python3
"""
Performance Metrics Module for Double Bottom Pattern Backtester.

This module calculates comprehensive trading performance statistics
from backtest results.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from backtester import BacktestResult, Trade


def calculate_metrics(result: BacktestResult) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from backtest results.
    
    Args:
        result: BacktestResult object from backtester
        
    Returns:
        Dictionary with all performance metrics
    """
    metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'total_return': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'average_trade_duration': 0.0,
        'average_win': 0.0,
        'average_loss': 0.0,
        'largest_win': 0.0,
        'largest_loss': 0.0,
        'gross_profit': 0.0,
        'gross_loss': 0.0,
        'net_profit': 0.0,
        'initial_capital': result.initial_capital,
        'final_capital': result.final_capital,
    }
    
    if not result.trades:
        return metrics
    
    trades = result.trades
    metrics['total_trades'] = len(trades)
    
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    metrics['winning_trades'] = len(winning_trades)
    metrics['losing_trades'] = len(losing_trades)
    
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
    
    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in losing_trades))
    
    metrics['gross_profit'] = round(gross_profit, 2)
    metrics['gross_loss'] = round(gross_loss, 2)
    metrics['net_profit'] = round(gross_profit - gross_loss, 2)
    
    if gross_loss > 0:
        metrics['profit_factor'] = round(gross_profit / gross_loss, 2)
    else:
        metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
    
    metrics['total_return'] = round(
        ((result.final_capital / result.initial_capital) - 1) * 100, 2
    )
    
    if not result.equity_curve.empty:
        equity = result.equity_curve['equity']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        metrics['max_drawdown'] = round(drawdown.min() * 100, 2)
    
    metrics['sharpe_ratio'] = round(calculate_sharpe_ratio(result), 2)
    
    durations = []
    for trade in trades:
        if trade.exit_date and trade.entry_date:
            duration = (trade.exit_date - trade.entry_date).days
            durations.append(duration)
    
    if durations:
        metrics['average_trade_duration'] = round(np.mean(durations), 1)
    
    if winning_trades:
        metrics['average_win'] = round(np.mean([t.pnl for t in winning_trades]), 2)
        metrics['largest_win'] = round(max(t.pnl for t in winning_trades), 2)
    
    if losing_trades:
        metrics['average_loss'] = round(np.mean([t.pnl for t in losing_trades]), 2)
        metrics['largest_loss'] = round(min(t.pnl for t in losing_trades), 2)
    
    return metrics


def calculate_sharpe_ratio(result: BacktestResult, risk_free_rate: float = 0.02) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        result: BacktestResult object
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Annualized Sharpe ratio
    """
    if result.equity_curve.empty or len(result.equity_curve) < 2:
        return 0.0
    
    equity = result.equity_curve['equity']
    returns = equity.pct_change().dropna()
    
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    return sharpe


def calculate_sortino_ratio(result: BacktestResult, risk_free_rate: float = 0.02) -> float:
    """
    Calculate the annualized Sortino ratio.
    
    Args:
        result: BacktestResult object
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Annualized Sortino ratio
    """
    if result.equity_curve.empty or len(result.equity_curve) < 2:
        return 0.0
    
    equity = result.equity_curve['equity']
    returns = equity.pct_change().dropna()
    
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) < 2:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = negative_returns.std()
    
    if downside_std == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(252)
    
    return round(sortino, 2)


def calculate_calmar_ratio(result: BacktestResult) -> float:
    """
    Calculate the Calmar ratio (annualized return / max drawdown).
    
    Args:
        result: BacktestResult object
        
    Returns:
        Calmar ratio
    """
    if result.equity_curve.empty or len(result.equity_curve) < 2:
        return 0.0
    
    equity = result.equity_curve['equity']
    
    days = (equity.index[-1] - equity.index[0]).days
    if days < 1:
        return 0.0
    
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (365 / days) - 1
    
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    if max_drawdown == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return round(annualized_return / max_drawdown, 2)


def print_metrics(metrics: Dict[str, Any]):
    """
    Print formatted performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
    """
    print("\n" + "="*60)
    print("BACKTEST PERFORMANCE METRICS")
    print("="*60)
    
    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades:        {metrics['total_trades']}")
    print(f"Winning Trades:      {metrics['winning_trades']}")
    print(f"Losing Trades:       {metrics['losing_trades']}")
    print(f"Win Rate:            {metrics['win_rate']:.1f}%")
    
    print("\n--- PROFIT & LOSS ---")
    print(f"Initial Capital:     ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital:       ${metrics['final_capital']:,.2f}")
    print(f"Net Profit:          ${metrics['net_profit']:+,.2f}")
    print(f"Total Return:        {metrics['total_return']:+.2f}%")
    print(f"Gross Profit:        ${metrics['gross_profit']:,.2f}")
    print(f"Gross Loss:          ${metrics['gross_loss']:,.2f}")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    
    print("\n--- RISK METRICS ---")
    print(f"Max Drawdown:        {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    
    print("\n--- TRADE ANALYSIS ---")
    print(f"Avg Trade Duration:  {metrics['average_trade_duration']:.1f} days")
    print(f"Average Win:         ${metrics['average_win']:+,.2f}")
    print(f"Average Loss:        ${metrics['average_loss']:+,.2f}")
    print(f"Largest Win:         ${metrics['largest_win']:+,.2f}")
    print(f"Largest Loss:        ${metrics['largest_loss']:+,.2f}")
    
    print("\n" + "="*60)


def export_metrics_to_csv(metrics: Dict[str, Any], filename: str = 'backtest_metrics.csv'):
    """
    Export metrics to a CSV file.
    
    Args:
        metrics: Dictionary of performance metrics
        filename: Output filename
    """
    df = pd.DataFrame([metrics])
    df.to_csv(filename, index=False)
    print(f"Metrics saved to: {filename}")


def enrich_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to trades DataFrame for analysis.
    
    Adds:
    - win: bool (pnl_dollars > 0)
    - abs_r: abs(pnl_r_multiple)
    - capped_r: pnl_r_multiple clipped to [-5, +10]
    - trade_duration_days: copy of hold_days
    - year: entry year
    - month: entry month period
    
    Args:
        trades_df: Raw trades DataFrame from run_backtest()
        
    Returns:
        Enriched DataFrame with derived columns
    """
    if trades_df.empty:
        return trades_df.copy()
    
    df = trades_df.copy()
    
    df['win'] = df['pnl_dollars'] > 0
    df['abs_r'] = df['pnl_r_multiple'].abs()
    df['capped_r'] = df['pnl_r_multiple'].clip(lower=-5, upper=10)
    df['trade_duration_days'] = df['hold_days']
    
    entry_dates = pd.to_datetime(df['entry_date'])
    df['year'] = entry_dates.dt.year
    df['month'] = entry_dates.dt.to_period('M')
    
    return df


def compute_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute core performance metrics from trades DataFrame.
    
    Metrics computed:
    - trade_count, win_rate, avg_r, median_r, expectancy_r
    - avg_win_r, avg_loss_r, win_loss_ratio, profit_factor
    - max_r, min_r, pct_trades_gt_1r, pct_trades_lt_minus_1r
    - avg_hold_days
    
    Args:
        trades_df: Trades DataFrame (raw or enriched)
        
    Returns:
        Dictionary of trade metrics
    """
    if trades_df.empty:
        return {
            'trade_count': 0,
            'win_rate': np.nan,
            'avg_r': np.nan,
            'median_r': np.nan,
            'expectancy_r': np.nan,
            'avg_win_r': np.nan,
            'avg_loss_r': np.nan,
            'win_loss_ratio': np.nan,
            'profit_factor': np.nan,
            'max_r': np.nan,
            'min_r': np.nan,
            'pct_trades_gt_1r': np.nan,
            'pct_trades_lt_minus_1r': np.nan,
            'avg_hold_days': np.nan,
        }
    
    n = len(trades_df)
    r_values = trades_df['pnl_r_multiple']
    wins = trades_df[trades_df['pnl_dollars'] > 0]
    losses = trades_df[trades_df['pnl_dollars'] <= 0]
    
    win_rate = len(wins) / n * 100
    avg_r = r_values.mean()
    median_r = r_values.median()
    expectancy_r = avg_r
    
    avg_win_r = wins['pnl_r_multiple'].mean() if len(wins) > 0 else np.nan
    avg_loss_r = losses['pnl_r_multiple'].mean() if len(losses) > 0 else np.nan
    
    if len(losses) > 0 and avg_loss_r != 0:
        win_loss_ratio = abs(avg_win_r / avg_loss_r) if not np.isnan(avg_win_r) else np.nan
    else:
        win_loss_ratio = np.inf if len(wins) > 0 else np.nan
    
    sum_wins = wins['pnl_dollars'].sum() if len(wins) > 0 else 0
    sum_losses = abs(losses['pnl_dollars'].sum()) if len(losses) > 0 else 0
    
    if sum_losses > 0:
        profit_factor = sum_wins / sum_losses
    else:
        profit_factor = np.inf if sum_wins > 0 else np.nan
    
    max_r = r_values.max()
    min_r = r_values.min()
    pct_trades_gt_1r = (r_values > 1).sum() / n * 100
    pct_trades_lt_minus_1r = (r_values < -1).sum() / n * 100
    avg_hold_days = trades_df['hold_days'].mean()
    
    return {
        'trade_count': n,
        'win_rate': round(win_rate, 1),
        'avg_r': round(avg_r, 2),
        'median_r': round(median_r, 2),
        'expectancy_r': round(expectancy_r, 2),
        'avg_win_r': round(avg_win_r, 2) if not np.isnan(avg_win_r) else np.nan,
        'avg_loss_r': round(avg_loss_r, 2) if not np.isnan(avg_loss_r) else np.nan,
        'win_loss_ratio': round(win_loss_ratio, 2) if not np.isinf(win_loss_ratio) else float('inf'),
        'profit_factor': round(profit_factor, 2) if not np.isinf(profit_factor) else float('inf'),
        'max_r': round(max_r, 2),
        'min_r': round(min_r, 2),
        'pct_trades_gt_1r': round(pct_trades_gt_1r, 1),
        'pct_trades_lt_minus_1r': round(pct_trades_lt_minus_1r, 1),
        'avg_hold_days': round(avg_hold_days, 1),
    }


def compute_equity_metrics(equity_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute equity curve metrics.
    
    Metrics computed:
    - start_equity, end_equity, total_return_pct
    - max_drawdown_pct, max_drawdown_abs
    - recovery_days, equity_volatility
    - sharpe_ratio (optional)
    
    Args:
        equity_df: Equity DataFrame with date, equity, drawdown_pct
        
    Returns:
        Dictionary of equity metrics
    """
    if equity_df.empty:
        return {
            'start_equity': np.nan,
            'end_equity': np.nan,
            'total_return_pct': np.nan,
            'max_drawdown_pct': np.nan,
            'max_drawdown_abs': np.nan,
            'recovery_days': np.nan,
            'equity_volatility': np.nan,
            'sharpe_ratio': np.nan,
        }
    
    equity = equity_df['equity']
    start_equity = equity.iloc[0]
    end_equity = equity.iloc[-1]
    total_return_pct = (end_equity / start_equity - 1) * 100
    
    max_drawdown_pct = equity_df['drawdown_pct'].min()
    peak = equity.cummax()
    drawdown_abs = equity - peak
    max_drawdown_abs = abs(drawdown_abs.min())
    
    underwater = drawdown_abs < 0
    if underwater.any():
        underwater_periods = underwater.astype(int).groupby((~underwater).cumsum()).cumsum()
        recovery_days = int(underwater_periods.max())
    else:
        recovery_days = 0
    
    daily_returns = equity.pct_change().dropna()
    equity_volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else np.nan
    
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan
    
    return {
        'start_equity': round(start_equity, 2),
        'end_equity': round(end_equity, 2),
        'total_return_pct': round(total_return_pct, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'max_drawdown_abs': round(max_drawdown_abs, 2),
        'recovery_days': recovery_days,
        'equity_volatility': round(equity_volatility, 2) if not np.isnan(equity_volatility) else np.nan,
        'sharpe_ratio': round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
    }


def compute_split_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics split by entry_kind (ALL / FORMING / CONFIRMED).
    
    Args:
        trades_df: Trades DataFrame with entry_kind column
        equity_df: Equity DataFrame
        
    Returns:
        Dictionary with keys 'ALL', 'FORMING', 'CONFIRMED',
        each containing trade metrics (and equity metrics for ALL)
    """
    all_trade_metrics = compute_trade_metrics(trades_df)
    all_equity_metrics = compute_equity_metrics(equity_df)
    
    all_metrics = {**all_trade_metrics, **all_equity_metrics}
    
    forming_trades = trades_df[trades_df['entry_kind'] == 'FORMING'] if not trades_df.empty else pd.DataFrame()
    confirmed_trades = trades_df[trades_df['entry_kind'] == 'CONFIRMED'] if not trades_df.empty else pd.DataFrame()
    
    forming_metrics = compute_trade_metrics(forming_trades)
    confirmed_metrics = compute_trade_metrics(confirmed_trades)
    
    return {
        'ALL': all_metrics,
        'FORMING': forming_metrics,
        'CONFIRMED': confirmed_metrics,
    }


def drawdown_attribution(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attribute drawdown periods to trade types.
    
    Identifies drawdown periods and analyzes which trade types
    contributed to them.
    
    Args:
        trades_df: Trades DataFrame with entry_kind
        equity_df: Equity DataFrame with drawdown_pct
        
    Returns:
        DataFrame with drawdown period analysis
    """
    if equity_df.empty or trades_df.empty:
        return pd.DataFrame()
    
    equity = equity_df['equity'].copy()
    dates = equity_df['date'].copy()
    
    peak = equity.cummax()
    in_drawdown = equity < peak
    
    drawdown_periods = []
    start_idx = None
    
    for i in range(len(in_drawdown)):
        if in_drawdown.iloc[i] and start_idx is None:
            start_idx = i
        elif not in_drawdown.iloc[i] and start_idx is not None:
            drawdown_periods.append((start_idx, i - 1))
            start_idx = None
    
    if start_idx is not None:
        drawdown_periods.append((start_idx, len(in_drawdown) - 1))
    
    results = []
    for start_idx, end_idx in drawdown_periods:
        start_date = pd.to_datetime(dates.iloc[start_idx])
        end_date = pd.to_datetime(dates.iloc[end_idx])
        
        dd_equity = equity.iloc[start_idx:end_idx + 1]
        peak_val = equity.iloc[max(0, start_idx - 1)] if start_idx > 0 else equity.iloc[0]
        max_dd_pct = ((dd_equity.min() - peak_val) / peak_val) * 100
        
        trades_df_copy = trades_df.copy()
        trades_df_copy['entry_date'] = pd.to_datetime(trades_df_copy['entry_date'])
        
        dd_trades = trades_df_copy[
            (trades_df_copy['entry_date'] >= start_date) &
            (trades_df_copy['entry_date'] <= end_date)
        ]
        
        n_trades = len(dd_trades)
        avg_r = dd_trades['pnl_r_multiple'].mean() if n_trades > 0 else np.nan
        pct_forming = (dd_trades['entry_kind'] == 'FORMING').sum() / n_trades * 100 if n_trades > 0 else np.nan
        
        results.append({
            'drawdown_start_date': start_date,
            'drawdown_end_date': end_date,
            'max_drawdown_pct': round(max_dd_pct, 2),
            'trades_during_drawdown': n_trades,
            'avg_r_during_dd': round(avg_r, 2) if not np.isnan(avg_r) else np.nan,
            'pct_forming_trades': round(pct_forming, 1) if not np.isnan(pct_forming) else np.nan,
        })
    
    return pd.DataFrame(results)


def print_metrics_report(metrics: Dict[str, Dict[str, Any]]) -> None:
    """
    Print formatted metrics report with ALL / FORMING / CONFIRMED sections.
    
    Args:
        metrics: Dictionary from compute_split_metrics()
    """
    print("\n" + "=" * 50)
    print("OVERALL PERFORMANCE")
    print("=" * 50)
    
    all_m = metrics.get('ALL', {})
    print(f"Trades: {all_m.get('trade_count', 0)}")
    
    win_rate = all_m.get('win_rate', np.nan)
    if not np.isnan(win_rate):
        print(f"Win Rate: {win_rate:.1f}%")
    
    expectancy = all_m.get('expectancy_r', np.nan)
    if not np.isnan(expectancy):
        print(f"Expectancy (R): {expectancy:+.2f}")
    
    pf = all_m.get('profit_factor', np.nan)
    if not np.isnan(pf) and not np.isinf(pf):
        print(f"Profit Factor: {pf:.2f}")
    elif np.isinf(pf):
        print(f"Profit Factor: Inf (no losses)")
    
    max_dd = all_m.get('max_drawdown_pct', np.nan)
    if not np.isnan(max_dd):
        print(f"Max Drawdown: {max_dd:.2f}%")
    
    total_ret = all_m.get('total_return_pct', np.nan)
    if not np.isnan(total_ret):
        print(f"Total Return: {total_ret:+.2f}%")
    
    sharpe = all_m.get('sharpe_ratio', np.nan)
    if not np.isnan(sharpe):
        print(f"Sharpe Ratio: {sharpe:.2f}")
    
    for kind in ['FORMING', 'CONFIRMED']:
        m = metrics.get(kind, {})
        if m.get('trade_count', 0) == 0:
            continue
        
        print("\n" + "-" * 50)
        print(f"{kind} TRADES")
        print("-" * 50)
        
        print(f"Trades: {m.get('trade_count', 0)}")
        
        win_rate = m.get('win_rate', np.nan)
        if not np.isnan(win_rate):
            print(f"Win Rate: {win_rate:.1f}%")
        
        expectancy = m.get('expectancy_r', np.nan)
        if not np.isnan(expectancy):
            print(f"Expectancy (R): {expectancy:+.2f}")
        
        avg_r = m.get('avg_r', np.nan)
        if not np.isnan(avg_r):
            print(f"Avg R: {avg_r:+.2f}")
        
        pf = m.get('profit_factor', np.nan)
        if not np.isnan(pf) and not np.isinf(pf):
            print(f"Profit Factor: {pf:.2f}")
        elif np.isinf(pf):
            print(f"Profit Factor: Inf (no losses)")
        
        avg_hold = m.get('avg_hold_days', np.nan)
        if not np.isnan(avg_hold):
            print(f"Avg Hold Days: {avg_hold:.1f}")
    
    print("\n" + "=" * 50)


def compute_score_bucket_metrics(
    trades_df: pd.DataFrame,
    score_col: str = 'pattern_score',
    pnl_col: str = 'pnl_dollars',
    r_col: str = 'pnl_r_multiple',
    buckets: int = 10
) -> pd.DataFrame:
    """
    Compute performance statistics by pattern score bucket (decile).
    
    Args:
        trades_df: DataFrame with trade results
        score_col: Column name for pattern score
        pnl_col: Column name for PnL in dollars
        r_col: Column name for R-multiple
        buckets: Number of buckets (default 10 for deciles)
    
    Returns:
        DataFrame with bucket stats
    """
    if trades_df.empty or score_col not in trades_df.columns:
        return pd.DataFrame()
    
    df = trades_df.copy()
    df = df.dropna(subset=[score_col])
    
    if df.empty or len(df) < buckets:
        try:
            df['score_bucket'] = pd.cut(
                df[score_col],
                bins=min(buckets, len(df)),
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            df['score_bucket'] = 0
    else:
        try:
            df['score_bucket'] = pd.qcut(
                df[score_col], 
                buckets, 
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            df['score_bucket'] = pd.cut(
                df[score_col],
                bins=buckets,
                labels=False
            )
    
    stats = []
    for bucket in sorted(df['score_bucket'].dropna().unique()):
        bucket_df = df[df['score_bucket'] == bucket]
        
        n_trades = len(bucket_df)
        if n_trades == 0:
            continue
        
        min_score = bucket_df[score_col].min()
        max_score = bucket_df[score_col].max()
        
        wins = bucket_df[bucket_df[pnl_col] > 0]
        losses = bucket_df[bucket_df[pnl_col] <= 0]
        
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
        
        total_pnl = bucket_df[pnl_col].sum()
        avg_r = bucket_df[r_col].mean() if r_col in bucket_df.columns else 0
        
        gross_profit = wins[pnl_col].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses[pnl_col].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
        
        stats.append({
            'bucket': int(bucket),
            'score_range': f'{min_score:.0f}-{max_score:.0f}',
            'min_score': round(min_score, 1),
            'max_score': round(max_score, 1),
            'trade_count': n_trades,
            'win_rate': round(win_rate, 1),
            'avg_r': round(avg_r, 3),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2)
        })
    
    result = pd.DataFrame(stats)
    if not result.empty:
        result = result.sort_values('min_score')
    
    return result


def compute_threshold_performance(
    trades_df: pd.DataFrame,
    thresholds: List[float] = [0, 50, 60, 70, 80],
    score_col: str = 'pattern_score',
    pnl_col: str = 'pnl_dollars',
    r_col: str = 'pnl_r_multiple'
) -> pd.DataFrame:
    """
    Compare performance at different minimum score thresholds.
    
    Args:
        trades_df: DataFrame with trade results
        thresholds: Score thresholds to compare
        score_col: Column name for pattern score
        pnl_col: Column name for PnL in dollars
        r_col: Column name for R-multiple
    
    Returns:
        DataFrame with threshold comparison
    """
    if trades_df.empty or score_col not in trades_df.columns:
        return pd.DataFrame()
    
    results = []
    
    for thresh in thresholds:
        df = trades_df[trades_df[score_col] >= thresh] if thresh > 0 else trades_df
        
        n_trades = len(df)
        if n_trades == 0:
            results.append({
                'min_score': thresh,
                'trade_count': 0,
                'win_rate': 0,
                'avg_r': 0,
                'profit_factor': 0,
                'total_pnl': 0
            })
            continue
        
        wins = df[df[pnl_col] > 0]
        losses = df[df[pnl_col] <= 0]
        win_rate = len(wins) / n_trades * 100
        
        total_pnl = df[pnl_col].sum()
        avg_r = df[r_col].mean() if r_col in df.columns else 0
        
        gross_profit = wins[pnl_col].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses[pnl_col].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
        
        results.append({
            'min_score': thresh,
            'trade_count': n_trades,
            'win_rate': round(win_rate, 1),
            'avg_r': round(avg_r, 3),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2)
        })
    
    return pd.DataFrame(results)
