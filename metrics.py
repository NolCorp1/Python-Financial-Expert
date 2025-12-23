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
