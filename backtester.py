#!/usr/bin/env python3
"""
Backtesting Engine for Double Bottom Pattern Scanner.

This module provides a backtesting framework to simulate trades based on
detected double bottom patterns and evaluate strategy performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategy import TradeSignal


@dataclass
class Trade:
    """Represents a completed or open trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    shares: int
    
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    def __repr__(self):
        status = "OPEN" if self.exit_date is None else "CLOSED"
        return (f"Trade({self.symbol}, {status}, entry=${self.entry_price:.2f}, "
                f"P&L=${self.pnl:.2f} ({self.pnl_pct:+.2f}%))")


@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: List[Trade]
    equity_curve: pd.DataFrame
    initial_capital: float
    final_capital: float
    
    def get_trade_blotter(self) -> pd.DataFrame:
        """Generate a trade blotter DataFrame."""
        records = []
        for trade in self.trades:
            records.append({
                'symbol': trade.symbol,
                'entry_date': trade.entry_date,
                'entry_price': trade.entry_price,
                'shares': trade.shares,
                'position_value': trade.entry_price * trade.shares,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'exit_date': trade.exit_date,
                'exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'holding_days': (trade.exit_date - trade.entry_date).days if trade.exit_date else None
            })
        return pd.DataFrame(records)


class Backtest:
    """
    Backtesting engine that simulates trades bar-by-bar.
    
    Features:
    - Trade tracking (entry, exit, P&L)
    - Multiple exit conditions (stop loss, take profit, trailing stop, max hold)
    - Equity curve generation
    - Trade blotter
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 max_hold_days: int = 60,
                 use_trailing_stop: bool = False,
                 trailing_stop_pct: float = 0.05):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            position_size: Fraction of capital per trade (0.1 = 10%)
            max_hold_days: Maximum days to hold a position
            use_trailing_stop: Whether to use trailing stops
            trailing_stop_pct: Trailing stop percentage from high
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_hold_days = max_hold_days
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
    def reset(self):
        """Reset the backtester state."""
        self.capital = self.initial_capital
        self.trades = []
        self.open_trades = []
        self.equity_history = []
    
    def _calculate_shares(self, entry_price: float) -> int:
        """Calculate number of shares to buy based on position sizing."""
        position_value = self.capital * self.position_size
        shares = int(position_value / entry_price)
        return max(1, shares)
    
    def _open_trade(self, signal: TradeSignal, current_date: datetime) -> Optional[Trade]:
        """Open a new trade based on a signal."""
        position_value = self.capital * self.position_size
        shares = self._calculate_shares(signal.entry_price)
        required_capital = shares * signal.entry_price
        
        if required_capital > self.capital:
            shares = int(self.capital / signal.entry_price)
            if shares < 1:
                return None
        
        trade = Trade(
            symbol=signal.symbol,
            entry_date=signal.entry_date,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=shares * signal.entry_price,
            shares=shares
        )
        
        self.capital -= shares * signal.entry_price
        self.open_trades.append(trade)
        
        return trade
    
    def _close_trade(self, trade: Trade, exit_date: datetime, 
                     exit_price: float, exit_reason: str):
        """Close an open trade."""
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl = (exit_price - trade.entry_price) * trade.shares
        trade.pnl_pct = ((exit_price / trade.entry_price) - 1) * 100
        
        self.capital += trade.shares * exit_price
        
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        
        self.trades.append(trade)
    
    def _check_exit_conditions(self, trade: Trade, bar: pd.Series, 
                               current_date: datetime, high_since_entry: float) -> Optional[Tuple[float, str]]:
        """
        Check if any exit condition is met.
        
        Args:
            trade: The open trade
            bar: Current price bar (OHLCV)
            current_date: Current date
            high_since_entry: Highest price since entry
            
        Returns:
            Tuple of (exit_price, exit_reason) or None
        """
        low = bar['Low']
        high = bar['High']
        close = bar['Close']
        
        if low <= trade.stop_loss:
            return (trade.stop_loss, 'stop_loss')
        
        if high >= trade.take_profit:
            return (trade.take_profit, 'take_profit')
        
        if self.use_trailing_stop:
            trailing_stop = high_since_entry * (1 - self.trailing_stop_pct)
            if trailing_stop > trade.stop_loss and low <= trailing_stop:
                return (trailing_stop, 'trailing_stop')
        
        days_held = (current_date - trade.entry_date).days
        if days_held >= self.max_hold_days:
            return (close, 'max_hold_period')
        
        return None
    
    def run(self, signals: List[TradeSignal], 
            price_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run the backtest simulation.
        
        Args:
            signals: List of trade signals
            price_data: Dictionary mapping symbols to price DataFrames
            
        Returns:
            BacktestResult object with all results
        """
        self.reset()
        
        if not signals:
            return BacktestResult(
                trades=[],
                equity_curve=pd.DataFrame(columns=['date', 'equity']),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital
            )
        
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)
        
        if not all_dates:
            return BacktestResult(
                trades=[],
                equity_curve=pd.DataFrame(columns=['date', 'equity']),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital
            )
        
        signal_queue = sorted(signals, key=lambda s: s.entry_date)
        signal_idx = 0
        
        trade_highs: Dict[int, float] = {}
        
        for current_date in all_dates:
            while (signal_idx < len(signal_queue) and 
                   signal_queue[signal_idx].entry_date <= current_date):
                signal = signal_queue[signal_idx]
                
                already_has_position = any(
                    t.symbol == signal.symbol for t in self.open_trades
                )
                
                if not already_has_position:
                    trade = self._open_trade(signal, current_date)
                    if trade:
                        trade_highs[id(trade)] = trade.entry_price
                
                signal_idx += 1
            
            trades_to_close = []
            for trade in self.open_trades:
                if trade.symbol not in price_data:
                    continue
                
                df = price_data[trade.symbol]
                if current_date not in df.index:
                    continue
                
                bar = df.loc[current_date]
                
                if id(trade) in trade_highs:
                    trade_highs[id(trade)] = max(trade_highs[id(trade)], bar['High'])
                else:
                    trade_highs[id(trade)] = bar['High']
                
                high_since_entry = trade_highs[id(trade)]
                exit_result = self._check_exit_conditions(
                    trade, bar, current_date, high_since_entry
                )
                
                if exit_result:
                    exit_price, exit_reason = exit_result
                    trades_to_close.append((trade, exit_price, exit_reason))
            
            for trade, exit_price, exit_reason in trades_to_close:
                self._close_trade(trade, current_date, exit_price, exit_reason)
                if id(trade) in trade_highs:
                    del trade_highs[id(trade)]
            
            open_position_value = sum(
                self._get_current_value(trade, current_date, price_data)
                for trade in self.open_trades
            )
            total_equity = self.capital + open_position_value
            self.equity_history.append((current_date, total_equity))
        
        for trade in list(self.open_trades):
            if trade.symbol in price_data:
                df = price_data[trade.symbol]
                if len(df) > 0:
                    last_date = df.index[-1]
                    last_price = df['Close'].iloc[-1]
                    self._close_trade(trade, last_date, last_price, 'end_of_data')
        
        equity_curve = pd.DataFrame(self.equity_history, columns=['date', 'equity'])
        equity_curve.set_index('date', inplace=True)
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_capital=self.capital
        )
    
    def _get_current_value(self, trade: Trade, current_date: datetime,
                           price_data: Dict[str, pd.DataFrame]) -> float:
        """Get current market value of an open trade."""
        if trade.symbol not in price_data:
            return trade.shares * trade.entry_price
        
        df = price_data[trade.symbol]
        
        if current_date in df.index:
            current_price = df.loc[current_date, 'Close']
        else:
            df_before = df[df.index <= current_date]
            if len(df_before) > 0:
                current_price = df_before['Close'].iloc[-1]
            else:
                current_price = trade.entry_price
        
        return trade.shares * current_price


def plot_equity_curve(result: BacktestResult, save_path: Optional[str] = None,
                      show: bool = True):
    """
    Plot the equity curve from backtest results.
    
    Args:
        result: BacktestResult object
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if result.equity_curve.empty:
        print("No equity data to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    dates = result.equity_curve.index
    equity = result.equity_curve['equity']
    
    ax1.plot(dates, equity, color='blue', linewidth=1.5, label='Equity')
    ax1.axhline(y=result.initial_capital, color='gray', linestyle='--', 
                linewidth=1, alpha=0.7, label=f'Initial Capital: ${result.initial_capital:,.0f}')
    ax1.fill_between(dates, result.initial_capital, equity, 
                     where=(equity >= result.initial_capital),
                     color='green', alpha=0.3)
    ax1.fill_between(dates, result.initial_capital, equity,
                     where=(equity < result.initial_capital),
                     color='red', alpha=0.3)
    
    returns = (result.final_capital / result.initial_capital - 1) * 100
    ax1.set_title(f'Equity Curve - Double Bottom Strategy\n'
                  f'Total Trades: {len(result.trades)} | '
                  f'Final Capital: ${result.final_capital:,.2f} | '
                  f'Return: {returns:+.2f}%',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    ax2.fill_between(dates, 0, drawdown, color='red', alpha=0.5)
    ax2.plot(dates, drawdown, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    max_dd = drawdown.min()
    ax2.axhline(y=max_dd, color='darkred', linestyle='--', linewidth=1,
                label=f'Max Drawdown: {max_dd:.2f}%')
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Equity curve saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_trade_blotter(result: BacktestResult, max_trades: int = 20):
    """Print a formatted trade blotter."""
    blotter = result.get_trade_blotter()
    
    if blotter.empty:
        print("\nNo trades executed.")
        return
    
    print("\n" + "="*100)
    print("TRADE BLOTTER")
    print("="*100)
    
    display_cols = ['symbol', 'entry_date', 'entry_price', 'shares', 
                    'exit_date', 'exit_price', 'exit_reason', 'pnl', 'pnl_pct', 'holding_days']
    
    blotter_display = blotter[display_cols].head(max_trades)
    
    blotter_display['entry_date'] = blotter_display['entry_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
    )
    blotter_display['exit_date'] = blotter_display['exit_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
    )
    blotter_display['pnl'] = blotter_display['pnl'].apply(lambda x: f"${x:+.2f}")
    blotter_display['pnl_pct'] = blotter_display['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
    
    print(blotter_display.to_string(index=False))
    
    if len(blotter) > max_trades:
        print(f"\n... and {len(blotter) - max_trades} more trades")
    
    print("="*100)
