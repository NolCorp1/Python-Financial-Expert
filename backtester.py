#!/usr/bin/env python3
"""
No-Lookahead Backtesting Engine for Double Bottom Pattern Scanner.

Features:
- Gap-aware exit logic (gap-through stops/targets)
- Slippage and commission modeling
- Risk-based position sizing (% equity risk per trade)
- Overlap control (one position per symbol, max positions)
- Trade blotter with R-multiple tracking
- Daily equity curve with drawdown
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math

from strategy import TradeSignal


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100000.0
    risk_fraction_per_trade: float = 0.01
    max_positions: int = 10
    one_position_per_symbol: bool = True
    allow_same_day_reentry: bool = False
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    min_price: float = 1.0
    max_hold_days: Optional[int] = None


@dataclass
class OpenPosition:
    """Tracks an open position during backtest."""
    symbol: str
    pattern_id: str
    entry_kind: str
    entry_date: pd.Timestamp
    entry_fill: float
    stop_loss: float
    take_profit: float
    shares: int
    entry_commission: float
    slippage_bps: float
    meta: dict = field(default_factory=dict)


def group_signals_by_symbol(signals: List[TradeSignal]) -> Dict[str, List[TradeSignal]]:
    """Group signals by symbol for efficient processing."""
    result = {}
    for sig in signals:
        if sig.symbol not in result:
            result[sig.symbol] = []
        result[sig.symbol].append(sig)
    for sym in result:
        result[sym].sort(key=lambda s: s.entry_date)
    return result


def run_backtest(
    signals_by_symbol: Dict[str, List[TradeSignal]],
    price_data_by_symbol: Dict[str, pd.DataFrame],
    cfg: BacktestConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run no-lookahead backtest simulation.
    
    Args:
        signals_by_symbol: Dict mapping symbol to list of TradeSignals
        price_data_by_symbol: Dict mapping symbol to OHLCV DataFrame
        cfg: BacktestConfig with simulation parameters
        
    Returns:
        trades_df: Trade blotter with all completed trades
        equity_df: Daily equity curve with drawdown
    """
    cash = cfg.initial_capital
    open_positions: Dict[str, OpenPosition] = {}
    completed_trades: List[dict] = []
    equity_history: List[dict] = []
    
    exited_today: Dict[pd.Timestamp, set] = {}
    
    all_signals = []
    for sym, sigs in signals_by_symbol.items():
        for sig in sigs:
            all_signals.append(sig)
    all_signals.sort(key=lambda s: s.entry_date)
    
    all_dates = set()
    for df in price_data_by_symbol.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    
    if not all_dates:
        return pd.DataFrame(), pd.DataFrame()
    
    signal_idx = 0
    running_max_equity = cfg.initial_capital
    
    for current_date in all_dates:
        positions_to_close = []
        for sym, pos in list(open_positions.items()):
            if sym not in price_data_by_symbol:
                continue
            df = price_data_by_symbol[sym]
            if current_date not in df.index:
                continue
            
            if current_date <= pos.entry_date:
                continue
            
            bar = df.loc[current_date]
            open_price = bar['Open']
            high_price = bar['High']
            low_price = bar['Low']
            
            exit_price = None
            exit_reason = None
            
            if open_price <= pos.stop_loss:
                exit_price = open_price * (1 - cfg.slippage_bps / 10000)
                exit_reason = "STOP"
            elif open_price >= pos.take_profit:
                exit_price = open_price * (1 - cfg.slippage_bps / 10000)
                exit_reason = "TARGET"
            elif low_price <= pos.stop_loss and high_price >= pos.take_profit:
                exit_price = pos.stop_loss * (1 - cfg.slippage_bps / 10000)
                exit_reason = "STOP"
            elif low_price <= pos.stop_loss:
                exit_price = pos.stop_loss * (1 - cfg.slippage_bps / 10000)
                exit_reason = "STOP"
            elif high_price >= pos.take_profit:
                exit_price = pos.take_profit * (1 - cfg.slippage_bps / 10000)
                exit_reason = "TARGET"
            elif cfg.max_hold_days is not None:
                hold_days = (current_date - pos.entry_date).days
                if hold_days >= cfg.max_hold_days:
                    exit_price = bar['Close'] * (1 - cfg.slippage_bps / 10000)
                    exit_reason = "TIME"
            
            if exit_price is not None:
                positions_to_close.append((sym, pos, current_date, exit_price, exit_reason))
        
        for sym, pos, exit_date, exit_price, exit_reason in positions_to_close:
            exit_commission = cfg.commission_per_trade
            proceeds = exit_price * pos.shares - exit_commission
            cash += proceeds
            
            pnl_dollars = (exit_price - pos.entry_fill) * pos.shares - pos.entry_commission - exit_commission
            risk_amount = (pos.entry_fill - pos.stop_loss) * pos.shares
            pnl_r_multiple = pnl_dollars / risk_amount if risk_amount > 0 else 0.0
            hold_days = (exit_date - pos.entry_date).days
            
            trade_record = {
                'symbol': sym,
                'pattern_id': pos.pattern_id,
                'entry_kind': pos.entry_kind,
                'entry_date': pos.entry_date,
                'entry_price': round(pos.entry_fill, 2),
                'stop_loss': round(pos.stop_loss, 2),
                'take_profit': round(pos.take_profit, 2),
                'shares': pos.shares,
                'exit_date': exit_date,
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'pnl_dollars': round(pnl_dollars, 2),
                'pnl_r_multiple': round(pnl_r_multiple, 2),
                'hold_days': hold_days,
                'slippage_bps': pos.slippage_bps,
                'commissions': round(pos.entry_commission + exit_commission, 2),
                'meta_json': json.dumps(pos.meta) if pos.meta else '{}'
            }
            completed_trades.append(trade_record)
            
            del open_positions[sym]
            
            if exit_date not in exited_today:
                exited_today[exit_date] = set()
            exited_today[exit_date].add(sym)
        
        while signal_idx < len(all_signals):
            signal = all_signals[signal_idx]
            
            if signal.entry_date > current_date:
                break
            
            if signal.entry_date < current_date:
                signal_idx += 1
                continue
            
            sym = signal.symbol
            
            if sym not in price_data_by_symbol:
                signal_idx += 1
                continue
            
            df = price_data_by_symbol[sym]
            if current_date not in df.index:
                signal_idx += 1
                continue
            
            bar = df.loc[current_date]
            open_price = bar['Open']
            
            if open_price < cfg.min_price:
                signal_idx += 1
                continue
            
            if cfg.one_position_per_symbol and sym in open_positions:
                signal_idx += 1
                continue
            
            if len(open_positions) >= cfg.max_positions:
                signal_idx += 1
                continue
            
            if not cfg.allow_same_day_reentry:
                if current_date in exited_today and sym in exited_today[current_date]:
                    signal_idx += 1
                    continue
            
            entry_fill = open_price * (1 + cfg.slippage_bps / 10000)
            stop_dist = entry_fill - signal.stop_loss
            
            if stop_dist <= 0:
                signal_idx += 1
                continue
            
            current_equity = cash + sum(
                pos.shares * _get_prior_close(price_data_by_symbol, pos.symbol, current_date, pos.entry_fill)
                for pos in open_positions.values()
            )
            
            risk_budget = cfg.risk_fraction_per_trade * current_equity
            shares = int(math.floor(risk_budget / stop_dist))
            
            if shares < 1:
                signal_idx += 1
                continue
            
            required_capital = entry_fill * shares + cfg.commission_per_trade
            if required_capital > cash:
                shares = int((cash - cfg.commission_per_trade) / entry_fill)
                if shares < 1:
                    signal_idx += 1
                    continue
            
            entry_commission = cfg.commission_per_trade
            cash -= (entry_fill * shares + entry_commission)
            
            position = OpenPosition(
                symbol=sym,
                pattern_id=signal.pattern_id,
                entry_kind=signal.entry_kind,
                entry_date=current_date,
                entry_fill=entry_fill,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                shares=shares,
                entry_commission=entry_commission,
                slippage_bps=cfg.slippage_bps,
                meta=signal.meta if hasattr(signal, 'meta') else {}
            )
            open_positions[sym] = position
            
            signal_idx += 1
        
        open_value = sum(
            pos.shares * _get_price(price_data_by_symbol, pos.symbol, current_date, pos.entry_fill)
            for pos in open_positions.values()
        )
        total_equity = cash + open_value
        running_max_equity = max(running_max_equity, total_equity)
        drawdown_pct = (total_equity / running_max_equity - 1) * 100 if running_max_equity > 0 else 0.0
        
        equity_history.append({
            'date': current_date,
            'equity': round(total_equity, 2),
            'drawdown_pct': round(drawdown_pct, 2)
        })
    
    for sym, pos in list(open_positions.items()):
        if sym in price_data_by_symbol:
            df = price_data_by_symbol[sym]
            if len(df) > 0:
                last_date = df.index[-1]
                last_close = df['Close'].iloc[-1]
                exit_price = last_close * (1 - cfg.slippage_bps / 10000)
                exit_commission = cfg.commission_per_trade
                
                proceeds = exit_price * pos.shares - exit_commission
                cash += proceeds
                
                pnl_dollars = (exit_price - pos.entry_fill) * pos.shares - pos.entry_commission - exit_commission
                risk_amount = (pos.entry_fill - pos.stop_loss) * pos.shares
                pnl_r_multiple = pnl_dollars / risk_amount if risk_amount > 0 else 0.0
                hold_days = (last_date - pos.entry_date).days
                
                trade_record = {
                    'symbol': sym,
                    'pattern_id': pos.pattern_id,
                    'entry_kind': pos.entry_kind,
                    'entry_date': pos.entry_date,
                    'entry_price': round(pos.entry_fill, 2),
                    'stop_loss': round(pos.stop_loss, 2),
                    'take_profit': round(pos.take_profit, 2),
                    'shares': pos.shares,
                    'exit_date': last_date,
                    'exit_price': round(exit_price, 2),
                    'exit_reason': 'EOD',
                    'pnl_dollars': round(pnl_dollars, 2),
                    'pnl_r_multiple': round(pnl_r_multiple, 2),
                    'hold_days': hold_days,
                    'slippage_bps': pos.slippage_bps,
                    'commissions': round(pos.entry_commission + exit_commission, 2),
                    'meta_json': json.dumps(pos.meta) if pos.meta else '{}'
                }
                completed_trades.append(trade_record)
    
    trades_df = pd.DataFrame(completed_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)
    
    equity_df = pd.DataFrame(equity_history)
    if not equity_df.empty:
        equity_df = equity_df.sort_values('date').reset_index(drop=True)
    
    return trades_df, equity_df


def _get_price(price_data: Dict[str, pd.DataFrame], symbol: str, 
               date: pd.Timestamp, fallback: float) -> float:
    """Get closing price for a symbol on a date, with fallback."""
    if symbol not in price_data:
        return fallback
    df = price_data[symbol]
    if date in df.index:
        return df.loc[date, 'Close']
    df_before = df[df.index <= date]
    if len(df_before) > 0:
        return df_before['Close'].iloc[-1]
    return fallback


def _get_prior_close(price_data: Dict[str, pd.DataFrame], symbol: str, 
                     date: pd.Timestamp, fallback: float) -> float:
    """Get the most recent close BEFORE the given date (no lookahead)."""
    if symbol not in price_data:
        return fallback
    df = price_data[symbol]
    df_before = df[df.index < date]
    if len(df_before) > 0:
        return df_before['Close'].iloc[-1]
    return fallback


@dataclass
class Trade:
    """Represents a completed or open trade (legacy compatibility)."""
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


@dataclass
class BacktestResult:
    """Container for backtest results (legacy compatibility)."""
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
    Legacy backtesting engine wrapper.
    Uses new run_backtest internally for compatibility.
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 max_hold_days: int = 60,
                 use_trailing_stop: bool = False,
                 trailing_stop_pct: float = 0.05):
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
        self.capital = self.initial_capital
        self.trades = []
        self.open_trades = []
        self.equity_history = []
    
    def run(self, signals: List[TradeSignal], 
            price_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Run backtest using new engine under the hood."""
        if not signals:
            return BacktestResult(
                trades=[],
                equity_curve=pd.DataFrame(columns=['date', 'equity']),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital
            )
        
        cfg = BacktestConfig(
            initial_capital=self.initial_capital,
            risk_fraction_per_trade=self.position_size,
            max_positions=10,
            one_position_per_symbol=True,
            max_hold_days=self.max_hold_days
        )
        
        signals_by_symbol = group_signals_by_symbol(signals)
        trades_df, equity_df = run_backtest(signals_by_symbol, price_data, cfg)
        
        legacy_trades = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                trade = Trade(
                    symbol=row['symbol'],
                    entry_date=row['entry_date'],
                    entry_price=row['entry_price'],
                    stop_loss=row['stop_loss'],
                    take_profit=row['take_profit'],
                    position_size=row['entry_price'] * row['shares'],
                    shares=row['shares'],
                    exit_date=row['exit_date'],
                    exit_price=row['exit_price'],
                    exit_reason=row['exit_reason'],
                    pnl=row['pnl_dollars'],
                    pnl_pct=((row['exit_price'] / row['entry_price']) - 1) * 100
                )
                legacy_trades.append(trade)
        
        if not equity_df.empty:
            equity_curve = equity_df[['date', 'equity']].copy()
            equity_curve.set_index('date', inplace=True)
            final_capital = equity_df['equity'].iloc[-1]
        else:
            equity_curve = pd.DataFrame(columns=['equity'])
            final_capital = self.initial_capital
        
        return BacktestResult(
            trades=legacy_trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_capital=final_capital
        )


def plot_equity_curve(result: BacktestResult, save_path: Optional[str] = None,
                      show: bool = True):
    """Plot the equity curve from backtest results."""
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


def plot_equity_from_df(equity_df: pd.DataFrame, initial_capital: float,
                        save_path: Optional[str] = None, show: bool = False):
    """Plot equity curve from DataFrame."""
    if equity_df.empty:
        print("No equity data to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    dates = equity_df['date']
    equity = equity_df['equity']
    final_capital = equity.iloc[-1]
    
    ax1.plot(dates, equity, color='blue', linewidth=1.5, label='Equity')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', 
                linewidth=1, alpha=0.7, label=f'Initial Capital: ${initial_capital:,.0f}')
    ax1.fill_between(dates, initial_capital, equity, 
                     where=(equity >= initial_capital),
                     color='green', alpha=0.3)
    ax1.fill_between(dates, initial_capital, equity,
                     where=(equity < initial_capital),
                     color='red', alpha=0.3)
    
    returns = (final_capital / initial_capital - 1) * 100
    ax1.set_title(f'Equity Curve - Double Bottom Strategy\n'
                  f'Final Capital: ${final_capital:,.2f} | Return: {returns:+.2f}%',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    drawdown = equity_df['drawdown_pct']
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
    
    blotter_display = blotter[display_cols].head(max_trades).copy()
    
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


def print_trades_df(trades_df: pd.DataFrame, max_trades: int = 20):
    """Print formatted trade blotter from new run_backtest output."""
    if trades_df.empty:
        print("\nNo trades executed.")
        return
    
    print("\n" + "="*120)
    print("TRADE BLOTTER")
    print("="*120)
    
    display_cols = ['symbol', 'entry_kind', 'entry_date', 'entry_price', 'shares',
                    'exit_date', 'exit_price', 'exit_reason', 'pnl_dollars', 'pnl_r_multiple', 'hold_days']
    
    cols_present = [c for c in display_cols if c in trades_df.columns]
    blotter_display = trades_df[cols_present].head(max_trades).copy()
    
    if 'entry_date' in blotter_display.columns:
        blotter_display['entry_date'] = blotter_display['entry_date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
        )
    if 'exit_date' in blotter_display.columns:
        blotter_display['exit_date'] = blotter_display['exit_date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
        )
    if 'pnl_dollars' in blotter_display.columns:
        blotter_display['pnl_dollars'] = blotter_display['pnl_dollars'].apply(lambda x: f"${x:+.2f}")
    if 'pnl_r_multiple' in blotter_display.columns:
        blotter_display['pnl_r_multiple'] = blotter_display['pnl_r_multiple'].apply(lambda x: f"{x:+.2f}R")
    
    print(blotter_display.to_string(index=False))
    
    if len(trades_df) > max_trades:
        print(f"\n... and {len(trades_df) - max_trades} more trades")
    
    print("="*120)
