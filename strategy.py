#!/usr/bin/env python3
"""
Trading Strategy Module for Double Bottom Pattern Scanner.

This module provides signal generation based on detected double bottom patterns.
Entry signals are generated when price closes above the neckline.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class TradeSignal:
    """Represents a trading signal from a detected pattern."""
    symbol: str
    entry_date: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    pattern_data: Dict[str, Any]
    signal_type: str = 'BUY'
    
    def __repr__(self):
        return (f"TradeSignal({self.symbol}, entry={self.entry_date.strftime('%Y-%m-%d')}, "
                f"price=${self.entry_price:.2f}, SL=${self.stop_loss:.2f}, TP=${self.take_profit:.2f})")


def calculate_stop_loss(pattern: Dict, buffer: float = 0.02) -> float:
    """
    Calculate stop loss level for a pattern.
    
    Stop loss is set at the support level (average of two bottoms) minus a buffer.
    
    Args:
        pattern: Pattern dictionary from detection
        buffer: Buffer percentage below support (default 2%)
        
    Returns:
        Stop loss price level
    """
    support_level = (pattern['bottom1_price'] + pattern['bottom2_price']) / 2
    stop_loss = support_level * (1 - buffer)
    return round(stop_loss, 2)


def calculate_take_profit(pattern: Dict) -> float:
    """
    Calculate take profit level for a pattern.
    
    Target price is neckline + pattern height (measured move).
    
    Args:
        pattern: Pattern dictionary from detection
        
    Returns:
        Take profit price level
    """
    return pattern['target_price']


def find_breakout_bar(df: pd.DataFrame, pattern: Dict, max_lookahead: int = 60) -> Optional[int]:
    """
    Find the bar where price closes above the neckline.
    
    Args:
        df: Price data DataFrame
        pattern: Pattern dictionary
        max_lookahead: Maximum bars to look ahead from second bottom
        
    Returns:
        Index of breakout bar or None if not found
    """
    neckline = pattern['neckline']
    start_idx = pattern['bottom2_idx'] + 1
    end_idx = min(start_idx + max_lookahead, len(df))
    
    for idx in range(start_idx, end_idx):
        if df['Close'].iloc[idx] > neckline:
            return idx
    
    return None


def generate_signals(df: pd.DataFrame, patterns: List[Dict], 
                    stop_loss_buffer: float = 0.02) -> List[TradeSignal]:
    """
    Generate trading signals from detected patterns.
    
    Entry rule: Buy at next session open after close above neckline.
    
    Args:
        df: Price data DataFrame with OHLCV
        patterns: List of detected pattern dictionaries
        stop_loss_buffer: Buffer percentage for stop loss (default 2%)
        
    Returns:
        List of TradeSignal objects
    """
    signals = []
    
    for pattern in patterns:
        breakout_idx = find_breakout_bar(df, pattern)
        
        if breakout_idx is None:
            continue
        
        entry_idx = breakout_idx + 1
        if entry_idx >= len(df):
            continue
        
        entry_date = df.index[entry_idx]
        entry_price = df['Open'].iloc[entry_idx]
        
        stop_loss = calculate_stop_loss(pattern, stop_loss_buffer)
        take_profit = calculate_take_profit(pattern)
        
        if entry_price <= stop_loss:
            continue
        
        if take_profit <= entry_price:
            continue
        
        signal = TradeSignal(
            symbol=pattern.get('symbol', 'UNKNOWN'),
            entry_date=entry_date,
            entry_price=round(entry_price, 2),
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_data=pattern
        )
        
        signals.append(signal)
    
    return signals


def generate_signals_from_scan_results(results_df: pd.DataFrame, 
                                       price_data: Dict[str, pd.DataFrame],
                                       stop_loss_buffer: float = 0.02) -> List[TradeSignal]:
    """
    Generate signals from scan results DataFrame.
    
    Args:
        results_df: DataFrame with scan results (from scan_stocks)
        price_data: Dictionary mapping symbols to their price DataFrames
        stop_loss_buffer: Buffer percentage for stop loss
        
    Returns:
        List of TradeSignal objects
    """
    all_signals = []
    
    for _, row in results_df.iterrows():
        symbol = row['symbol']
        
        if symbol not in price_data:
            continue
        
        df = price_data[symbol]
        
        pattern = row.to_dict()
        
        signals = generate_signals(df, [pattern], stop_loss_buffer)
        all_signals.extend(signals)
    
    return all_signals
