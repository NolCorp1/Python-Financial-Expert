#!/usr/bin/env python3
"""
Trading Strategy Module for Double Bottom Pattern Scanner.

This module provides signal generation based on detected double bottom patterns.
Supports both CONFIRMED (neckline breakout) and FORMING (early trigger) entries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class TradeSignal:
    """Represents a trading signal from a detected pattern."""
    symbol: str
    pattern_id: str
    entry_date: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_kind: str
    trigger_level: float
    risk_per_share: float
    meta: dict = field(default_factory=dict)
    signal_type: str = 'BUY'
    
    def __repr__(self):
        return (f"TradeSignal({self.symbol}, {self.entry_kind}, "
                f"entry={self.entry_date.strftime('%Y-%m-%d')}, "
                f"price=${self.entry_price:.2f}, SL=${self.stop_loss:.2f}, "
                f"TP=${self.take_profit:.2f})")


def _find_date_index(df: pd.DataFrame, target_date) -> Optional[int]:
    """Find the index in df corresponding to target_date."""
    if target_date is None or pd.isna(target_date):
        return None
    
    if hasattr(target_date, 'tz_localize'):
        target_date = pd.Timestamp(target_date)
    
    for i, idx in enumerate(df.index):
        if pd.Timestamp(idx).date() == pd.Timestamp(target_date).date():
            return i
    return None


def generate_signals(
    df: pd.DataFrame,
    patterns: List[Dict],
    stop_loss_buffer: float = 0.02,
    forming_lookahead_days: int = 20,
    forming_requires_green: bool = True,
) -> List[TradeSignal]:
    """
    Generate trading signals from detected patterns.
    
    Supports two entry modes:
    - CONFIRMED: Enter at next bar open after neckline breakout close
    - FORMING: Enter at next bar open after trigger level breakout
    
    Args:
        df: Price data DataFrame with OHLCV
        patterns: List of detected pattern dictionaries
        stop_loss_buffer: Buffer percentage below support (default 2%)
        forming_lookahead_days: Max bars to search for forming trigger
        forming_requires_green: Require green candle for forming trigger
        
    Returns:
        List of TradeSignal objects
    """
    signals = []
    seen_pattern_ids = set()
    
    confirmed_patterns = [p for p in patterns if p.get('status') == 'CONFIRMED']
    forming_patterns = [p for p in patterns if p.get('status') == 'FORMING']
    
    for pattern in confirmed_patterns:
        signal = _generate_confirmed_signal(df, pattern, stop_loss_buffer)
        if signal and signal.pattern_id not in seen_pattern_ids:
            signals.append(signal)
            seen_pattern_ids.add(signal.pattern_id)
    
    for pattern in forming_patterns:
        pattern_id = pattern.get('pattern_id', '')
        if pattern_id in seen_pattern_ids:
            continue
        
        signal = _generate_forming_signal(
            df, pattern, stop_loss_buffer, 
            forming_lookahead_days, forming_requires_green
        )
        if signal:
            signals.append(signal)
            seen_pattern_ids.add(signal.pattern_id)
    
    return signals


def _generate_confirmed_signal(
    df: pd.DataFrame,
    pattern: Dict,
    stop_loss_buffer: float
) -> Optional[TradeSignal]:
    """Generate signal for a CONFIRMED pattern."""
    breakout_date = pattern.get('breakout_date')
    if breakout_date is None or pd.isna(breakout_date):
        return None
    
    breakout_idx = _find_date_index(df, breakout_date)
    if breakout_idx is None:
        return None
    
    entry_idx = breakout_idx + 1
    if entry_idx >= len(df):
        return None
    
    entry_date = df.index[entry_idx]
    entry_price = float(df['Open'].iloc[entry_idx])
    
    bottom1_price = pattern['bottom1_price']
    bottom2_price = pattern['bottom2_price']
    stop_loss = min(bottom1_price, bottom2_price) * (1 - stop_loss_buffer)
    take_profit = pattern['target_price']
    
    if entry_price <= stop_loss:
        return None
    if take_profit <= entry_price:
        return None
    
    risk_per_share = entry_price - stop_loss
    
    meta = {
        'avg_bottom': pattern.get('avg_bottom'),
        'height': pattern.get('height'),
        'separation_days': pattern.get('separation_days'),
        'pattern_score': pattern.get('score') or pattern.get('strength_score'),
        'pattern_status': pattern.get('status'),
    }
    
    return TradeSignal(
        symbol=pattern.get('symbol', 'UNKNOWN'),
        pattern_id=pattern.get('pattern_id', ''),
        entry_date=entry_date,
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        entry_kind='CONFIRMED',
        trigger_level=pattern['neckline'],
        risk_per_share=round(risk_per_share, 2),
        meta=meta
    )


def _generate_forming_signal(
    df: pd.DataFrame,
    pattern: Dict,
    stop_loss_buffer: float,
    forming_lookahead_days: int,
    forming_requires_green: bool
) -> Optional[TradeSignal]:
    """Generate signal for a FORMING pattern."""
    trigger_level = pattern.get('forming_trigger_level')
    if trigger_level is None or pd.isna(trigger_level):
        return None
    
    bottom2_date = pattern.get('bottom2_date')
    bottom2_idx = _find_date_index(df, bottom2_date)
    
    if bottom2_idx is None:
        bottom2_idx = pattern.get('bottom2_idx')
        if bottom2_idx is None:
            return None
    
    start_idx = bottom2_idx + 1
    end_idx = min(bottom2_idx + forming_lookahead_days + 1, len(df))
    
    trigger_idx = None
    for idx in range(start_idx, end_idx):
        close_price = df['Close'].iloc[idx]
        open_price = df['Open'].iloc[idx]
        
        if close_price > trigger_level:
            if forming_requires_green:
                if close_price > open_price:
                    trigger_idx = idx
                    break
            else:
                trigger_idx = idx
                break
    
    if trigger_idx is None:
        return None
    
    entry_idx = trigger_idx + 1
    if entry_idx >= len(df):
        return None
    
    entry_date = df.index[entry_idx]
    entry_price = float(df['Open'].iloc[entry_idx])
    
    bottom1_price = pattern['bottom1_price']
    bottom2_price = pattern['bottom2_price']
    stop_loss = min(bottom1_price, bottom2_price) * (1 - stop_loss_buffer)
    take_profit = pattern['target_price']
    
    if entry_price <= stop_loss:
        return None
    if take_profit <= entry_price:
        return None
    
    risk_per_share = entry_price - stop_loss
    
    meta = {
        'avg_bottom': pattern.get('avg_bottom'),
        'height': pattern.get('height'),
        'separation_days': pattern.get('separation_days'),
        'pattern_score': pattern.get('score') or pattern.get('strength_score'),
        'pattern_status': pattern.get('status'),
    }
    
    return TradeSignal(
        symbol=pattern.get('symbol', 'UNKNOWN'),
        pattern_id=pattern.get('pattern_id', ''),
        entry_date=entry_date,
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        entry_kind='FORMING',
        trigger_level=round(trigger_level, 2),
        risk_per_share=round(risk_per_share, 2),
        meta=meta
    )


def generate_signals_from_scan_results(
    results_df: pd.DataFrame, 
    price_data: Dict[str, pd.DataFrame],
    stop_loss_buffer: float = 0.02,
    forming_lookahead_days: int = 20,
    forming_requires_green: bool = True
) -> List[TradeSignal]:
    """
    Generate signals from scan results DataFrame.
    
    Args:
        results_df: DataFrame with scan results (from scan_stocks)
        price_data: Dictionary mapping symbols to their price DataFrames
        stop_loss_buffer: Buffer percentage for stop loss
        forming_lookahead_days: Max bars to search for forming trigger
        forming_requires_green: Require green candle for forming trigger
        
    Returns:
        List of TradeSignal objects
    """
    all_signals = []
    
    for symbol in results_df['symbol'].unique():
        if symbol not in price_data:
            continue
        
        df = price_data[symbol]
        symbol_patterns = results_df[results_df['symbol'] == symbol].to_dict('records')
        
        signals = generate_signals(
            df, symbol_patterns, 
            stop_loss_buffer=stop_loss_buffer,
            forming_lookahead_days=forming_lookahead_days,
            forming_requires_green=forming_requires_green
        )
        all_signals.extend(signals)
    
    return all_signals
