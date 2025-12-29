#!/usr/bin/env python3
"""
Pattern Quality Scoring Module for Double Bottom Scanner.

Computes quality scores for detected double bottom patterns using
only pre-entry data (no lookahead). Higher scores indicate higher
probability of profitable trades.

Features:
- Symmetry: How closely bottoms match in price
- Neckline strength: Height of neckline above bottoms
- Separation quality: Time between bottoms (prefer 30-120 days)
- Breakout strength: Candle quality on breakout (CONFIRMED only)
- Volume signature: Volume patterns at key points
- Trend context: Position relative to long-term trend

Version: 1.0.0
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np


DEFAULT_WEIGHTS = {
    'symmetry': 0.20,
    'neckline': 0.20,
    'separation': 0.15,
    'breakout_strength': 0.15,
    'volume': 0.20,
    'trend_context': 0.10,
}


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to [min_val, max_val] range."""
    return max(min_val, min(max_val, value))


def compute_symmetry_score(
    bottom1_price: float,
    bottom2_price: float,
    price_tolerance: float = 0.04
) -> float:
    """
    Compute symmetry score based on how closely bottoms match.
    
    Higher score means bottoms are more similar in price.
    Score of 1.0 means identical bottoms.
    
    Args:
        bottom1_price: Price at first bottom
        bottom2_price: Price at second bottom
        price_tolerance: Maximum allowed price difference (from config)
    
    Returns:
        Score between 0 and 1
    """
    avg_bottom = (bottom1_price + bottom2_price) / 2
    if avg_bottom <= 0:
        return 0.5
    
    bottom_diff_pct = abs(bottom1_price - bottom2_price) / avg_bottom
    symmetry_score = clamp(1 - bottom_diff_pct / price_tolerance, 0, 1)
    
    return symmetry_score


def compute_neckline_score(
    neckline: float,
    avg_bottom: float,
    min_peak_height: float = 0.06
) -> float:
    """
    Compute neckline strength score based on height above bottoms.
    
    Higher neckline relative to bottoms indicates stronger pattern.
    
    Args:
        neckline: Price at neckline (peak between bottoms)
        avg_bottom: Average of bottom1 and bottom2 prices
        min_peak_height: Minimum required peak height (from config)
    
    Returns:
        Score between 0 and 1
    """
    if avg_bottom <= 0:
        return 0.5
    
    neckline_rise_pct = (neckline - avg_bottom) / avg_bottom
    neckline_score = clamp((neckline_rise_pct - min_peak_height) / (2 * min_peak_height), 0, 1)
    
    return neckline_score


def compute_separation_score(separation_days: int) -> float:
    """
    Compute separation quality score based on time between bottoms.
    
    Prefer 30-120 days; penalize too short or too long.
    Optimal around 75 days.
    
    Args:
        separation_days: Number of days between bottom1 and bottom2
    
    Returns:
        Score between 0 and 1
    """
    optimal_days = 75
    max_deviation = 75
    
    deviation = abs(separation_days - optimal_days)
    sep_score = clamp(1 - deviation / max_deviation, 0, 1)
    
    return sep_score


def compute_breakout_strength_score(
    df: pd.DataFrame,
    breakout_date: Optional[pd.Timestamp],
    entry_kind: str,
    neckline: float = 0.0
) -> float:
    """
    Compute breakout candle strength (CONFIRMED patterns only).
    
    Measures the quality of the breakout candle with ATR normalization:
    - Body size as percentage of price (normalized)
    - Margin above neckline
    - ATR penalty to reduce "blow-off" candle bias
    - Green vs red candle multiplier
    
    For FORMING patterns, returns baseline 0.5.
    
    Args:
        df: Price DataFrame with OHLCV
        breakout_date: Date of neckline breakout (for CONFIRMED)
        entry_kind: 'CONFIRMED' or 'FORMING'
        neckline: Neckline price level for margin calculation
    
    Returns:
        Score between 0 and 1
    """
    if entry_kind != 'CONFIRMED' or breakout_date is None:
        return 0.5
    
    try:
        if hasattr(breakout_date, 'date'):
            breakout_dt = pd.Timestamp(breakout_date).date()
        else:
            breakout_dt = pd.Timestamp(breakout_date).date()
        
        breakout_idx = None
        for i, idx in enumerate(df.index):
            if pd.Timestamp(idx).date() == breakout_dt:
                breakout_idx = i
                break
        
        if breakout_idx is None:
            return 0.5
        
        open_price = df['Open'].iloc[breakout_idx]
        close_price = df['Close'].iloc[breakout_idx]
        high_price = df['High'].iloc[breakout_idx]
        low_price = df['Low'].iloc[breakout_idx]
        
        if open_price <= 0 or close_price <= 0:
            return 0.5
        
        body_pct = abs(close_price - open_price) / open_price
        body_score = clamp(body_pct / 0.02, 0, 1)
        
        margin_score = 0.5
        if neckline > 0:
            margin_pct = (close_price - neckline) / neckline
            margin_score = clamp(margin_pct / 0.01, 0, 1)
        
        atr_penalty = 0.0
        if breakout_idx >= 15:
            tr_values = []
            for i in range(breakout_idx - 14, breakout_idx):
                h = df['High'].iloc[i]
                l = df['Low'].iloc[i]
                c_prev = df['Close'].iloc[i - 1] if i > 0 else l
                tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
                tr_values.append(tr)
            
            if tr_values:
                atr14 = np.mean(tr_values)
                atr_pct = atr14 / close_price if close_price > 0 else 0
                atr_penalty = clamp(atr_pct / 0.05, 0, 1)
        
        is_green = close_price > open_price
        color_mult = 1.0 if is_green else 0.7
        
        breakout_score = (0.5 * body_score + 0.5 * margin_score) * (1 - 0.3 * atr_penalty)
        breakout_score *= color_mult
        
        return clamp(breakout_score, 0, 1)
        
    except Exception:
        return 0.5


def compute_volume_score(
    df: pd.DataFrame,
    bottom1_idx: Optional[int],
    bottom2_idx: Optional[int],
    breakout_date: Optional[pd.Timestamp],
    entry_kind: str
) -> float:
    """
    Compute volume signature score with log transform.
    
    For CONFIRMED: Check if breakout volume exceeds 20-day average
        Uses log transform to reduce extreme spike influence
    For FORMING: Check if volume dried up at bottom2 vs bottom1
        Uses cap to reduce extreme ratios
    
    Args:
        df: Price DataFrame with OHLCV
        bottom1_idx: Index of first bottom in df
        bottom2_idx: Index of second bottom in df
        breakout_date: Date of breakout (CONFIRMED only)
        entry_kind: 'CONFIRMED' or 'FORMING'
    
    Returns:
        Score between 0 and 1
    """
    try:
        if entry_kind == 'CONFIRMED' and breakout_date is not None:
            if hasattr(breakout_date, 'date'):
                breakout_dt = pd.Timestamp(breakout_date).date()
            else:
                breakout_dt = pd.Timestamp(breakout_date).date()
            
            breakout_idx = None
            for i, idx in enumerate(df.index):
                if pd.Timestamp(idx).date() == breakout_dt:
                    breakout_idx = i
                    break
            
            if breakout_idx is None or breakout_idx < 20:
                return 0.5
            
            vol_ma20 = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
            if vol_ma20 <= 0:
                return 0.5
            
            breakout_vol = df['Volume'].iloc[breakout_idx]
            vol_ratio = breakout_vol / vol_ma20
            
            if vol_ratio <= 0:
                return 0.0
            
            log_score = (np.log(vol_ratio) - np.log(1.0)) / np.log(2.0)
            vol_score = clamp(log_score, 0, 1)
            return vol_score
        
        else:
            if bottom1_idx is None or bottom2_idx is None:
                return 0.5
            
            if bottom1_idx >= len(df) or bottom2_idx >= len(df):
                return 0.5
            
            vol_bottom1 = df['Volume'].iloc[bottom1_idx]
            vol_bottom2 = df['Volume'].iloc[bottom2_idx]
            
            if vol_bottom1 <= 0:
                return 0.5
            
            vol_ratio_bottoms = min(vol_bottom2 / vol_bottom1, 2.0)
            vol_score = clamp((1.1 - vol_ratio_bottoms) / 0.6, 0, 1)
            
            return vol_score
            
    except Exception:
        return 0.5


def compute_trend_context_score(
    df: pd.DataFrame,
    entry_date: pd.Timestamp,
    ma_period: int = 200,
    trend_mode: str = "NEUTRAL"
) -> float:
    """
    Compute trend context score based on position vs long-term MA.
    
    Uses data up to entry_date - 1 (no lookahead).
    
    Args:
        df: Price DataFrame with OHLCV
        entry_date: Date of signal entry
        ma_period: Moving average period (default 200)
        trend_mode: Scoring mode:
            - ABOVE_MA200: 1.0 if above MA200, 0.5 if below (original)
            - BELOW_MA200: 1.0 if below MA200, 0.5 if above (inverted)
            - NEUTRAL: always 0.5 (removes trend influence)
    
    Returns:
        Score between 0.5 and 1.0
    """
    if trend_mode == "NEUTRAL":
        return 0.5
    
    try:
        entry_dt = pd.Timestamp(entry_date).date()
        
        pre_entry_idx = None
        for i, idx in enumerate(df.index):
            if pd.Timestamp(idx).date() >= entry_dt:
                pre_entry_idx = i - 1
                break
        
        if pre_entry_idx is None or pre_entry_idx < ma_period:
            return 0.5
        
        ma200 = df['Close'].iloc[pre_entry_idx-ma_period+1:pre_entry_idx+1].mean()
        close_pre = df['Close'].iloc[pre_entry_idx]
        
        if trend_mode == "ABOVE_MA200":
            return 1.0 if close_pre > ma200 else 0.5
        elif trend_mode == "BELOW_MA200":
            return 1.0 if close_pre < ma200 else 0.5
        else:
            return 0.5
            
    except Exception:
        return 0.5


def compute_pattern_quality_score(
    df: pd.DataFrame,
    pattern: Dict[str, Any],
    entry_date: pd.Timestamp,
    entry_kind: str,
    weights: Optional[Dict[str, float]] = None,
    price_tolerance: float = 0.04,
    min_peak_height: float = 0.06,
    score_policy: str = "RAW",
    trend_mode: str = "NEUTRAL"
) -> Tuple[float, Dict[str, float]]:
    """
    Compute comprehensive pattern quality score.
    
    Combines all feature scores using weighted average.
    Uses only data available before entry (no lookahead).
    
    Args:
        df: Price DataFrame with OHLCV
        pattern: Pattern dictionary from scanner
        entry_date: Date of signal entry
        entry_kind: 'CONFIRMED' or 'FORMING'
        weights: Feature weights (defaults to DEFAULT_WEIGHTS)
        price_tolerance: Config value for symmetry scoring
        min_peak_height: Config value for neckline scoring
        score_policy: 'RAW' (default) or 'INVERT' (100 - score)
        trend_mode: 'NEUTRAL', 'ABOVE_MA200', or 'BELOW_MA200'
    
    Returns:
        Tuple of (pattern_score 0-100, feature_dict)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    bottom1_price = pattern.get('bottom1_price', 0)
    bottom2_price = pattern.get('bottom2_price', 0)
    neckline = pattern.get('neckline', 0)
    avg_bottom = pattern.get('avg_bottom') or ((bottom1_price + bottom2_price) / 2)
    separation_days = pattern.get('separation_days', 75)
    breakout_date = pattern.get('breakout_date')
    
    bottom1_idx = pattern.get('bottom1_idx')
    bottom2_idx = pattern.get('bottom2_idx')
    
    features = {}
    
    features['symmetry'] = compute_symmetry_score(
        bottom1_price, bottom2_price, price_tolerance
    )
    
    features['neckline'] = compute_neckline_score(
        neckline, avg_bottom, min_peak_height
    )
    
    features['separation'] = compute_separation_score(separation_days)
    
    features['breakout_strength'] = compute_breakout_strength_score(
        df, breakout_date, entry_kind, neckline=neckline
    )
    
    features['volume'] = compute_volume_score(
        df, bottom1_idx, bottom2_idx, breakout_date, entry_kind
    )
    
    features['trend_context'] = compute_trend_context_score(
        df, entry_date, ma_period=200, trend_mode=trend_mode
    )
    
    score_0_1 = sum(weights[k] * features[k] for k in weights.keys())
    pattern_score = round(100 * score_0_1, 2)
    
    if score_policy == "INVERT":
        pattern_score = round(100 - pattern_score, 2)
    
    features_rounded = {k: round(v, 4) for k, v in features.items()}
    
    return pattern_score, features_rounded


def filter_signals_by_score(
    signals: List[Any],
    min_score: float = 0.0,
    top_k_per_day: int = 0,
    top_k_per_week: int = 0
) -> List[Any]:
    """
    Filter signals based on pattern quality score.
    
    Applies:
    1. Minimum score threshold
    2. Top-K per day selection
    3. Top-K per week selection (alternative)
    
    Tie-breaking:
    1. CONFIRMED over FORMING
    2. Higher neckline_rise_pct (via meta)
    
    Args:
        signals: List of TradeSignal objects with pattern_score in meta
        min_score: Minimum pattern_score to keep (0 = no filter)
        top_k_per_day: Keep only top K signals per day (0 = disabled)
        top_k_per_week: Keep only top K signals per week (0 = disabled)
    
    Returns:
        Filtered list of signals
    """
    if not signals:
        return signals
    
    filtered = signals
    
    if min_score > 0:
        filtered = [
            s for s in filtered 
            if s.meta.get('pattern_score', 0) >= min_score
        ]
    
    if top_k_per_day > 0:
        filtered = _apply_top_k(filtered, top_k_per_day, group_by='day')
    
    if top_k_per_week > 0:
        filtered = _apply_top_k(filtered, top_k_per_week, group_by='week')
    
    return filtered


def _apply_top_k(
    signals: List[Any],
    k: int,
    group_by: str = 'day'
) -> List[Any]:
    """Apply top-K selection within time groups."""
    if not signals:
        return signals
    
    from collections import defaultdict
    groups = defaultdict(list)
    
    for s in signals:
        if group_by == 'day':
            key = pd.Timestamp(s.entry_date).date()
        else:
            ts = pd.Timestamp(s.entry_date)
            key = (ts.year, ts.isocalendar().week)
        groups[key].append(s)
    
    result = []
    for key in sorted(groups.keys()):
        group = groups[key]
        
        def sort_key(sig):
            score = sig.meta.get('pattern_score', 0)
            is_confirmed = 1 if sig.entry_kind == 'CONFIRMED' else 0
            neckline_pct = sig.meta.get('neckline_rise_pct', 0) or 0
            return (-score, -is_confirmed, -neckline_pct)
        
        sorted_group = sorted(group, key=sort_key)
        result.extend(sorted_group[:k])
    
    return result


def compute_score_bucket_stats(
    trades_df: pd.DataFrame,
    score_col: str = 'pattern_score',
    buckets: int = 10
) -> pd.DataFrame:
    """
    Compute performance statistics by score bucket (decile).
    
    Args:
        trades_df: DataFrame with trade results
        score_col: Column name for pattern score
        buckets: Number of buckets (default 10 for deciles)
    
    Returns:
        DataFrame with bucket stats
    """
    if trades_df.empty or score_col not in trades_df.columns:
        return pd.DataFrame()
    
    df = trades_df.copy()
    df = df.dropna(subset=[score_col])
    
    if df.empty:
        return pd.DataFrame()
    
    try:
        df['score_bucket'] = pd.qcut(
            df[score_col], 
            buckets, 
            labels=[f'{i*10}-{(i+1)*10}%' for i in range(buckets)],
            duplicates='drop'
        )
    except ValueError:
        df['score_bucket'] = pd.cut(
            df[score_col],
            bins=buckets,
            labels=[f'B{i+1}' for i in range(buckets)]
        )
    
    stats = []
    for bucket in df['score_bucket'].dropna().unique():
        bucket_df = df[df['score_bucket'] == bucket]
        
        n_trades = len(bucket_df)
        if n_trades == 0:
            continue
        
        pnl_col = 'pnl' if 'pnl' in bucket_df.columns else 'pnl_dollars'
        r_col = 'pnl_r_multiple' if 'pnl_r_multiple' in bucket_df.columns else None
        
        wins = bucket_df[bucket_df[pnl_col] > 0] if pnl_col in bucket_df.columns else pd.DataFrame()
        losses = bucket_df[bucket_df[pnl_col] <= 0] if pnl_col in bucket_df.columns else pd.DataFrame()
        
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
        
        total_pnl = bucket_df[pnl_col].sum() if pnl_col in bucket_df.columns else 0
        
        avg_r = bucket_df[r_col].mean() if r_col and r_col in bucket_df.columns else 0
        
        gross_profit = wins[pnl_col].sum() if not wins.empty and pnl_col in wins.columns else 0
        gross_loss = abs(losses[pnl_col].sum()) if not losses.empty and pnl_col in losses.columns else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
        
        stats.append({
            'bucket': bucket,
            'min_score': bucket_df[score_col].min(),
            'max_score': bucket_df[score_col].max(),
            'trade_count': n_trades,
            'win_rate': round(win_rate, 1),
            'avg_r': round(avg_r, 3) if avg_r else 0,
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2)
        })
    
    result = pd.DataFrame(stats)
    if not result.empty:
        result = result.sort_values('min_score')
    
    return result


def compute_threshold_comparison(
    trades_df: pd.DataFrame,
    thresholds: List[float] = [60, 70, 80],
    score_col: str = 'pattern_score'
) -> pd.DataFrame:
    """
    Compare performance at different score thresholds.
    
    Args:
        trades_df: DataFrame with trade results
        thresholds: Score thresholds to compare
        score_col: Column name for pattern score
    
    Returns:
        DataFrame with threshold comparison
    """
    if trades_df.empty or score_col not in trades_df.columns:
        return pd.DataFrame()
    
    results = []
    
    for thresh in [0] + thresholds:
        df = trades_df[trades_df[score_col] >= thresh] if thresh > 0 else trades_df
        
        n_trades = len(df)
        if n_trades == 0:
            continue
        
        pnl_col = 'pnl' if 'pnl' in df.columns else 'pnl_dollars'
        r_col = 'pnl_r_multiple' if 'pnl_r_multiple' in df.columns else None
        
        wins = df[df[pnl_col] > 0] if pnl_col in df.columns else pd.DataFrame()
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
        
        total_pnl = df[pnl_col].sum() if pnl_col in df.columns else 0
        avg_r = df[r_col].mean() if r_col and r_col in df.columns else 0
        
        results.append({
            'min_score_threshold': thresh,
            'trade_count': n_trades,
            'win_rate': round(win_rate, 1),
            'avg_r': round(avg_r, 3) if avg_r else 0,
            'total_pnl': round(total_pnl, 2)
        })
    
    return pd.DataFrame(results)
