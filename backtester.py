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


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    else:
        return obj


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
    
    risk_fraction_confirmed: float = 0.010
    risk_fraction_forming: float = 0.006
    
    max_positions_total: int = 10
    max_positions_forming: int = 3
    max_positions_confirmed: int = 10
    
    forming_max_hold_days: int = 60
    forming_no_progress_days: int = 20
    forming_no_progress_r: float = 0.50
    forming_no_progress_action: str = "EXIT"
    forming_tighten_stop_to_r: float = -0.25
    
    confirmed_move_stop_to_be_at_r: float = 0.50
    
    confirmed_partial_tp_enabled: bool = True
    confirmed_partial_tp_at_r: float = 1.00
    confirmed_partial_tp_fraction: float = 0.50
    
    confirmed_trailing_enabled: bool = True
    confirmed_trailing_start_r: float = 1.00
    confirmed_trailing_atr_mult: float = 2.0
    atr_length: int = 14
    
    use_correlation_caps: bool = True
    corr_lookback_days: int = 60
    max_corr_to_existing: float = 0.80
    use_cluster_caps: bool = True
    n_clusters: int = 8
    max_positions_per_cluster: int = 2
    
    use_regime_filter: bool = True
    regime_symbol: str = "QQQ"
    regime_trend_fast_ma: int = 50
    regime_trend_slow_ma: int = 200
    regime_vol_lookback: int = 20
    regime_vol_high_threshold: float = 0.03
    regime_soft_gate: bool = True
    regime_downtrend_forming_risk_mult: float = 0.85
    regime_highvol_forming_risk_mult: float = 0.70
    regime_highvol_confirmed_risk_mult: float = 0.85
    
    use_score_risk_scaling: bool = True
    score_risk_alpha: float = 1.0
    score_risk_min_mult: float = 0.60
    score_risk_max_mult: float = 1.20
    score_risk_apply_to: str = "BOTH"
    score_risk_missing_policy: str = "NEUTRAL"
    max_risk_fraction_per_trade: float = 0.02
    
    # Portfolio capital allocation (Task 20)
    use_portfolio_allocator: bool = True
    daily_risk_budget: float = 0.04
    weekly_risk_budget: Optional[float] = None
    daily_risk_budget_forming: Optional[float] = 0.015
    daily_risk_budget_confirmed: Optional[float] = None
    allocation_rank_metric: str = "score_weighted"
    max_signals_per_day: Optional[int] = None
    allocation_scaling_mode: str = "PROPORTIONAL"
    min_allocation_scale: float = 0.25


def compute_signal_allocation_score(signal: 'TradeSignal') -> float:
    """
    Compute ranking score for capital allocation.
    Uses only pre-entry data (no lookahead).
    Higher score = higher priority for capital.
    """
    score = 1.0

    if signal.meta:
        ps = signal.meta.get("pattern_score")
        sr = signal.meta.get("score_risk_mult", 1.0)
        rr = signal.meta.get("regime_risk_mult", 1.0)

        if ps is not None:
            try:
                score *= float(ps) / 100.0
            except (ValueError, TypeError):
                pass

        try:
            score *= float(sr)
        except (ValueError, TypeError):
            pass

        try:
            score *= float(rr)
        except (ValueError, TypeError):
            pass

    # Prefer CONFIRMED slightly by default
    if getattr(signal, 'entry_kind', '') == "CONFIRMED":
        score *= 1.10

    return float(max(0.0, score))


def compute_score_risk_mult(
    pattern_score: Optional[float],
    alpha: float,
    min_mult: float,
    max_mult: float,
    missing_policy: str = "NEUTRAL",
) -> float:
    """
    Convert pattern_score [0,100] into a smooth risk multiplier in [min_mult, max_mult].
    No-lookahead safe: uses only pre-entry score.
    """
    if pattern_score is None:
        return 1.0 if missing_policy.upper() == "NEUTRAL" else float(min_mult)

    try:
        s = float(pattern_score)
    except Exception:
        return 1.0 if missing_policy.upper() == "NEUTRAL" else float(min_mult)

    s = max(0.0, min(100.0, s)) / 100.0
    shaped = s ** float(alpha)
    return float(min_mult) + (float(max_mult) - float(min_mult)) * shaped


@dataclass
class AllocationCandidate:
    """
    Represents a signal candidate for capital allocation.
    Holds pre-computed data for ranking and sizing.
    """
    signal: 'TradeSignal'
    entry_fill: float
    stop_dist: float
    risk_fraction: float  # base risk fraction after regime/score adjustments
    allocation_score: float
    regime_risk_mult: float
    score_risk_mult: float
    allocation_scale: float = 1.0  # set by allocator
    allocation_rank: int = 0  # set by allocator
    allocation_budget_used: float = 0.0  # set by allocator


def allocate_daily_signals(
    candidates: List[AllocationCandidate],
    current_equity: float,
    cfg: BacktestConfig,
    weekly_risk_used: float = 0.0
) -> List[AllocationCandidate]:
    """
    Apply portfolio capital allocation constraints to daily signal candidates.
    Ranks candidates and allocates capital subject to daily and weekly budgets.
    
    Uses greedy allocation: add candidates in priority order until budget exhausted.
    This ensures total allocated risk never exceeds configured budgets.
    
    No-lookahead safe: uses only pre-computed allocation_score.
    
    Args:
        candidates: List of AllocationCandidate objects
        current_equity: Current portfolio equity
        cfg: BacktestConfig with budget parameters
        weekly_risk_used: Risk already used in rolling weekly window (for weekly cap)
    
    Returns:
        Filtered and scaled list of candidates with allocation_scale populated.
    """
    if not candidates:
        return []
    
    # Sort by allocation_score descending (highest priority first)
    candidates.sort(key=lambda c: c.allocation_score, reverse=True)
    
    # Apply max_signals_per_day soft cap
    if cfg.max_signals_per_day is not None and len(candidates) > cfg.max_signals_per_day:
        candidates = candidates[:cfg.max_signals_per_day]
    
    # Assign ranks
    for rank, cand in enumerate(candidates, start=1):
        cand.allocation_rank = rank
    
    # Budget limits
    daily_budget = cfg.daily_risk_budget
    forming_budget = cfg.daily_risk_budget_forming if cfg.daily_risk_budget_forming is not None else daily_budget
    confirmed_budget = cfg.daily_risk_budget_confirmed if cfg.daily_risk_budget_confirmed is not None else daily_budget
    weekly_budget = cfg.weekly_risk_budget if cfg.weekly_risk_budget is not None else float('inf')
    
    # Track remaining budgets (greedy allocation)
    remaining_daily = daily_budget
    remaining_forming = forming_budget
    remaining_confirmed = confirmed_budget
    remaining_weekly = max(0.0, weekly_budget - weekly_risk_used)
    
    result = []
    for cand in candidates:
        requested = cand.risk_fraction
        is_forming = cand.signal.entry_kind == "FORMING"
        
        # Determine applicable budget caps
        if is_forming:
            kind_cap = min(remaining_daily, remaining_forming, remaining_weekly)
        else:
            kind_cap = min(remaining_daily, remaining_confirmed, remaining_weekly)
        
        # Skip if no budget remains
        if kind_cap <= 0:
            continue
        
        # Calculate scale needed to fit in budget
        if requested <= kind_cap:
            scale = 1.0
        else:
            scale = kind_cap / requested
        
        # Apply scaling mode
        if scale < cfg.min_allocation_scale:
            if cfg.allocation_scaling_mode.upper() == "HARD_CUTOFF":
                # Drop this candidate if it would scale below minimum
                continue
            else:
                # PROPORTIONAL mode: clamp to min_allocation_scale
                # Check if we have enough budget for the minimum allocation
                min_alloc = requested * cfg.min_allocation_scale
                if min_alloc > kind_cap:
                    continue  # Can't fit even minimum allocation
                scale = cfg.min_allocation_scale
        
        # Compute final allocated risk
        allocated = requested * scale
        
        # Update remaining budgets
        remaining_daily -= allocated
        remaining_weekly -= allocated
        if is_forming:
            remaining_forming -= allocated
        else:
            remaining_confirmed -= allocated
        
        cand.allocation_scale = scale
        cand.allocation_budget_used = allocated
        result.append(cand)
    
    return result


@dataclass
class OpenPosition:
    """Tracks an open position during backtest."""
    symbol: str
    pattern_id: str
    entry_kind: str
    entry_date: pd.Timestamp
    entry_fill: float
    stop_loss: float
    original_stop_loss: float
    take_profit: float
    shares: int
    entry_commission: float
    slippage_bps: float
    meta: dict = field(default_factory=dict)
    bars_held: int = 0
    mfe_r: float = 0.0
    tightened: bool = False
    moved_to_be: bool = False
    shares_initial: int = 0
    shares_remaining: int = 0
    partial_tp_done: bool = False
    partial_tp_date: Optional[pd.Timestamp] = None
    realized_pnl_dollars: float = 0.0
    trailing_active: bool = False


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


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) for a price DataFrame.
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        length: ATR period (default 14)
        
    Returns:
        Series with ATR values aligned to df index
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    
    return atr


def compute_regime(df_regime: pd.DataFrame, cfg: 'BacktestConfig') -> pd.DataFrame:
    """
    Compute market regime (trend + volatility) with no-lookahead (1-bar shift).
    
    Args:
        df_regime: OHLCV DataFrame for regime symbol (QQQ/SPY)
        cfg: BacktestConfig with regime parameters
        
    Returns:
        DataFrame with:
        - trend_signal: "UP" or "DOWN" (shifted by 1 bar)
        - atr_pct: ATR as percentage of close (shifted by 1 bar)
        - vol_signal: "HIGH" or "NORMAL" (shifted by 1 bar)
    """
    close = df_regime['Close']
    
    fast_ma = close.rolling(window=cfg.regime_trend_fast_ma, min_periods=1).mean()
    slow_ma = close.rolling(window=cfg.regime_trend_slow_ma, min_periods=1).mean()
    trend = (fast_ma > slow_ma).map({True: "UP", False: "DOWN"})
    
    atr = compute_atr(df_regime, length=cfg.regime_vol_lookback)
    atr_pct = atr / close
    vol = (atr_pct > cfg.regime_vol_high_threshold).map({True: "HIGH", False: "NORMAL"})
    
    regime_df = pd.DataFrame({
        'trend_signal': trend.shift(1),
        'atr_pct': atr_pct.shift(1),
        'vol_signal': vol.shift(1)
    }, index=df_regime.index)
    
    return regime_df


def compute_returns_matrix(price_data_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a returns matrix from price data.
    
    Args:
        price_data_by_symbol: Dict mapping symbol to OHLCV DataFrame
        
    Returns:
        DataFrame with dates as index, symbols as columns, daily returns as values
    """
    close_prices = {}
    for sym, df in price_data_by_symbol.items():
        if 'Close' in df.columns and len(df) > 0:
            close_prices[sym] = df['Close']
    
    if not close_prices:
        return pd.DataFrame()
    
    prices_df = pd.DataFrame(close_prices)
    returns_df = prices_df.pct_change().dropna(how='all')
    
    return returns_df


def compute_symbol_correlation_and_clusters(
    returns_df: pd.DataFrame,
    lookback_days: int,
    n_clusters: int
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute correlation matrix and assign symbols to clusters.
    
    Uses hierarchical clustering on correlation distance.
    Falls back to single cluster if clustering fails.
    
    Args:
        returns_df: Returns matrix (dates x symbols)
        lookback_days: Number of trailing days to use
        n_clusters: Target number of clusters
        
    Returns:
        corr_matrix: Symbol-to-symbol correlation DataFrame
        clusters: Dict mapping symbol -> cluster_id (0 to n_clusters-1)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    symbols = list(returns_df.columns)
    clusters = {s: 0 for s in symbols}
    
    tail_returns = returns_df.tail(lookback_days)
    min_periods = int(lookback_days * 0.7)
    corr_matrix = tail_returns.corr(min_periods=min_periods)
    
    corr_matrix = corr_matrix.fillna(0)
    
    valid_symbols = [s for s in symbols if s in corr_matrix.columns]
    if len(valid_symbols) < 2:
        return corr_matrix, clusters
    
    try:
        corr_sub = corr_matrix.loc[valid_symbols, valid_symbols]
        
        dist_matrix = 1 - corr_sub.values
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.clip(dist_matrix, 0, 2)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        
        condensed_dist = squareform(dist_matrix, checks=False)
        
        if len(condensed_dist) == 0:
            return corr_matrix, clusters
            
        linkage_matrix = linkage(condensed_dist, method='average')
        
        actual_n_clusters = min(n_clusters, len(valid_symbols))
        cluster_labels = fcluster(linkage_matrix, t=actual_n_clusters, criterion='maxclust')
        
        for i, sym in enumerate(valid_symbols):
            clusters[sym] = int(cluster_labels[i] - 1)
            
    except Exception as e:
        pass
    
    return corr_matrix, clusters


@dataclass
class BacktestContext:
    """
    Precomputed, window-invariant backtest artifacts.
    Safe to reuse across parameter combos as long as price_data_by_symbol 
    and cfg.atr_length are unchanged.
    """
    atr_by_symbol: Dict[str, pd.Series]
    returns_df: pd.DataFrame
    all_dates: List[pd.Timestamp]
    regime_df: Optional[pd.DataFrame] = None
    corr_cluster_cache: Dict[Tuple[pd.Timestamp, int, int], Tuple[pd.DataFrame, Dict[str, int]]] = field(default_factory=dict)


def build_backtest_context(
    price_data_by_symbol: Dict[str, pd.DataFrame],
    cfg: BacktestConfig
) -> BacktestContext:
    """
    Build precomputed backtest context for reuse across parameter combos.
    
    Precomputes:
    - ATR series for all symbols
    - Returns matrix for correlation/cluster computation
    - Master date calendar
    - Regime DataFrame (if enabled and regime symbol available)
    
    Args:
        price_data_by_symbol: Dict mapping symbol to OHLCV DataFrame
        cfg: BacktestConfig with parameters (atr_length, regime settings)
        
    Returns:
        BacktestContext with precomputed artifacts
    """
    atr_by_symbol: Dict[str, pd.Series] = {}
    for sym, df in price_data_by_symbol.items():
        atr_by_symbol[sym] = compute_atr(df, cfg.atr_length)
    
    returns_df = compute_returns_matrix(price_data_by_symbol)
    
    all_dates_set: set = set()
    for df in price_data_by_symbol.values():
        all_dates_set.update(df.index.tolist())
    all_dates = sorted(all_dates_set)
    
    regime_df = None
    if cfg.use_regime_filter and cfg.regime_symbol in price_data_by_symbol:
        df_regime = price_data_by_symbol[cfg.regime_symbol]
        if len(df_regime) > 0:
            regime_df = compute_regime(df_regime, cfg)
    
    return BacktestContext(
        atr_by_symbol=atr_by_symbol,
        returns_df=returns_df,
        all_dates=all_dates,
        regime_df=regime_df
    )


def run_backtest(
    signals_by_symbol: Dict[str, List[TradeSignal]],
    price_data_by_symbol: Dict[str, pd.DataFrame],
    cfg: BacktestConfig,
    return_diagnostics: bool = False,
    ctx: Optional['BacktestContext'] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run no-lookahead backtest simulation with portfolio rules engine.
    
    Args:
        signals_by_symbol: Dict mapping symbol to list of TradeSignals
        price_data_by_symbol: Dict mapping symbol to OHLCV DataFrame
        cfg: BacktestConfig with simulation parameters
        return_diagnostics: If True, return (trades_df, equity_df, diagnostics)
        ctx: Optional BacktestContext with precomputed artifacts for speedup
        
    Returns:
        trades_df: Trade blotter with all completed trades
        equity_df: Daily equity curve with drawdown
        diagnostics: (optional) Dict with skip counters, regime stats, effective risk
    """
    if ctx is None:
        ctx = build_backtest_context(price_data_by_symbol, cfg)
    
    atr_by_symbol = ctx.atr_by_symbol
    returns_df = ctx.returns_df
    all_dates = ctx.all_dates
    regime_df = ctx.regime_df if ctx.regime_df is not None else pd.DataFrame()
    
    cash = cfg.initial_capital
    open_positions: Dict[str, OpenPosition] = {}
    completed_trades: List[dict] = []
    equity_history: List[dict] = []
    
    exited_today: Dict[pd.Timestamp, set] = {}
    
    skipped_max_total = 0
    skipped_max_forming = 0
    skipped_max_confirmed = 0
    skipped_symbol_already_open = 0
    skipped_invalid_sizing = 0
    skipped_corr_cap = 0
    skipped_cluster_cap = 0
    reduced_regime_forming_downtrend = 0
    reduced_regime_forming_highvol = 0
    reduced_regime_confirmed_highvol = 0
    skipped_regime_hard = 0
    
    regime_downtrend_days = 0
    regime_highvol_days = 0
    total_risk_forming = 0.0
    total_risk_confirmed = 0.0
    count_forming = 0
    count_confirmed = 0
    
    # Weekly risk tracking for rolling budget enforcement
    # List of (date, risk_used) tuples for last 5 trading days
    weekly_risk_history: List[Tuple[pd.Timestamp, float]] = []
    
    corr_matrix = pd.DataFrame()
    clusters: Dict[str, int] = {}
    
    all_signals = []
    for sym, sigs in signals_by_symbol.items():
        for sig in sigs:
            all_signals.append(sig)
    all_signals.sort(key=lambda s: s.entry_date)
    
    if cfg.use_correlation_caps or cfg.use_cluster_caps:
        if all_signals and len(returns_df) > 0:
            first_signal_date = all_signals[0].entry_date
            cache_key = (first_signal_date, cfg.corr_lookback_days, cfg.n_clusters)
            
            if cache_key in ctx.corr_cluster_cache:
                corr_matrix, clusters = ctx.corr_cluster_cache[cache_key]
            else:
                returns_before_signal = returns_df[returns_df.index < first_signal_date]
                if len(returns_before_signal) >= cfg.corr_lookback_days * 0.5:
                    corr_matrix, clusters = compute_symbol_correlation_and_clusters(
                        returns_before_signal,
                        cfg.corr_lookback_days,
                        cfg.n_clusters
                    )
                else:
                    corr_matrix, clusters = pd.DataFrame(), {}
                ctx.corr_cluster_cache[cache_key] = (corr_matrix, clusters)
    
    if not all_dates:
        return pd.DataFrame(), pd.DataFrame()
    
    signal_idx = 0
    running_max_equity = cfg.initial_capital
    
    def close_position(sym: str, pos: OpenPosition, exit_date: pd.Timestamp, 
                       exit_price: float, exit_reason: str) -> dict:
        """Helper to create trade record and update cash."""
        nonlocal cash
        exit_commission = cfg.commission_per_trade
        proceeds = exit_price * pos.shares_remaining - exit_commission
        cash += proceeds
        
        final_leg_pnl = (exit_price - pos.entry_fill) * pos.shares_remaining - exit_commission
        total_pnl_dollars = pos.realized_pnl_dollars + final_leg_pnl - pos.entry_commission
        
        risk_amount = (pos.entry_fill - pos.original_stop_loss) * pos.shares_initial
        pnl_r_multiple = total_pnl_dollars / risk_amount if risk_amount > 0 else 0.0
        hold_days = (exit_date - pos.entry_date).days
        
        pattern_score = pos.meta.get('pattern_score', 0) if pos.meta else 0
        score_features = pos.meta.get('score_features', {}) if pos.meta else {}
        
        return {
            'symbol': sym,
            'pattern_id': pos.pattern_id,
            'entry_kind': pos.entry_kind,
            'entry_date': pos.entry_date,
            'entry_price': round(pos.entry_fill, 2),
            'stop_loss': round(pos.stop_loss, 2),
            'original_stop_loss': round(pos.original_stop_loss, 2),
            'take_profit': round(pos.take_profit, 2),
            'shares': pos.shares_initial,
            'exit_date': exit_date,
            'exit_price': round(exit_price, 2),
            'exit_reason': exit_reason,
            'pnl_dollars': round(total_pnl_dollars, 2),
            'pnl_r_multiple': round(pnl_r_multiple, 2),
            'hold_days': hold_days,
            'bars_held': pos.bars_held,
            'mfe_r': round(pos.mfe_r, 2),
            'slippage_bps': pos.slippage_bps,
            'commissions': round(pos.entry_commission + exit_commission, 2),
            'partial_tp_done': pos.partial_tp_done,
            'pattern_score': round(pattern_score, 2) if pattern_score else 0,
            'score_symmetry': round(score_features.get('symmetry', 0), 4),
            'score_neckline': round(score_features.get('neckline', 0), 4),
            'score_separation': round(score_features.get('separation', 0), 4),
            'score_breakout': round(score_features.get('breakout_strength', 0), 4),
            'score_volume': round(score_features.get('volume', 0), 4),
            'score_trend': round(score_features.get('trend_context', 0), 4),
            'risk_fraction_applied': round(pos.meta.get('risk_fraction_applied', 0.0), 6) if pos.meta else 0.0,
            'score_risk_mult': round(pos.meta.get('score_risk_mult', 1.0), 4) if pos.meta else 1.0,
            'regime_risk_mult': round(pos.meta.get('regime_risk_mult', 1.0), 4) if pos.meta else 1.0,
            'allocation_score': round(pos.meta.get('allocation_score', 0.0), 4) if pos.meta else 0.0,
            'allocation_scale': round(pos.meta.get('allocation_scale', 1.0), 4) if pos.meta else 1.0,
            'allocation_rank': int(pos.meta.get('allocation_rank', 0)) if pos.meta else 0,
            'allocation_budget_used': round(pos.meta.get('allocation_budget_used', 0.0), 6) if pos.meta else 0.0,
            'daily_risk_budget': round(cfg.daily_risk_budget, 4),
            'meta_json': json.dumps(_convert_to_serializable(pos.meta)) if pos.meta else '{}'
        }
    
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
            close_price = bar['Close']
            
            prior_close = None
            prior_atr = None
            df_before = df[df.index < current_date]
            if len(df_before) > 0:
                prior_close = df_before['Close'].iloc[-1]
                if sym in atr_by_symbol:
                    atr_series = atr_by_symbol[sym]
                    prior_date = df_before.index[-1]
                    if prior_date in atr_series.index:
                        prior_atr = atr_series.loc[prior_date]
            
            if pos.entry_kind == "CONFIRMED" and cfg.confirmed_trailing_enabled and prior_close is not None and prior_atr is not None:
                trailing_start_ok = pos.mfe_r >= cfg.confirmed_trailing_start_r or pos.partial_tp_done
                if trailing_start_ok:
                    pos.trailing_active = True
                    trail_stop_candidate = prior_close - cfg.confirmed_trailing_atr_mult * prior_atr
                    pos.stop_loss = max(pos.stop_loss, trail_stop_candidate)
            
            pos.bars_held += 1
            
            exit_price = None
            exit_reason = None
            
            if pos.entry_kind == "FORMING":
                if pos.bars_held >= cfg.forming_max_hold_days:
                    exit_price = open_price * (1 - cfg.slippage_bps / 10000)
                    exit_reason = "TIME"
                elif pos.bars_held >= cfg.forming_no_progress_days and pos.mfe_r < cfg.forming_no_progress_r:
                    if cfg.forming_no_progress_action == "EXIT":
                        exit_price = open_price * (1 - cfg.slippage_bps / 10000)
                        exit_reason = "NO_PROGRESS"
                    elif cfg.forming_no_progress_action == "TIGHTEN_STOP" and not pos.tightened:
                        stop_dist_orig = pos.entry_fill - pos.original_stop_loss
                        new_stop = pos.entry_fill + cfg.forming_tighten_stop_to_r * stop_dist_orig
                        pos.stop_loss = max(pos.stop_loss, new_stop)
                        pos.tightened = True
            
            if exit_price is None:
                stop_dist = pos.entry_fill - pos.original_stop_loss
                if stop_dist > 0:
                    current_r = (high_price - pos.entry_fill) / stop_dist
                    pos.mfe_r = max(pos.mfe_r, current_r)
                
                if pos.entry_kind == "CONFIRMED" and cfg.confirmed_move_stop_to_be_at_r > 0:
                    if pos.mfe_r >= cfg.confirmed_move_stop_to_be_at_r and not pos.moved_to_be:
                        pos.stop_loss = max(pos.stop_loss, pos.entry_fill)
                        pos.moved_to_be = True
                
                if pos.entry_kind == "CONFIRMED" and cfg.confirmed_partial_tp_enabled and not pos.partial_tp_done:
                    stop_dist_orig = pos.entry_fill - pos.original_stop_loss
                    partial_tp_level = pos.entry_fill + cfg.confirmed_partial_tp_at_r * stop_dist_orig
                    
                    partial_fill_price = None
                    if open_price >= partial_tp_level:
                        partial_fill_price = open_price * (1 - cfg.slippage_bps / 10000)
                    elif high_price >= partial_tp_level:
                        partial_fill_price = partial_tp_level * (1 - cfg.slippage_bps / 10000)
                    
                    if partial_fill_price is not None:
                        shares_to_sell = int(math.floor(pos.shares_initial * cfg.confirmed_partial_tp_fraction))
                        shares_to_sell = max(1, min(shares_to_sell, pos.shares_remaining - 1))
                        
                        if shares_to_sell > 0 and pos.shares_remaining > 1:
                            partial_pnl = (partial_fill_price - pos.entry_fill) * shares_to_sell - cfg.commission_per_trade
                            pos.realized_pnl_dollars += partial_pnl
                            pos.shares_remaining -= shares_to_sell
                            pos.partial_tp_done = True
                            pos.partial_tp_date = current_date
                            cash += partial_fill_price * shares_to_sell - cfg.commission_per_trade
            
            if exit_price is None:
                if open_price <= pos.stop_loss:
                    exit_price = open_price * (1 - cfg.slippage_bps / 10000)
                    exit_reason = "STOP"
                elif pos.shares_remaining <= 0:
                    continue
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
                    if pos.bars_held >= cfg.max_hold_days:
                        exit_price = close_price * (1 - cfg.slippage_bps / 10000)
                        exit_reason = "TIME"
            
            if exit_price is not None:
                positions_to_close.append((sym, pos, current_date, exit_price, exit_reason))
        
        for sym, pos, exit_date, exit_price, exit_reason in positions_to_close:
            trade_record = close_position(sym, pos, exit_date, exit_price, exit_reason)
            completed_trades.append(trade_record)
            
            del open_positions[sym]
            
            if exit_date not in exited_today:
                exited_today[exit_date] = set()
            exited_today[exit_date].add(sym)
        
        # Collect all signals for current_date
        todays_signals = []
        while signal_idx < len(all_signals):
            signal = all_signals[signal_idx]
            if signal.entry_date > current_date:
                break
            if signal.entry_date == current_date:
                todays_signals.append(signal)
            signal_idx += 1
        
        # Compute current equity once for all signals
        current_equity = cash + sum(
            pos.shares_remaining * _get_prior_close(price_data_by_symbol, pos.symbol, current_date, pos.entry_fill)
            for pos in open_positions.values()
        )
        
        # Pre-compute regime signals for current_date (once)
        current_trend = None
        current_vol = None
        if cfg.use_regime_filter and len(regime_df) > 0:
            if current_date in regime_df.index:
                current_trend = regime_df.loc[current_date, 'trend_signal']
                current_vol = regime_df.loc[current_date, 'vol_signal']
        
        # Build candidates for allocation
        candidates: List[AllocationCandidate] = []
        symbols_in_candidates: set = set()
        
        for signal in todays_signals:
            sym = signal.symbol
            
            # Basic validation
            if sym not in price_data_by_symbol:
                continue
            df = price_data_by_symbol[sym]
            if current_date not in df.index:
                continue
            bar = df.loc[current_date]
            open_price = bar['Open']
            if open_price < cfg.min_price:
                continue
            
            # Symbol already open check
            if cfg.one_position_per_symbol and sym in open_positions:
                skipped_symbol_already_open += 1
                continue
            
            # For batched processing, also skip if we already have a candidate for this symbol
            if cfg.one_position_per_symbol and sym in symbols_in_candidates:
                continue
            
            # Position limits check (using current state)
            open_total = len(open_positions) + len(candidates)
            open_forming = sum(1 for p in open_positions.values() if p.entry_kind == "FORMING") + \
                          sum(1 for c in candidates if c.signal.entry_kind == "FORMING")
            open_confirmed = sum(1 for p in open_positions.values() if p.entry_kind == "CONFIRMED") + \
                            sum(1 for c in candidates if c.signal.entry_kind == "CONFIRMED")
            
            if open_total >= cfg.max_positions_total:
                skipped_max_total += 1
                continue
            if signal.entry_kind == "FORMING" and open_forming >= cfg.max_positions_forming:
                skipped_max_forming += 1
                continue
            if signal.entry_kind == "CONFIRMED" and open_confirmed >= cfg.max_positions_confirmed:
                skipped_max_confirmed += 1
                continue
            
            # Correlation caps
            if cfg.use_correlation_caps and len(open_positions) > 0 and sym in corr_matrix.columns:
                max_corr_found = 0.0
                for open_sym in open_positions.keys():
                    if open_sym in corr_matrix.columns:
                        try:
                            corr_val = corr_matrix.loc[sym, open_sym]
                            if not pd.isna(corr_val):
                                max_corr_found = max(max_corr_found, abs(corr_val))
                        except (KeyError, TypeError):
                            pass
                if max_corr_found >= cfg.max_corr_to_existing:
                    skipped_corr_cap += 1
                    continue
            
            # Cluster caps
            if cfg.use_cluster_caps and sym in clusters:
                sym_cluster = clusters[sym]
                cluster_count = sum(
                    1 for p in open_positions.values() 
                    if clusters.get(p.symbol, -1) == sym_cluster
                )
                if cluster_count >= cfg.max_positions_per_cluster:
                    skipped_cluster_cap += 1
                    continue
            
            # Regime filter
            regime_risk_mult = 1.0
            if cfg.use_regime_filter:
                if cfg.regime_soft_gate:
                    if signal.entry_kind == "FORMING":
                        if current_trend == "DOWN":
                            regime_risk_mult *= cfg.regime_downtrend_forming_risk_mult
                            reduced_regime_forming_downtrend += 1
                        if current_vol == "HIGH":
                            regime_risk_mult *= cfg.regime_highvol_forming_risk_mult
                            reduced_regime_forming_highvol += 1
                    else:
                        if current_vol == "HIGH":
                            regime_risk_mult *= cfg.regime_highvol_confirmed_risk_mult
                            reduced_regime_confirmed_highvol += 1
                else:
                    if signal.entry_kind == "FORMING":
                        if current_trend == "DOWN" or current_vol == "HIGH":
                            skipped_regime_hard += 1
                            continue
            
            # Same day reentry check
            if not cfg.allow_same_day_reentry:
                if current_date in exited_today and sym in exited_today[current_date]:
                    continue
            
            # Compute entry and stop distance
            entry_fill = open_price * (1 + cfg.slippage_bps / 10000)
            stop_dist = entry_fill - signal.stop_loss
            if stop_dist <= 0:
                skipped_invalid_sizing += 1
                continue
            
            # Compute risk fraction
            if signal.entry_kind == "CONFIRMED":
                risk_fraction = cfg.risk_fraction_confirmed
            else:
                risk_fraction = cfg.risk_fraction_forming
            risk_fraction *= regime_risk_mult
            
            # Apply score-based scaling
            score_mult = 1.0
            if cfg.use_score_risk_scaling:
                apply_to = (cfg.score_risk_apply_to or "BOTH").upper()
                kind = signal.entry_kind.upper()
                should_apply = (
                    apply_to == "BOTH"
                    or (apply_to == "FORMING" and kind == "FORMING")
                    or (apply_to == "CONFIRMED" and kind == "CONFIRMED")
                )
                if should_apply:
                    pattern_score = signal.meta.get("pattern_score", None) if signal.meta else None
                    score_mult = compute_score_risk_mult(
                        pattern_score=pattern_score,
                        alpha=cfg.score_risk_alpha,
                        min_mult=cfg.score_risk_min_mult,
                        max_mult=cfg.score_risk_max_mult,
                        missing_policy=cfg.score_risk_missing_policy,
                    )
                    risk_fraction *= score_mult
            
            # Cap risk fraction
            if cfg.max_risk_fraction_per_trade is not None:
                risk_fraction = min(risk_fraction, cfg.max_risk_fraction_per_trade)
            
            # Compute allocation score for ranking
            alloc_score = compute_signal_allocation_score(signal)
            
            # Create candidate
            candidate = AllocationCandidate(
                signal=signal,
                entry_fill=entry_fill,
                stop_dist=stop_dist,
                risk_fraction=risk_fraction,
                allocation_score=alloc_score,
                regime_risk_mult=regime_risk_mult,
                score_risk_mult=score_mult,
            )
            candidates.append(candidate)
            symbols_in_candidates.add(sym)
        
        # Apply portfolio allocation if enabled
        if cfg.use_portfolio_allocator and candidates:
            # Compute rolling weekly risk used (last 5 trading days)
            weekly_risk_used = sum(risk for _, risk in weekly_risk_history[-4:])  # -4 because today not yet added
            candidates = allocate_daily_signals(candidates, current_equity, cfg, weekly_risk_used)
        
        # Execute entries for allocated candidates
        for cand in candidates:
            signal = cand.signal
            sym = signal.symbol
            
            # Apply allocation scale to risk fraction
            final_risk_fraction = cand.risk_fraction * cand.allocation_scale
            
            # Calculate shares from scaled risk
            risk_budget = final_risk_fraction * current_equity
            shares = int(math.floor(risk_budget / cand.stop_dist))
            
            if shares < 1:
                skipped_invalid_sizing += 1
                continue
            
            # Check available capital
            required_capital = cand.entry_fill * shares + cfg.commission_per_trade
            if required_capital > cash:
                shares = int((cash - cfg.commission_per_trade) / cand.entry_fill)
                if shares < 1:
                    skipped_invalid_sizing += 1
                    continue
            
            entry_commission = cfg.commission_per_trade
            cash -= (cand.entry_fill * shares + entry_commission)
            
            # Build position metadata
            pos_meta = signal.meta.copy() if signal.meta else {}
            pos_meta["risk_fraction_applied"] = float(final_risk_fraction)
            pos_meta["score_risk_mult"] = float(cand.score_risk_mult)
            pos_meta["regime_risk_mult"] = float(cand.regime_risk_mult)
            pos_meta["allocation_score"] = float(cand.allocation_score)
            pos_meta["allocation_scale"] = float(cand.allocation_scale)
            pos_meta["allocation_rank"] = int(cand.allocation_rank)
            pos_meta["allocation_budget_used"] = float(cand.allocation_budget_used)
            
            position = OpenPosition(
                symbol=sym,
                pattern_id=signal.pattern_id,
                entry_kind=signal.entry_kind,
                entry_date=current_date,
                entry_fill=cand.entry_fill,
                stop_loss=signal.stop_loss,
                original_stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                shares=shares,
                entry_commission=entry_commission,
                slippage_bps=cfg.slippage_bps,
                meta=pos_meta,
                shares_initial=shares,
                shares_remaining=shares
            )
            open_positions[sym] = position
            
            if signal.entry_kind == "FORMING":
                total_risk_forming += final_risk_fraction
                count_forming += 1
            else:
                total_risk_confirmed += final_risk_fraction
                count_confirmed += 1
        
        # Track daily risk for weekly budget enforcement
        todays_risk = sum(c.allocation_budget_used for c in candidates)
        weekly_risk_history.append((current_date, todays_risk))
        # Keep only last 5 trading days
        if len(weekly_risk_history) > 5:
            weekly_risk_history = weekly_risk_history[-5:]
        
        open_value = sum(
            pos.shares_remaining * _get_price(price_data_by_symbol, pos.symbol, current_date, pos.entry_fill)
            for pos in open_positions.values()
        )
        total_equity = cash + open_value
        running_max_equity = max(running_max_equity, total_equity)
        drawdown_pct = (total_equity / running_max_equity - 1) * 100 if running_max_equity > 0 else 0.0
        
        equity_history.append({
            'date': current_date,
            'equity': round(total_equity, 2),
            'drawdown_pct': round(drawdown_pct, 2),
            'open_positions': len(open_positions),
            'open_forming': sum(1 for p in open_positions.values() if p.entry_kind == "FORMING"),
            'open_confirmed': sum(1 for p in open_positions.values() if p.entry_kind == "CONFIRMED")
        })
    
    for sym, pos in list(open_positions.items()):
        if sym in price_data_by_symbol:
            df = price_data_by_symbol[sym]
            if len(df) > 0:
                last_date = df.index[-1]
                last_close = df['Close'].iloc[-1]
                exit_price = last_close * (1 - cfg.slippage_bps / 10000)
                trade_record = close_position(sym, pos, last_date, exit_price, 'EOD')
                completed_trades.append(trade_record)
    
    skip_counts = {
        'skipped_max_total': skipped_max_total,
        'skipped_max_forming': skipped_max_forming,
        'skipped_max_confirmed': skipped_max_confirmed,
        'skipped_symbol_already_open': skipped_symbol_already_open,
        'skipped_invalid_sizing': skipped_invalid_sizing,
        'skipped_corr_cap': skipped_corr_cap,
        'skipped_cluster_cap': skipped_cluster_cap,
        'reduced_regime_forming_downtrend': reduced_regime_forming_downtrend,
        'reduced_regime_forming_highvol': reduced_regime_forming_highvol,
        'reduced_regime_confirmed_highvol': reduced_regime_confirmed_highvol,
        'skipped_regime_hard': skipped_regime_hard
    }
    
    if any(skip_counts.values()):
        print("\n  SKIP REASONS:")
        if skipped_max_total > 0:
            print(f"    - Max total positions: {skipped_max_total}")
        if skipped_max_forming > 0:
            print(f"    - Max FORMING positions: {skipped_max_forming}")
        if skipped_max_confirmed > 0:
            print(f"    - Max CONFIRMED positions: {skipped_max_confirmed}")
        if skipped_corr_cap > 0:
            print(f"    - Correlation cap: {skipped_corr_cap}")
        if skipped_cluster_cap > 0:
            print(f"    - Cluster cap: {skipped_cluster_cap}")
        if skipped_regime_hard > 0:
            print(f"    - Regime hard skip (FORMING): {skipped_regime_hard}")
        if skipped_symbol_already_open > 0:
            print(f"    - Symbol already open: {skipped_symbol_already_open}")
        if skipped_invalid_sizing > 0:
            print(f"    - Invalid sizing: {skipped_invalid_sizing}")
    
    if reduced_regime_forming_downtrend > 0 or reduced_regime_forming_highvol > 0 or reduced_regime_confirmed_highvol > 0:
        print("\n  REGIME RISK REDUCTIONS:")
        if reduced_regime_forming_downtrend > 0:
            print(f"    - FORMING in downtrend (x{cfg.regime_downtrend_forming_risk_mult:.2f}): {reduced_regime_forming_downtrend}")
        if reduced_regime_forming_highvol > 0:
            print(f"    - FORMING in high vol (x{cfg.regime_highvol_forming_risk_mult:.2f}): {reduced_regime_forming_highvol}")
        if reduced_regime_confirmed_highvol > 0:
            print(f"    - CONFIRMED in high vol (x{cfg.regime_highvol_confirmed_risk_mult:.2f}): {reduced_regime_confirmed_highvol}")
    
    trades_df = pd.DataFrame(completed_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)
    
    equity_df = pd.DataFrame(equity_history)
    if not equity_df.empty:
        equity_df = equity_df.sort_values('date').reset_index(drop=True)
    
    if return_diagnostics:
        if len(regime_df) > 0:
            regime_downtrend_days = int((regime_df['trend_signal'] == 'DOWN').sum())
            regime_highvol_days = int((regime_df['vol_signal'] == 'HIGH').sum())
        
        diagnostics = {
            'skipped_max_total': skipped_max_total,
            'skipped_max_forming': skipped_max_forming,
            'skipped_max_confirmed': skipped_max_confirmed,
            'skipped_symbol_already_open': skipped_symbol_already_open,
            'skipped_invalid_sizing': skipped_invalid_sizing,
            'skipped_corr_cap': skipped_corr_cap,
            'skipped_cluster_cap': skipped_cluster_cap,
            'skipped_regime_hard': skipped_regime_hard,
            'reduced_regime_forming_downtrend': reduced_regime_forming_downtrend,
            'reduced_regime_forming_highvol': reduced_regime_forming_highvol,
            'reduced_regime_confirmed_highvol': reduced_regime_confirmed_highvol,
            'regime_downtrend_days': regime_downtrend_days,
            'regime_highvol_days': regime_highvol_days,
            'avg_effective_risk_forming': total_risk_forming / count_forming if count_forming > 0 else 0.0,
            'avg_effective_risk_confirmed': total_risk_confirmed / count_confirmed if count_confirmed > 0 else 0.0,
            'count_forming': count_forming,
            'count_confirmed': count_confirmed,
        }
        return trades_df, equity_df, diagnostics
    
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
