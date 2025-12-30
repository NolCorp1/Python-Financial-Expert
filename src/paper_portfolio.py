"""
Paper Portfolio State Management

Provides persistent state tracking for paper trading operations.
Stores positions, cash, equity history, and enables mark-to-market updates.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid


DEFAULT_STATE = {
    "as_of_date": None,
    "cash": 100000.0,
    "equity": 100000.0,
    "initial_capital": 100000.0,
    "open_positions": [],
    "closed_positions": [],
    "history": [],
}


def generate_position_uid() -> str:
    """Generate a unique position ID."""
    return str(uuid.uuid4())[:8]


def load_state(path: str) -> Dict[str, Any]:
    """
    Load paper portfolio state from JSON file.
    Creates default state if file doesn't exist.
    
    Args:
        path: Path to portfolio_state.json
        
    Returns:
        Portfolio state dictionary
    """
    path = Path(path)
    
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        state = DEFAULT_STATE.copy()
        state["open_positions"] = []
        state["closed_positions"] = []
        state["history"] = []
        return state
    
    with open(path, 'r') as f:
        state = json.load(f)
    
    for key, default_val in DEFAULT_STATE.items():
        if key not in state:
            if isinstance(default_val, list):
                state[key] = []
            else:
                state[key] = default_val
    
    return state


def save_state(path: str, state: Dict[str, Any], backup: bool = True) -> None:
    """
    Save paper portfolio state to JSON file.
    
    Args:
        path: Path to portfolio_state.json
        state: Portfolio state dictionary
        backup: If True, create backup before overwriting
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if backup and path.exists():
        backup_dir = path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"portfolio_state_{timestamp}.json"
        shutil.copy(path, backup_path)
        
        backups = sorted(backup_dir.glob("portfolio_state_*.json"), reverse=True)
        for old_backup in backups[10:]:
            old_backup.unlink()
    
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def mark_to_market(
    state: Dict[str, Any],
    prices_by_symbol: Dict[str, float],
    date: str
) -> Tuple[Dict[str, Any], float]:
    """
    Mark open positions to market prices and compute daily PnL.
    
    Args:
        state: Current portfolio state
        prices_by_symbol: Dict of symbol -> close price for date
        date: Current date (YYYY-MM-DD)
        
    Returns:
        Tuple of (updated_state, daily_pnl)
    """
    state = state.copy()
    state["open_positions"] = [p.copy() for p in state["open_positions"]]
    
    prior_equity = state.get("equity", state.get("initial_capital", 100000.0))
    
    positions_value = 0.0
    for pos in state["open_positions"]:
        symbol = pos["symbol"]
        current_price = prices_by_symbol.get(symbol)
        
        if current_price is not None:
            pos["current_price"] = current_price
            pos["unrealized_pnl"] = (current_price - pos["entry_price"]) * pos["qty"]
            pos["unrealized_r"] = compute_unrealized_r(pos)
            positions_value += current_price * pos["qty"]
        else:
            positions_value += pos.get("current_price", pos["entry_price"]) * pos["qty"]
    
    state["equity"] = state["cash"] + positions_value
    state["as_of_date"] = date
    
    daily_pnl = state["equity"] - prior_equity
    
    return state, daily_pnl


def compute_unrealized_r(pos: Dict[str, Any]) -> float:
    """Compute unrealized R-multiple for a position."""
    entry_price = pos.get("entry_price", 0)
    stop_price = pos.get("stop_price", 0)
    current_price = pos.get("current_price", entry_price)
    
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return 0.0
    
    gain_per_share = current_price - entry_price
    return gain_per_share / risk_per_share


def apply_fills(
    state: Dict[str, Any],
    fills: List[Dict[str, Any]],
    date: str
) -> Dict[str, Any]:
    """
    Apply order fills to update portfolio state.
    
    Args:
        state: Current portfolio state
        fills: List of fill dictionaries with:
            - symbol
            - side: "BUY" or "SELL"
            - qty
            - fill_price
            - order_info (metadata from original order)
        date: Fill date (YYYY-MM-DD)
        
    Returns:
        Updated portfolio state
    """
    state = state.copy()
    state["open_positions"] = [p.copy() for p in state["open_positions"]]
    state["closed_positions"] = list(state["closed_positions"])
    
    for fill in fills:
        symbol = fill["symbol"]
        side = fill["side"]
        qty = fill["qty"]
        fill_price = fill["fill_price"]
        order_info = fill.get("order_info", {})
        
        if side == "BUY":
            cost = qty * fill_price
            state["cash"] -= cost
            
            new_position = {
                "symbol": symbol,
                "position_uid": generate_position_uid(),
                "qty": qty,
                "entry_date": date,
                "entry_price": fill_price,
                "stop_price": order_info.get("stop_price", fill_price * 0.95),
                "risk_budget_used": order_info.get("risk_budget", 0.01),
                "stage": order_info.get("stage", "FORMING"),
                "current_price": fill_price,
                "unrealized_pnl": 0.0,
                "unrealized_r": 0.0,
                "metadata": order_info.get("metadata", {}),
            }
            state["open_positions"].append(new_position)
            
        elif side == "SELL":
            pos_idx = None
            for i, pos in enumerate(state["open_positions"]):
                if pos["symbol"] == symbol:
                    pos_idx = i
                    break
            
            if pos_idx is not None:
                pos = state["open_positions"].pop(pos_idx)
                proceeds = qty * fill_price
                state["cash"] += proceeds
                
                closed = pos.copy()
                closed["exit_date"] = date
                closed["exit_price"] = fill_price
                closed["realized_pnl"] = (fill_price - pos["entry_price"]) * qty
                closed["realized_r"] = compute_realized_r(pos, fill_price)
                closed["exit_reason"] = order_info.get("reason", "MANUAL")
                state["closed_positions"].append(closed)
    
    return state


def compute_realized_r(pos: Dict[str, Any], exit_price: float) -> float:
    """Compute realized R-multiple for a closed position."""
    entry_price = pos.get("entry_price", 0)
    stop_price = pos.get("stop_price", 0)
    
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return 0.0
    
    gain_per_share = exit_price - entry_price
    return gain_per_share / risk_per_share


def enforce_caps(
    state: Dict[str, Any],
    max_total: int,
    max_forming: int
) -> None:
    """
    Validate that current positions don't exceed caps.
    Raises ValueError if violated.
    
    Args:
        state: Current portfolio state
        max_total: Maximum total open positions
        max_forming: Maximum forming stage positions
    """
    open_positions = state.get("open_positions", [])
    
    total_count = len(open_positions)
    forming_count = sum(1 for p in open_positions if p.get("stage") == "FORMING")
    confirmed_count = total_count - forming_count
    
    errors = []
    
    if total_count > max_total:
        errors.append(f"Total positions ({total_count}) exceeds max_total ({max_total})")
    
    if forming_count > max_forming:
        errors.append(f"Forming positions ({forming_count}) exceeds max_forming ({max_forming})")
    
    if errors:
        raise ValueError("Position cap violation: " + "; ".join(errors))


def get_position_counts(state: Dict[str, Any]) -> Dict[str, int]:
    """Get current position counts by stage."""
    open_positions = state.get("open_positions", [])
    
    total = len(open_positions)
    forming = sum(1 for p in open_positions if p.get("stage") == "FORMING")
    confirmed = total - forming
    
    return {
        "total": total,
        "forming": forming,
        "confirmed": confirmed,
    }


def get_open_symbols(state: Dict[str, Any]) -> set:
    """Get set of symbols with open positions."""
    return {p["symbol"] for p in state.get("open_positions", [])}


def compute_portfolio_stats(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute current portfolio statistics.
    
    Returns dict with:
        - equity
        - cash
        - positions_value
        - total_positions
        - total_unrealized_pnl
        - drawdown_from_peak
        - inception_return_pct
    """
    equity = state.get("equity", state.get("initial_capital", 100000.0))
    cash = state.get("cash", equity)
    initial_capital = state.get("initial_capital", 100000.0)
    
    positions_value = 0.0
    total_unrealized_pnl = 0.0
    
    for pos in state.get("open_positions", []):
        current_price = pos.get("current_price", pos.get("entry_price", 0))
        qty = pos.get("qty", 0)
        entry_price = pos.get("entry_price", 0)
        
        positions_value += current_price * qty
        total_unrealized_pnl += (current_price - entry_price) * qty
    
    history = state.get("history", [])
    peak_equity = initial_capital
    for snap in history:
        if snap.get("equity", 0) > peak_equity:
            peak_equity = snap["equity"]
    if equity > peak_equity:
        peak_equity = equity
    
    drawdown_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0.0
    inception_return_pct = ((equity - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0
    
    return {
        "equity": equity,
        "cash": cash,
        "positions_value": positions_value,
        "total_positions": len(state.get("open_positions", [])),
        "total_unrealized_pnl": total_unrealized_pnl,
        "drawdown_from_peak_pct": drawdown_pct,
        "inception_return_pct": inception_return_pct,
        "peak_equity": peak_equity,
    }


def add_daily_snapshot(
    state: Dict[str, Any],
    date: str,
    daily_pnl: float
) -> Dict[str, Any]:
    """
    Add daily snapshot to history.
    
    Args:
        state: Current portfolio state
        date: Snapshot date (YYYY-MM-DD)
        daily_pnl: Daily P&L value
        
    Returns:
        Updated state with new snapshot
    """
    state = state.copy()
    state["history"] = list(state.get("history", []))
    
    counts = get_position_counts(state)
    
    snapshot = {
        "date": date,
        "equity": state.get("equity", 0),
        "cash": state.get("cash", 0),
        "open_positions_count": counts["total"],
        "forming_count": counts["forming"],
        "confirmed_count": counts["confirmed"],
        "daily_pnl": daily_pnl,
    }
    
    state["history"].append(snapshot)
    
    return state
