#!/usr/bin/env python3
"""
Paper Trading Daily Runner

Runs the double bottom scanner pipeline for a single date,
generates paper orders, updates portfolio state, and produces daily reports.

Usage:
    python -m scripts.run_paper_daily --date 2025-12-30 --universe demo --max-stocks 20
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paper_portfolio import (
    load_state,
    save_state,
    mark_to_market,
    apply_fills,
    enforce_caps,
    get_position_counts,
    get_open_symbols,
    compute_portfolio_stats,
    add_daily_snapshot,
)


def get_ny_date() -> str:
    """Get current date in New York timezone (YYYY-MM-DD)."""
    try:
        from zoneinfo import ZoneInfo
        ny_tz = ZoneInfo("America/New_York")
        return datetime.now(ny_tz).strftime("%Y-%m-%d")
    except ImportError:
        return datetime.now().strftime("%Y-%m-%d")


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return ""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_scanner_pipeline(
    date: str,
    args: argparse.Namespace,
    state: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """
    Run the pattern scanner and signal generation pipeline for a single date.
    
    Returns:
        Tuple of (allocated_candidates, price_data, recycling_stats, signal_intake_stats)
    """
    import random as _random
    from double_bottom_scanner import DEFAULT_CONFIG
    from main import scan_stocks_with_liquidity
    from strategy import generate_signals
    from backtester import (
        BacktestConfig, group_signals_by_symbol,
        compute_signal_allocation_score, AllocationCandidate,
        allocate_daily_signals,
    )
    from universe import get_demo_symbols, get_nasdaq_symbols_cached
    
    config = DEFAULT_CONFIG.copy()
    config['min_pattern_days'] = args.min_pattern_days
    config['max_pattern_days'] = args.max_pattern_days
    config['min_double_bottom_prominence'] = args.min_prominence
    config['price_tolerance'] = args.price_tolerance
    
    if args.universe == 'demo':
        symbols = sorted(get_demo_symbols())
    elif args.universe == 'nasdaq':
        symbols = sorted(get_nasdaq_symbols_cached(max_age_hours=24, limit=None))
    else:
        symbols = sorted(get_demo_symbols())
    
    if args.max_stocks and args.max_stocks < len(symbols):
        if args.symbols_seed is not None:
            _random.seed(args.symbols_seed)
            symbols = sorted(_random.sample(symbols, args.max_stocks))
        else:
            symbols = symbols[:args.max_stocks]
    
    regime_symbol = args.regime_symbol.upper()
    symbols_with_regime = list(symbols)
    if args.use_regime_filter and regime_symbol not in symbols_with_regime:
        symbols_with_regime.append(regime_symbol)
    
    liquidity_config = {
        'min_price': args.min_price,
        'min_dollar_volume': args.min_dollar_vol
    } if not args.disable_liquidity_filter else None
    
    results, price_data, price_provenance = scan_stocks_with_liquidity(
        symbols_with_regime,
        config,
        use_liquidity_filter=not args.disable_liquidity_filter,
        liquidity_config=liquidity_config,
        verbose=not args.quiet,
        price_cache_policy=args.price_cache_policy,
        price_cache_dir=args.price_cache_dir
    )
    
    if results.empty:
        print("  No patterns detected.")
        return [], price_data, {}
    
    if args.use_regime_filter:
        results = results[results['symbol'] != regime_symbol]
    
    all_signals = []
    for sym in price_data:
        if args.use_regime_filter and sym == regime_symbol:
            continue
        df = price_data[sym]
        patterns = results[results['symbol'] == sym].to_dict('records')
        signals = generate_signals(
            df, patterns,
            compute_scores=True,
            price_tolerance=args.price_tolerance,
            min_peak_height=args.min_prominence,
            min_pattern_score=args.min_pattern_score,
            score_policy=args.score_policy,
            trend_mode=args.trend_score_mode,
        )
        all_signals.extend(signals)
    
    target_date = pd.Timestamp(date)
    daily_signals = [s for s in all_signals if s.entry_date == target_date]
    
    print(f"  Signals for {date}: {len(daily_signals)}")
    
    if not daily_signals:
        return [], price_data, {}
    
    backtest_cfg = BacktestConfig(
        initial_capital=state.get("equity", state.get("initial_capital", 100000.0)),
        risk_fraction_per_trade=args.risk_fraction,
        risk_fraction_forming=args.risk_fraction_forming,
        risk_fraction_confirmed=args.risk_fraction_confirmed,
        max_positions_total=args.max_positions_total,
        max_positions_forming=args.max_positions_forming,
        daily_risk_budget=args.daily_risk_budget,
        daily_risk_budget_forming=args.daily_risk_budget_forming,
        use_portfolio_allocator=True,
        use_score_risk_scaling=args.use_score_risk_scaling,
        use_capital_recycling=args.use_capital_recycling,
        recycle_trigger_mode=args.recycle_trigger_mode,
        recycle_min_hold_days=args.recycle_min_hold_days,
        recycle_min_score_gap=args.recycle_min_score_gap,
        recycle_exclude_confirmed_winners=args.recycle_exclude_confirmed_winners,
        recycle_replace_only_if_improves_score=args.recycle_replace_only_if_improves_score,
        recycle_min_expected_edge_r=args.recycle_min_expected_edge_r,
    )
    
    open_symbols = get_open_symbols(state)
    counts = get_position_counts(state)
    
    available_total = args.max_positions_total - counts["total"]
    available_forming = args.max_positions_forming - counts["forming"]
    
    signal_intake_stats = {
        "total_generated": len(daily_signals),
        "skipped": {},
        "accepted": 0,
        "forming_accepted": 0,
        "confirmed_accepted": 0,
    }
    
    skipped_already_open = 0
    skipped_position_cap = 0
    skipped_forming_cap = 0
    
    filtered_signals = []
    for sig in daily_signals:
        if sig.symbol in open_symbols:
            skipped_already_open += 1
            continue
        if available_total <= 0:
            skipped_position_cap += 1
            continue
        stage = getattr(sig, 'entry_kind', 'FORMING')
        if stage == "FORMING" and available_forming <= 0:
            skipped_forming_cap += 1
            continue
        filtered_signals.append(sig)
        available_total -= 1
        if stage == "FORMING":
            available_forming -= 1
    
    if skipped_already_open > 0:
        signal_intake_stats["skipped"]["already_open"] = skipped_already_open
    if skipped_position_cap > 0:
        signal_intake_stats["skipped"]["position_cap"] = skipped_position_cap
    if skipped_forming_cap > 0:
        signal_intake_stats["skipped"]["forming_cap"] = skipped_forming_cap
    
    candidates = []
    target_date = pd.Timestamp(date)
    for sig in filtered_signals:
        symbol = sig.symbol
        if symbol not in price_data:
            continue
        df = price_data[symbol]
        
        if target_date not in df.index:
            continue
        
        row = df.loc[target_date]
        entry_fill = float(row['Close'])
        stop_dist = entry_fill - sig.stop_loss
        if stop_dist <= 0:
            stop_dist = entry_fill * 0.05
        
        risk_fraction = args.risk_fraction_forming if sig.entry_kind == "FORMING" else args.risk_fraction_confirmed
        alloc_score = compute_signal_allocation_score(sig)
        
        cand = AllocationCandidate(
            signal=sig,
            entry_fill=entry_fill,
            stop_dist=stop_dist,
            risk_fraction=risk_fraction,
            allocation_score=alloc_score,
            regime_risk_mult=1.0,
            score_risk_mult=1.0,
        )
        candidates.append(cand)
    
    if candidates:
        allocated = allocate_daily_signals(
            candidates,
            current_equity=state.get("equity", 100000.0),
            cfg=backtest_cfg,
            weekly_risk_used=0.0
        )
    else:
        allocated = []
    
    for cand in allocated:
        stage = getattr(cand.signal, 'entry_kind', 'FORMING')
        signal_intake_stats["accepted"] += 1
        if stage == "FORMING":
            signal_intake_stats["forming_accepted"] += 1
        else:
            signal_intake_stats["confirmed_accepted"] += 1
    
    recycling_stats = {
        "recycle_events_count": 0,
        "avg_swap_edge_r": 0.0,
        "false_recycle_rate": 0.0,
        "capital_freed": 0.0,
        "blocked_attempts": 0,
        "accepted_attempts": 0,
    }
    
    return allocated, price_data, recycling_stats, signal_intake_stats


def generate_orders(
    allocated: List[Any],
    price_data: Dict[str, pd.DataFrame],
    date: str,
    state: Dict[str, Any],
    fill_model: str = "close"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert allocated signals into paper orders and simulated fills.
    
    Args:
        allocated: List of AllocationCandidate objects
        price_data: Price data by symbol
        date: Order date (YYYY-MM-DD)
        state: Current portfolio state
        fill_model: "close" or "next_open"
        
    Returns:
        Tuple of (orders, fills)
    """
    orders = []
    fills = []
    
    target_date = pd.Timestamp(date)
    equity = state.get("equity", state.get("initial_capital", 100000.0))
    
    for cand in allocated:
        sig = cand.signal
        symbol = sig.symbol
        
        if symbol not in price_data:
            continue
            
        df = price_data[symbol]
        
        if target_date not in df.index:
            dates_near = [d for d in df.index if abs((d - target_date).days) <= 5]
            if dates_near:
                target_date_for_symbol = min(dates_near, key=lambda d: abs((d - target_date).days))
            else:
                continue
        else:
            target_date_for_symbol = target_date
        
        row = df.loc[target_date_for_symbol]
        
        if fill_model == "close":
            fill_price = float(row['Close'])
        else:
            next_idx = df.index.get_loc(target_date_for_symbol) + 1
            if next_idx < len(df):
                fill_price = float(df.iloc[next_idx]['Open'])
            else:
                fill_price = float(row['Close'])
        
        risk_per_share = fill_price - sig.stop_loss
        if risk_per_share <= 0:
            risk_per_share = fill_price * 0.05
        
        risk_budget = cand.risk_fraction * cand.allocation_scale * equity
        qty = int(risk_budget / risk_per_share)
        
        if qty <= 0:
            continue
        
        order = {
            "date": date,
            "symbol": symbol,
            "side": "BUY",
            "qty": qty,
            "order_type": "MARKET",
            "limit_price": None,
            "reason": "PATTERN_SIGNAL",
            "stage": getattr(sig, 'entry_kind', 'FORMING'),
            "score": sig.meta.get("pattern_score") if sig.meta else None,
            "risk_budget": risk_budget,
            "stop_price": sig.stop_loss,
            "take_profit": getattr(sig, 'take_profit', None),
            "metadata_json": json.dumps({
                "pattern_id": sig.pattern_id,
                "entry_date": str(sig.entry_date),
                "allocation_score": cand.allocation_score,
                "allocation_rank": cand.allocation_rank,
                "allocation_scale": cand.allocation_scale,
            }),
        }
        orders.append(order)
        
        fill = {
            "symbol": symbol,
            "side": "BUY",
            "qty": qty,
            "fill_price": fill_price,
            "order_info": {
                "stop_price": sig.stop_loss,
                "risk_budget": cand.risk_fraction * cand.allocation_scale,
                "stage": getattr(sig, 'entry_kind', 'FORMING'),
                "metadata": {
                    "pattern_id": sig.pattern_id,
                    "pattern_score": sig.meta.get("pattern_score") if sig.meta else None,
                },
            },
        }
        fills.append(fill)
    
    return orders, fills


def export_orders(orders: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    """Export orders to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "orders.csv"
    json_path = output_dir / "orders.json"
    
    if orders:
        df = pd.DataFrame(orders)
        df.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=[
            "date", "symbol", "side", "qty", "order_type", "limit_price",
            "reason", "stage", "score", "risk_budget", "stop_price",
            "take_profit", "metadata_json"
        ]).to_csv(csv_path, index=False)
    
    with open(json_path, 'w') as f:
        json.dump(orders, f, indent=2, default=str)
    
    return csv_path, json_path


def generate_report(
    date: str,
    state: Dict[str, Any],
    orders: List[Dict[str, Any]],
    recycling_stats: Dict[str, Any],
    daily_pnl: float,
    output_dir: Path,
    signal_intake_stats: Optional[Dict[str, Any]] = None,
    starting_equity: Optional[float] = None,
) -> Path:
    """Generate daily report markdown file with comprehensive attribution sections."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    
    stats = compute_portfolio_stats(state)
    counts = get_position_counts(state)
    
    if starting_equity is None:
        starting_equity = stats['equity'] - daily_pnl
    
    daily_pnl_pct = (daily_pnl / starting_equity * 100) if starting_equity > 0 else 0.0
    cash = stats['cash']
    positions_value = stats['positions_value']
    invested_pct = (positions_value / stats['equity'] * 100) if stats['equity'] > 0 else 0.0
    
    lines = [
        f"# Paper Trading Daily Report",
        f"",
        f"**Date:** {date}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 1. Portfolio Snapshot",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Starting Equity | ${starting_equity:,.2f} |",
        f"| Ending Equity | ${stats['equity']:,.2f} |",
        f"| Daily PnL ($) | ${daily_pnl:+,.2f} |",
        f"| Daily PnL (%) | {daily_pnl_pct:+.2f}% |",
        f"| Cash | ${cash:,.2f} ({100 - invested_pct:.1f}%) |",
        f"| Invested Capital | ${positions_value:,.2f} ({invested_pct:.1f}%) |",
        f"| Open Positions | {counts['total']} |",
        f"| FORMING | {counts['forming']} |",
        f"| CONFIRMED | {counts['confirmed']} |",
        f"| Inception Return | {stats['inception_return_pct']:+.2f}% |",
        f"| Drawdown from Peak | {stats['drawdown_from_peak_pct']:.2f}% |",
        f"",
    ]
    
    open_positions = state.get("open_positions", [])
    if open_positions:
        lines.extend([
            f"## 2. Exposure Summary",
            f"",
            f"### Exposure by Symbol",
            f"",
            f"| Symbol | Stage | Value | % of Equity |",
            f"|--------|-------|-------|-------------|",
        ])
        
        exposures = []
        for pos in open_positions:
            pos_value = pos.get("current_price", pos.get("entry_price", 0)) * pos.get("qty", 0)
            pct_equity = (pos_value / stats['equity'] * 100) if stats['equity'] > 0 else 0.0
            exposures.append({
                "symbol": pos["symbol"],
                "stage": pos.get("stage", "?"),
                "value": pos_value,
                "pct": pct_equity
            })
        
        exposures = sorted(exposures, key=lambda x: -x["value"])
        largest_pct = exposures[0]["pct"] if exposures else 0.0
        
        for exp in exposures:
            lines.append(f"| {exp['symbol']} | {exp['stage']} | ${exp['value']:,.2f} | {exp['pct']:.1f}% |")
        
        lines.extend([
            f"",
            f"**Largest Single Position:** {largest_pct:.1f}% of equity",
            f"",
        ])
    else:
        lines.extend([
            f"## 2. Exposure Summary",
            f"",
            f"No open positions.",
            f"",
        ])
    
    if orders:
        lines.extend([
            f"## New Orders ({len(orders)})",
            f"",
            f"| Symbol | Side | Qty | Stage | Score | Risk Budget |",
            f"|--------|------|-----|-------|-------|-------------|",
        ])
        for o in orders:
            score = o.get('score', 'N/A')
            if score is not None:
                score = f"{score:.1f}"
            else:
                score = "N/A"
            lines.append(
                f"| {o['symbol']} | {o['side']} | {o['qty']} | {o['stage']} | {score} | ${o['risk_budget']:,.0f} |"
            )
        lines.append("")
    else:
        lines.extend([
            f"## New Orders",
            f"",
            f"No new orders for this date.",
            f"",
        ])
    
    open_positions = state.get("open_positions", [])
    if open_positions:
        lines.extend([
            f"## Open Positions ({len(open_positions)})",
            f"",
            f"| Symbol | Stage | Entry | Current | Unrealized R | Days Held |",
            f"|--------|-------|-------|---------|--------------|-----------|",
        ])
        for pos in open_positions:
            entry_date = pos.get("entry_date", "?")
            if isinstance(entry_date, str):
                try:
                    days_held = (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(entry_date, "%Y-%m-%d")).days
                except:
                    days_held = 0
            else:
                days_held = 0
            lines.append(
                f"| {pos['symbol']} | {pos.get('stage', '?')} | ${pos.get('entry_price', 0):.2f} | ${pos.get('current_price', 0):.2f} | {pos.get('unrealized_r', 0):+.2f}R | {days_held} |"
            )
        lines.append("")
    
    lines.extend([
        f"## 3. Recycling Summary",
        f"",
    ])
    
    recycle_count = recycling_stats.get("recycle_events_count", 0)
    if recycle_count > 0:
        capital_freed = recycling_stats.get("capital_freed", 0.0)
        capital_freed_pct = (capital_freed / stats['equity'] * 100) if stats['equity'] > 0 else 0.0
        blocked = recycling_stats.get("blocked_attempts", 0)
        accepted = recycling_stats.get("accepted_attempts", recycle_count)
        
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Recycle Events Today | {recycle_count} |",
            f"| Capital Freed ($) | ${capital_freed:,.2f} |",
            f"| Capital Freed (%) | {capital_freed_pct:.2f}% |",
            f"| Avg Swap Edge (R) | {recycling_stats.get('avg_swap_edge_r', 0):+.3f}R |",
            f"| Blocked Attempts | {blocked} |",
            f"| Accepted Attempts | {accepted} |",
            f"| False Recycle Rate | {recycling_stats.get('false_recycle_rate', 0):.1f}% |",
            f"",
        ])
    else:
        lines.extend([
            f"No recycling events today.",
            f"",
        ])
    
    lines.extend([
        f"## 4. Signal Intake Summary",
        f"",
    ])
    
    if signal_intake_stats:
        total_gen = signal_intake_stats.get("total_generated", 0)
        skipped = signal_intake_stats.get("skipped", {})
        accepted = signal_intake_stats.get("accepted", 0)
        forming_accepted = signal_intake_stats.get("forming_accepted", 0)
        confirmed_accepted = signal_intake_stats.get("confirmed_accepted", 0)
        
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Signals Generated | {total_gen} |",
        ])
        
        if skipped:
            for reason, count in sorted(skipped.items()):
                lines.append(f"| Skipped ({reason}) | {count} |")
        
        lines.extend([
            f"| Signals Accepted | {accepted} |",
            f"| FORMING Accepted | {forming_accepted} |",
            f"| CONFIRMED Accepted | {confirmed_accepted} |",
            f"",
        ])
    else:
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Signals Accepted | {len(orders)} |",
            f"",
        ])
    
    lines.extend([
        f"---",
        f"*Report generated by Paper Trading Runner*",
    ])
    
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    
    return report_path


def generate_manifest(
    date: str,
    args: argparse.Namespace,
    output_dir: Path,
    orders_csv_hash: str,
    report_hash: str,
    price_provenance: Optional[Dict] = None,
) -> Path:
    """Generate paper manifest for audit trail."""
    manifest_path = output_dir / "paper_manifest.json"
    
    manifest = {
        "run_id": f"paper_{date}_{datetime.now().strftime('%H%M%S')}",
        "date": date,
        "git_commit": get_git_commit(),
        "generated_at": datetime.now().isoformat(),
        "cli_args": {
            "universe": args.universe,
            "max_stocks": args.max_stocks,
            "symbols_seed": args.symbols_seed,
            "price_cache_policy": args.price_cache_policy,
            "use_capital_recycling": args.use_capital_recycling,
            "recycle_trigger_mode": args.recycle_trigger_mode,
            "max_positions_total": args.max_positions_total,
            "max_positions_forming": args.max_positions_forming,
            "daily_risk_budget": args.daily_risk_budget,
        },
        "recycling_settings": {
            "enabled": args.use_capital_recycling,
            "trigger_mode": args.recycle_trigger_mode,
            "min_hold_days": args.recycle_min_hold_days,
            "min_score_gap": args.recycle_min_score_gap,
            "min_expected_edge_r": args.recycle_min_expected_edge_r,
        },
        "hashes": {
            "orders_csv": orders_csv_hash,
            "report_md": report_hash,
        },
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    manifest["hashes"]["paper_manifest"] = compute_file_hash(manifest_path)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


def check_idempotence(
    date: str,
    output_dir: Path,
    state: Dict[str, Any],
    force: bool
) -> bool:
    """
    Check if run should proceed based on idempotence rules.
    
    Returns:
        True if should proceed, False if should skip
    """
    manifest_path = output_dir / "paper_manifest.json"
    
    if not manifest_path.exists():
        return True
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        manifest_date = manifest.get("date")
    except Exception:
        manifest_date = None
    
    if manifest_date == date:
        if force:
            print(f"  --force specified, overwriting outputs for {date}")
            return True
        else:
            print(f"  SKIP: Already ran for {date} (manifest exists)")
            print(f"  Use --force to override")
            return False
    
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Paper Trading Daily Runner"
    )
    
    parser.add_argument("--date", type=str, default=None,
                        help="Date to run (YYYY-MM-DD), default: today NY time")
    parser.add_argument("--universe", type=str, default="demo",
                        help="Stock universe: demo, nasdaq, custom")
    parser.add_argument("--max-stocks", type=int, default=20,
                        help="Maximum stocks to scan")
    parser.add_argument("--symbols-seed", type=int, default=None,
                        help="Seed for reproducible symbol selection")
    parser.add_argument("--price-cache-policy", type=str, default="AUTO",
                        choices=["AUTO", "READONLY", "REFRESH", "OFF"],
                        help="Price cache policy")
    parser.add_argument("--price-cache-dir", type=str, default="data/price_cache",
                        help="Price cache directory")
    parser.add_argument("--paper-state-path", type=str,
                        default="data/paper/portfolio_state.json",
                        help="Path to paper portfolio state file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/paper/YYYYMMDD/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write state changes")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if already ran for date")
    parser.add_argument("--export-orders", type=str, default="true",
                        help="Export orders CSV/JSON")
    parser.add_argument("--export-report", type=str, default="true",
                        help="Export daily report")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--fill-model", type=str, default="close",
                        choices=["close", "next_open"],
                        help="Fill price model")
    
    parser.add_argument("--symbol-cache-dir", type=str, default="data/symbol_cache")
    parser.add_argument("--min-pattern-days", type=int, default=10)
    parser.add_argument("--max-pattern-days", type=int, default=90)
    parser.add_argument("--min-prominence", type=float, default=0.03)
    parser.add_argument("--price-tolerance", type=float, default=0.03)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-vol", type=float, default=1e6)
    parser.add_argument("--disable-liquidity-filter", action="store_true")
    parser.add_argument("--min-pattern-score", type=float, default=0.0)
    parser.add_argument("--score-policy", type=str, default="V3_INVERTED")
    parser.add_argument("--trend-score-mode", type=str, default="ma_context")
    parser.add_argument("--use-regime-filter", action="store_true", default=True)
    parser.add_argument("--regime-symbol", type=str, default="QQQ")
    
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--risk-fraction-forming", type=float, default=0.006)
    parser.add_argument("--risk-fraction-confirmed", type=float, default=0.010)
    parser.add_argument("--max-positions-total", type=int, default=10)
    parser.add_argument("--max-positions-forming", type=int, default=3)
    parser.add_argument("--daily-risk-budget", type=float, default=0.04)
    parser.add_argument("--daily-risk-budget-forming", type=float, default=0.015)
    parser.add_argument("--use-score-risk-scaling", action="store_true", default=True)
    
    parser.add_argument("--use-capital-recycling", type=str, default="false")
    parser.add_argument("--recycle-trigger-mode", type=str, default="BUDGET_BLOCKED")
    parser.add_argument("--recycle-min-hold-days", type=int, default=3)
    parser.add_argument("--recycle-min-score-gap", type=float, default=0.0)
    parser.add_argument("--recycle-exclude-confirmed-winners", type=str, default="true")
    parser.add_argument("--recycle-replace-only-if-improves-score", type=str, default="false")
    parser.add_argument("--recycle-min-expected-edge-r", type=float, default=0.15)
    
    args = parser.parse_args()
    
    if args.date is None:
        args.date = get_ny_date()
    
    if args.output_dir is None:
        date_str = args.date.replace("-", "")
        args.output_dir = f"outputs/paper/{date_str}"
    
    args.use_capital_recycling = args.use_capital_recycling.lower() == "true"
    args.recycle_exclude_confirmed_winners = args.recycle_exclude_confirmed_winners.lower() == "true"
    args.recycle_replace_only_if_improves_score = args.recycle_replace_only_if_improves_score.lower() == "true"
    
    return args


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PAPER TRADING DAILY RUNNER")
    print("=" * 60)
    print(f"Date: {args.date}")
    print(f"Universe: {args.universe}")
    print(f"Max stocks: {args.max_stocks}")
    print(f"Price cache: {args.price_cache_policy}")
    print(f"Recycling: {'enabled' if args.use_capital_recycling else 'disabled'}")
    print("-" * 60)
    
    output_dir = Path(args.output_dir)
    
    print("\n[1/6] Loading portfolio state...")
    state = load_state(args.paper_state_path)
    print(f"  Equity: ${state.get('equity', 0):,.2f}")
    print(f"  Open positions: {len(state.get('open_positions', []))}")
    
    if not check_idempotence(args.date, output_dir, state, args.force):
        return 0
    
    print("\n[2/6] Running scanner pipeline...")
    allocated, price_data, recycling_stats, signal_intake_stats = run_scanner_pipeline(
        args.date, args, state
    )
    print(f"  Allocated signals: {len(allocated)}")
    
    starting_equity = state.get("equity", state.get("initial_capital", 100000.0))
    
    print("\n[3/6] Marking portfolio to market...")
    prices_by_symbol = {}
    for sym, df in price_data.items():
        target_date = pd.Timestamp(args.date)
        if target_date in df.index:
            prices_by_symbol[sym] = float(df.loc[target_date, 'Close'])
        elif len(df) > 0:
            prices_by_symbol[sym] = float(df.iloc[-1]['Close'])
    
    state, daily_pnl = mark_to_market(state, prices_by_symbol, args.date)
    print(f"  Daily PnL: ${daily_pnl:+,.2f}")
    print(f"  Current equity: ${state['equity']:,.2f}")
    
    print("\n[4/6] Generating orders...")
    orders, fills = generate_orders(allocated, price_data, args.date, state, args.fill_model)
    print(f"  Orders generated: {len(orders)}")
    
    if not args.dry_run and fills:
        print("\n[5/6] Applying fills and updating state...")
        state = apply_fills(state, fills, args.date)
        state = add_daily_snapshot(state, args.date, daily_pnl)
        
        try:
            enforce_caps(state, args.max_positions_total, args.max_positions_forming)
            print("  Position caps: OK")
        except ValueError as e:
            print(f"  WARNING: {e}")
        
        save_state(args.paper_state_path, state)
        print(f"  State saved to: {args.paper_state_path}")
    else:
        state = add_daily_snapshot(state, args.date, daily_pnl)
        if args.dry_run:
            print("\n[5/6] DRY RUN - state not modified")
        else:
            print("\n[5/6] No fills to apply")
    
    print("\n[6/6] Exporting outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.export_orders.lower() == "true":
        csv_path, json_path = export_orders(orders, output_dir)
        print(f"  Orders CSV: {csv_path}")
        print(f"  Orders JSON: {json_path}")
    else:
        csv_path = output_dir / "orders.csv"
        export_orders([], output_dir)
    
    if args.export_report.lower() == "true":
        report_path = generate_report(
            args.date, state, orders, recycling_stats, daily_pnl, output_dir,
            signal_intake_stats=signal_intake_stats,
            starting_equity=starting_equity,
        )
        print(f"  Report: {report_path}")
    else:
        report_path = output_dir / "report.md"
    
    manifest_path = generate_manifest(
        args.date, args, output_dir,
        orders_csv_hash=compute_file_hash(csv_path),
        report_hash=compute_file_hash(report_path) if report_path.exists() else "",
    )
    print(f"  Manifest: {manifest_path}")
    
    print("\n" + "=" * 60)
    print("PAPER TRADING RUN COMPLETE")
    print("=" * 60)
    
    stats = compute_portfolio_stats(state)
    counts = get_position_counts(state)
    print(f"Equity: ${stats['equity']:,.2f}")
    print(f"Daily PnL: ${daily_pnl:+,.2f}")
    print(f"Inception Return: {stats['inception_return_pct']:+.2f}%")
    print(f"Open Positions: {counts['total']} (F:{counts['forming']}, C:{counts['confirmed']})")
    print(f"New Orders: {len(orders)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
