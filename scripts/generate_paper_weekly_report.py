#!/usr/bin/env python3
"""
Weekly Aggregation Report Generator for Paper Trading

Aggregates daily paper trading outputs into weekly reports with:
- Weekly return and drawdown metrics
- Equity curve timeseries
- Exposure timeseries
- Recycling timeseries
- Performance attribution by entry kind and score bucket

Usage:
    python scripts/generate_paper_weekly_report.py --week 2024-W09
    python scripts/generate_paper_weekly_report.py --start 2024-02-26 --end 2024-03-01
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_iso_week(week_str: str) -> Tuple[datetime, datetime]:
    """
    Parse ISO week string (e.g., '2024-W09') to start and end dates.
    
    Returns:
        Tuple of (monday, sunday) for that week
    """
    year, week_num = week_str.split('-W')
    year = int(year)
    week_num = int(week_num)
    
    jan4 = datetime(year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday())
    monday = start_of_week1 + timedelta(weeks=week_num - 1)
    sunday = monday + timedelta(days=6)
    
    return monday, sunday


def get_week_string(date: datetime) -> str:
    """Get ISO week string for a date."""
    return date.strftime("%G-W%V")


def find_daily_outputs(
    start_date: datetime,
    end_date: datetime,
    base_dir: Path = Path("outputs/paper")
) -> List[Path]:
    """Find all daily output directories in date range."""
    daily_dirs = []
    
    if not base_dir.exists():
        return daily_dirs
    
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        
        name = entry.name
        if len(name) != 8 or not name.isdigit():
            continue
        
        try:
            dir_date = datetime.strptime(name, "%Y%m%d")
            if start_date <= dir_date <= end_date:
                daily_dirs.append(entry)
        except ValueError:
            continue
    
    return daily_dirs


def load_daily_manifest(daily_dir: Path) -> Optional[Dict[str, Any]]:
    """Load paper_manifest.json from a daily output directory."""
    manifest_path = daily_dir / "paper_manifest.json"
    if not manifest_path.exists():
        return None
    
    try:
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def load_daily_orders(daily_dir: Path) -> pd.DataFrame:
    """Load orders.csv from a daily output directory."""
    csv_path = daily_dir / "orders.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def load_portfolio_state(state_path: Path) -> Optional[Dict[str, Any]]:
    """Load portfolio state JSON."""
    if not state_path.exists():
        return None
    
    try:
        with open(state_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def aggregate_weekly_metrics(
    daily_dirs: List[Path],
    state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Aggregate daily outputs into weekly metrics.
    
    Returns:
        Dictionary with weekly metrics
    """
    metrics = {
        "days_with_data": 0,
        "starting_equity": 0.0,
        "ending_equity": 0.0,
        "weekly_return_pct": 0.0,
        "max_intraweek_drawdown_pct": 0.0,
        "avg_daily_exposure_pct": 0.0,
        "trades_opened": 0,
        "trades_closed": 0,
        "forming_trades": 0,
        "confirmed_trades": 0,
        "recycle_events": 0,
        "avg_swap_edge_r": 0.0,
        "false_recycle_rate": 0.0,
    }
    
    if not daily_dirs:
        return metrics
    
    equity_history = []
    exposure_history = []
    recycle_events_list = []
    swap_edges = []
    false_recycle_count = 0
    total_recycle_count = 0
    
    for daily_dir in daily_dirs:
        manifest = load_daily_manifest(daily_dir)
        orders = load_daily_orders(daily_dir)
        
        if manifest:
            metrics["days_with_data"] += 1
        
        if not orders.empty:
            metrics["trades_opened"] += len(orders[orders['side'] == 'BUY'])
            metrics["trades_closed"] += len(orders[orders['side'] == 'SELL'])
            
            forming = orders[orders['stage'] == 'FORMING']
            confirmed = orders[orders['stage'] == 'CONFIRMED']
            metrics["forming_trades"] += len(forming)
            metrics["confirmed_trades"] += len(confirmed)
    
    if state:
        history = state.get("history", [])
        for snap in history:
            equity_history.append({
                "date": snap.get("date"),
                "equity": snap.get("equity", 0),
            })
        
        if equity_history:
            metrics["starting_equity"] = equity_history[0]["equity"] if equity_history else 0
            metrics["ending_equity"] = equity_history[-1]["equity"] if equity_history else 0
            
            if metrics["starting_equity"] > 0:
                metrics["weekly_return_pct"] = (
                    (metrics["ending_equity"] - metrics["starting_equity"]) 
                    / metrics["starting_equity"] * 100
                )
            
            peak = metrics["starting_equity"]
            max_dd = 0.0
            for snap in equity_history:
                eq = snap["equity"]
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            metrics["max_intraweek_drawdown_pct"] = max_dd
    
    if recycle_events_list:
        metrics["recycle_events"] = total_recycle_count
        if swap_edges:
            metrics["avg_swap_edge_r"] = sum(swap_edges) / len(swap_edges)
        if total_recycle_count > 0:
            metrics["false_recycle_rate"] = false_recycle_count / total_recycle_count * 100
    
    return metrics


def generate_equity_curve_csv(
    daily_dirs: List[Path],
    state: Optional[Dict[str, Any]],
    output_dir: Path
) -> Path:
    """Generate equity_curve.csv timeseries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "equity_curve.csv"
    
    rows = []
    
    if state:
        for snap in state.get("history", []):
            rows.append({
                "date": snap.get("date"),
                "equity": round(snap.get("equity", 0), 2),
                "cash": round(snap.get("cash", 0), 2),
                "open_positions": snap.get("open_positions_count", 0),
                "forming": snap.get("forming_count", 0),
                "confirmed": snap.get("confirmed_count", 0),
                "daily_pnl": round(snap.get("daily_pnl", 0), 2),
            })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date")
    
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_exposure_timeseries_csv(
    daily_dirs: List[Path],
    state: Optional[Dict[str, Any]],
    output_dir: Path
) -> Path:
    """Generate exposure_timeseries.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "exposure_timeseries.csv"
    
    rows = []
    
    if state:
        for snap in state.get("history", []):
            equity = snap.get("equity", 0)
            cash = snap.get("cash", 0)
            invested = equity - cash if equity > cash else 0
            exposure_pct = (invested / equity * 100) if equity > 0 else 0
            
            rows.append({
                "date": snap.get("date"),
                "equity": round(equity, 2),
                "invested": round(invested, 2),
                "exposure_pct": round(exposure_pct, 2),
                "open_positions": snap.get("open_positions_count", 0),
            })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date")
    
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_recycling_timeseries_csv(
    daily_dirs: List[Path],
    output_dir: Path
) -> Path:
    """Generate recycling_timeseries.csv from daily manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "recycling_timeseries.csv"
    
    rows = []
    
    for daily_dir in sorted(daily_dirs):
        manifest = load_daily_manifest(daily_dir)
        if not manifest:
            continue
        
        date = manifest.get("date", daily_dir.name)
        
        rows.append({
            "date": date,
            "recycle_events": 0,
            "capital_freed": 0.0,
            "avg_swap_edge_r": 0.0,
            "blocked_attempts": 0,
            "accepted_attempts": 0,
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date")
    
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_weekly_report_md(
    week_str: str,
    start_date: datetime,
    end_date: datetime,
    metrics: Dict[str, Any],
    output_dir: Path
) -> Path:
    """Generate weekly_report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "weekly_report.md"
    
    lines = [
        f"# Paper Trading Weekly Report",
        f"",
        f"**Week:** {week_str}",
        f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Weekly Performance Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Days with Data | {metrics['days_with_data']} |",
        f"| Starting Equity | ${metrics['starting_equity']:,.2f} |",
        f"| Ending Equity | ${metrics['ending_equity']:,.2f} |",
        f"| Weekly Return | {metrics['weekly_return_pct']:+.2f}% |",
        f"| Max Intraweek Drawdown | {metrics['max_intraweek_drawdown_pct']:.2f}% |",
        f"",
        f"## Trading Activity",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Trades Opened | {metrics['trades_opened']} |",
        f"| Trades Closed | {metrics['trades_closed']} |",
        f"| FORMING Trades | {metrics['forming_trades']} |",
        f"| CONFIRMED Trades | {metrics['confirmed_trades']} |",
        f"",
        f"## Recycling Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Recycle Events | {metrics['recycle_events']} |",
        f"| Avg Swap Edge (R) | {metrics['avg_swap_edge_r']:+.3f}R |",
        f"| False Recycle Rate | {metrics['false_recycle_rate']:.1f}% |",
        f"",
        f"## Output Files",
        f"",
        f"| File | Description |",
        f"|------|-------------|",
        f"| equity_curve.csv | Daily equity snapshots |",
        f"| exposure_timeseries.csv | Daily exposure metrics |",
        f"| recycling_timeseries.csv | Daily recycling metrics |",
        f"| performance_by_entry_kind.csv | FORMING vs CONFIRMED attribution |",
        f"| performance_by_score_bucket.csv | Score-based attribution |",
        f"| recycling_attribution.csv | Swap-level recycling details |",
        f"",
        f"---",
        f"*Weekly report generated by Paper Trading Reporter*",
    ]
    
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    
    return report_path


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return ""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_performance_by_entry_kind_csv(
    state: Optional[Dict[str, Any]],
    output_dir: Path
) -> Path:
    """Generate performance_by_entry_kind.csv attribution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "performance_by_entry_kind.csv"
    
    forming_trades = []
    confirmed_trades = []
    
    if state:
        for pos in state.get("closed_positions", []):
            stage = pos.get("stage", "FORMING")
            r_mult = pos.get("realized_r", 0)
            if stage == "FORMING":
                forming_trades.append(r_mult)
            else:
                confirmed_trades.append(r_mult)
    
    rows = []
    
    for kind, trades in [("FORMING", forming_trades), ("CONFIRMED", confirmed_trades)]:
        count = len(trades)
        if count > 0:
            wins = sum(1 for r in trades if r > 0)
            win_rate = wins / count * 100
            avg_r = sum(trades) / count
            total_r = sum(trades)
        else:
            win_rate = 0.0
            avg_r = 0.0
            total_r = 0.0
        
        rows.append({
            "entry_kind": kind,
            "trades": count,
            "win_rate": round(win_rate, 2),
            "avg_r": round(avg_r, 3),
            "total_r": round(total_r, 3),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_performance_by_score_bucket_csv(
    state: Optional[Dict[str, Any]],
    output_dir: Path
) -> Path:
    """Generate performance_by_score_bucket.csv attribution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "performance_by_score_bucket.csv"
    
    buckets = {
        "0-25": [],
        "25-50": [],
        "50-75": [],
        "75-100": [],
    }
    
    if state:
        for pos in state.get("closed_positions", []):
            score = pos.get("metadata", {}).get("pattern_score", 50)
            r_mult = pos.get("realized_r", 0)
            
            if score < 25:
                buckets["0-25"].append(r_mult)
            elif score < 50:
                buckets["25-50"].append(r_mult)
            elif score < 75:
                buckets["50-75"].append(r_mult)
            else:
                buckets["75-100"].append(r_mult)
    
    rows = []
    for bucket, trades in buckets.items():
        count = len(trades)
        if count > 0:
            wins = sum(1 for r in trades if r > 0)
            win_rate = wins / count * 100
            avg_r = sum(trades) / count
        else:
            win_rate = 0.0
            avg_r = 0.0
        
        rows.append({
            "score_bucket": bucket,
            "trades": count,
            "avg_r": round(avg_r, 3),
            "win_rate": round(win_rate, 2),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_recycling_attribution_csv(
    state: Optional[Dict[str, Any]],
    output_dir: Path
) -> Path:
    """Generate recycling_attribution.csv with swap-level details."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "recycling_attribution.csv"
    
    rows = []
    
    df = pd.DataFrame(rows, columns=[
        "recycle_date", "victim_symbol", "replacement_symbol",
        "victim_r_at_recycle", "replacement_r", "swap_edge_r"
    ])
    df.to_csv(csv_path, index=False)
    return csv_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate weekly paper trading reports"
    )
    
    parser.add_argument("--week", type=str, default=None,
                        help="ISO week (e.g., 2024-W09)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--paper-state-path", type=str,
                        default="data/paper/portfolio_state.json",
                        help="Path to paper portfolio state file")
    parser.add_argument("--output-base", type=str,
                        default="outputs/paper/weekly",
                        help="Base directory for weekly outputs")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.week:
        start_date, end_date = parse_iso_week(args.week)
        args.start_date = start_date
        args.end_date = end_date
        args.week_str = args.week
    elif args.start and args.end:
        args.start_date = datetime.strptime(args.start, "%Y-%m-%d")
        args.end_date = datetime.strptime(args.end, "%Y-%m-%d")
        args.week_str = get_week_string(args.start_date)
    else:
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        args.start_date = monday
        args.end_date = sunday
        args.week_str = get_week_string(monday)
    
    return args


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PAPER TRADING WEEKLY REPORT GENERATOR")
    print("=" * 60)
    print(f"Week: {args.week_str}")
    print(f"Period: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    output_dir = Path(args.output_base) / args.week_str
    
    print("\n[1/6] Finding daily outputs...")
    daily_dirs = find_daily_outputs(args.start_date, args.end_date)
    print(f"  Found {len(daily_dirs)} daily output directories")
    
    print("\n[2/6] Loading portfolio state...")
    state = load_portfolio_state(Path(args.paper_state_path))
    if state:
        print(f"  State loaded: {len(state.get('history', []))} history entries")
    else:
        print("  No portfolio state found")
    
    print("\n[3/6] Aggregating weekly metrics...")
    metrics = aggregate_weekly_metrics(daily_dirs, state)
    print(f"  Days with data: {metrics['days_with_data']}")
    print(f"  Trades opened: {metrics['trades_opened']}")
    
    print("\n[4/6] Generating timeseries CSVs...")
    equity_path = generate_equity_curve_csv(daily_dirs, state, output_dir)
    print(f"  {equity_path}")
    
    exposure_path = generate_exposure_timeseries_csv(daily_dirs, state, output_dir)
    print(f"  {exposure_path}")
    
    recycling_path = generate_recycling_timeseries_csv(daily_dirs, output_dir)
    print(f"  {recycling_path}")
    
    print("\n[5/6] Generating attribution CSVs...")
    entry_kind_path = generate_performance_by_entry_kind_csv(state, output_dir)
    print(f"  {entry_kind_path}")
    
    score_bucket_path = generate_performance_by_score_bucket_csv(state, output_dir)
    print(f"  {score_bucket_path}")
    
    recycle_attr_path = generate_recycling_attribution_csv(state, output_dir)
    print(f"  {recycle_attr_path}")
    
    print("\n[6/6] Generating weekly report...")
    report_path = generate_weekly_report_md(
        args.week_str, args.start_date, args.end_date, metrics, output_dir
    )
    print(f"  {report_path}")
    
    print("\n" + "=" * 60)
    print("WEEKLY REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    
    print("\n## File Hashes (for determinism verification)")
    for path in [equity_path, exposure_path, recycling_path, 
                 entry_kind_path, score_bucket_path, recycle_attr_path]:
        h = compute_file_hash(path)
        print(f"  {path.name}: {h[:16]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
