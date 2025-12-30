#!/usr/bin/env python3
"""
Task 26 Part B: Recycling Guard Sweep Runner
Runs multiple backtest configurations to find optimal quality gate settings.
"""

import subprocess
import json
import os
import sys
from itertools import product
import pandas as pd

BASE_CMD = [
    "python", "main.py",
    "--backtest-v2",
    "--universe", "demo",
    "--max-stocks", "20",
    "--symbols-seed", "123",
    "--quiet",
    "--price-cache-policy", "READONLY",
    "--daily-risk-budget", "0.01",
    "--daily-risk-budget-forming", "0.003",
    "--max-positions-total", "4",
    "--max-positions-forming", "2",
    "--use-capital-recycling", "true",
    "--recycle-trigger-mode", "ALWAYS",
    "--recycle-action", "PARTIAL",
]

SWEEP_GRID = {
    'recycle_min_score_gap': [0, 3, 5],
    'recycle_min_hold_days': [5, 8, 10],
    'recycle_exclude_confirmed_winners': ['false', 'true'],
}

def run_single_config(score_gap, hold_days, exclude_winners):
    """Run a single backtest configuration and extract metrics."""
    cmd = BASE_CMD + [
        "--recycle-min-score-gap", str(score_gap),
        "--recycle-min-hold-days", str(hold_days),
        "--recycle-exclude-confirmed-winners", exclude_winners,
        "--recycle-replace-only-if-improves-score", "true",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    
    try:
        with open('outputs/metrics.json', 'r') as f:
            metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ERROR reading metrics: {e}")
        return None
    
    overall = metrics.get('overall', {})
    recycling_eff = metrics.get('recycling_effectiveness', {})
    recycling_debug = metrics.get('recycling_debug', {})
    
    return {
        'score_gap': score_gap,
        'hold_days': hold_days,
        'exclude_winners': exclude_winners,
        'total_return_pct': overall.get('total_return_pct', 0),
        'max_drawdown_pct': overall.get('max_drawdown_pct', 0),
        'trades': overall.get('total_trades', 0),
        'recycle_events': recycling_eff.get('recycle_events_count', 0),
        'avg_swap_edge_r': recycling_eff.get('avg_swap_edge_r', 0),
        'false_recycle_rate': recycling_eff.get('false_recycle_rate', 0),
        'pct_trades_recycled': recycling_eff.get('pct_trades_recycled', 0),
        'quality_gate_denied': sum(recycling_debug.get('quality_gate_denied_reasons', {}).values()),
    }


def main():
    print("=" * 60)
    print("RECYCLING GUARD SWEEP")
    print("=" * 60)
    
    os.makedirs('outputs', exist_ok=True)
    
    configs = list(product(
        SWEEP_GRID['recycle_min_score_gap'],
        SWEEP_GRID['recycle_min_hold_days'],
        SWEEP_GRID['recycle_exclude_confirmed_winners'],
    ))
    
    print(f"Running {len(configs)} configurations...")
    print()
    
    results = []
    for i, (score_gap, hold_days, exclude_winners) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] score_gap={score_gap}, hold_days={hold_days}, exclude_winners={exclude_winners}")
        result = run_single_config(score_gap, hold_days, exclude_winners)
        if result:
            results.append(result)
            print(f"  -> Return={result['total_return_pct']:+.2f}%, FalseRate={result['false_recycle_rate']:.1f}%, SwapEdge={result['avg_swap_edge_r']:+.3f}R")
    
    if not results:
        print("No successful runs!")
        return 1
    
    df = pd.DataFrame(results)
    df.to_csv('outputs/recycling_guard_sweep.csv', index=False)
    print(f"\nSaved: outputs/recycling_guard_sweep.csv ({len(df)} rows)")
    
    df_sorted = df.sort_values(
        ['false_recycle_rate', 'avg_swap_edge_r', 'total_return_pct'],
        ascending=[True, False, False]
    ).head(5)
    
    print("\n" + "=" * 60)
    print("TOP 5 CONFIGURATIONS (by false_recycle_rate, swap_edge, return)")
    print("=" * 60)
    print(f"{'Gap':>4} {'Days':>4} {'ExclWin':>7} {'Return%':>8} {'MaxDD%':>7} {'Trades':>6} {'Recycle':>7} {'SwapEdge':>9} {'FalseRate':>9}")
    print("-" * 70)
    for _, row in df_sorted.iterrows():
        print(f"{row['score_gap']:>4} {row['hold_days']:>4} {row['exclude_winners']:>7} "
              f"{row['total_return_pct']:>+7.2f}% {row['max_drawdown_pct']:>+6.2f}% "
              f"{row['trades']:>6} {row['recycle_events']:>7} "
              f"{row['avg_swap_edge_r']:>+8.3f}R {row['false_recycle_rate']:>8.1f}%")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG:")
    best = df_sorted.iloc[0]
    print(f"  --recycle-min-score-gap {best['score_gap']}")
    print(f"  --recycle-min-hold-days {best['hold_days']}")
    print(f"  --recycle-exclude-confirmed-winners {best['exclude_winners']}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
