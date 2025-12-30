#!/usr/bin/env python3
"""
Task 27 Part A: Recycling Robustness Validation Runner

Runs a matrix of backtests across different seeds, stress modes, and universes
to validate that recycling quality gates remain robust.

Usage:
    python -m scripts.validate_recycling_robustness
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys


SEEDS = [1, 2, 3, 7, 42, 123]
STRESS_MODES = ['NONE', 'LOW_BUDGET', 'HIGH_SIGNAL_DENSITY', 'BOTH']
UNIVERSES = ['demo']

BASE_CMD = [
    'python', 'main.py',
    '--backtest-v2',
    '--price-cache-policy', 'READONLY',
    '--use-capital-recycling', 'true',
    '--recycle-trigger-mode', 'ALWAYS',
    '--daily-risk-budget', '0.01',
    '--daily-risk-budget-forming', '0.003',
    '--max-positions-total', '4',
    '--max-positions-forming', '2',
    '--max-stocks', '20',
    '--quiet',
]


def run_backtest(seed: int, stress_mode: str, universe: str) -> dict:
    """Run a single backtest and extract metrics."""
    cmd = BASE_CMD + [
        '--symbols-seed', str(seed),
        '--recycling-stress-mode', stress_mode,
        '--universe', universe,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        metrics_path = Path('outputs/metrics.json')
        if not metrics_path.exists():
            return {
                'seed': seed,
                'stress_mode': stress_mode,
                'universe': universe,
                'status': 'FAILED',
                'error': 'metrics.json not found',
            }
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        all_metrics = metrics.get('ALL', {})
        eff = metrics.get('recycling_effectiveness', {})
        debug = metrics.get('recycling_debug', {})
        
        return {
            'seed': seed,
            'stress_mode': stress_mode,
            'universe': universe,
            'status': 'OK',
            'total_return_pct': all_metrics.get('total_return_pct', 0),
            'max_drawdown_pct': all_metrics.get('max_drawdown_pct', 0),
            'trades': all_metrics.get('trade_count', 0),
            'recycle_events_count': eff.get('recycle_events_count', 0),
            'pct_trades_recycled': eff.get('pct_trades_recycled', 0),
            'avg_swap_edge_r': eff.get('avg_swap_edge_r', 0),
            'false_recycle_rate': eff.get('false_recycle_rate', 0),
            'avg_capital_reuse_efficiency': eff.get('avg_capital_reuse_efficiency', 0),
            'blocked_signals_total': debug.get('blocked_signals_total', 0),
            'recycling_attempts': debug.get('recycling_attempts', 0),
            'recycling_denied_no_positions': debug.get('recycling_denied_reasons', {}).get('no_recyclable_positions', 0),
        }
        
    except subprocess.TimeoutExpired:
        return {
            'seed': seed,
            'stress_mode': stress_mode,
            'universe': universe,
            'status': 'TIMEOUT',
            'error': 'timeout after 300s',
        }
    except Exception as e:
        return {
            'seed': seed,
            'stress_mode': stress_mode,
            'universe': universe,
            'status': 'ERROR',
            'error': str(e),
        }


def main():
    print("=" * 70)
    print("RECYCLING ROBUSTNESS VALIDATION")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Stress modes: {STRESS_MODES}")
    print(f"Universes: {UNIVERSES}")
    print(f"Total runs: {len(SEEDS) * len(STRESS_MODES) * len(UNIVERSES)}")
    print("-" * 70)
    
    results = []
    total = len(SEEDS) * len(STRESS_MODES) * len(UNIVERSES)
    current = 0
    
    for universe in UNIVERSES:
        for stress_mode in STRESS_MODES:
            for seed in SEEDS:
                current += 1
                print(f"[{current}/{total}] seed={seed}, stress={stress_mode}, universe={universe}...", end=' ', flush=True)
                
                result = run_backtest(seed, stress_mode, universe)
                results.append(result)
                
                if result['status'] == 'OK':
                    events = result.get('recycle_events_count', 0)
                    edge = result.get('avg_swap_edge_r', 0)
                    false_rate = result.get('false_recycle_rate', 0)
                    print(f"events={events}, edge={edge:+.3f}R, false={false_rate:.1f}%")
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")
    
    df = pd.DataFrame(results)
    output_path = Path('outputs/recycling_robustness_matrix.csv')
    df.to_csv(output_path, index=False)
    print("-" * 70)
    print(f"Results saved to: {output_path}")
    print()
    
    ok_runs = df[df['status'] == 'OK']
    if len(ok_runs) == 0:
        print("ERROR: No successful runs!")
        return 1
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful runs: {len(ok_runs)}/{len(df)}")
    print()
    
    print("WORST FALSE RECYCLE RATES (Top 5):")
    worst_false = ok_runs.nlargest(5, 'false_recycle_rate')[['seed', 'stress_mode', 'universe', 'recycle_events_count', 'false_recycle_rate', 'avg_swap_edge_r']]
    print(worst_false.to_string(index=False))
    print()
    
    print("WORST SWAP EDGE (Bottom 5):")
    worst_edge = ok_runs.nsmallest(5, 'avg_swap_edge_r')[['seed', 'stress_mode', 'universe', 'recycle_events_count', 'avg_swap_edge_r', 'false_recycle_rate']]
    print(worst_edge.to_string(index=False))
    print()
    
    failures = []
    for _, row in ok_runs.iterrows():
        if row['recycle_events_count'] > 0:
            if row['false_recycle_rate'] > 40:
                failures.append(f"seed={row['seed']}, stress={row['stress_mode']}: false_rate={row['false_recycle_rate']:.1f}% > 40%")
            if row['avg_swap_edge_r'] < 0:
                failures.append(f"seed={row['seed']}, stress={row['stress_mode']}: avg_swap_edge={row['avg_swap_edge_r']:.3f}R < 0")
    
    if failures:
        print("=" * 70)
        print("QUALITY THRESHOLD FAILURES:")
        print("=" * 70)
        for f in failures:
            print(f"  FAIL: {f}")
        print()
        return 1
    else:
        print("=" * 70)
        print("ALL QUALITY THRESHOLDS PASSED!")
        print("  - false_recycle_rate <= 40% for all runs with recycling events")
        print("  - avg_swap_edge_r >= 0.00R for all runs with recycling events")
        print("=" * 70)
        return 0


if __name__ == '__main__':
    sys.exit(main())
