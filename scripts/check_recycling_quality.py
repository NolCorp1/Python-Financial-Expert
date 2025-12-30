#!/usr/bin/env python3
"""
Task 27 Part B: Recycling Quality Regression Checker

Loads outputs/metrics.json and asserts minimum quality bars for recycling.

Usage:
    python -m scripts.check_recycling_quality

Exit codes:
    0 = PASS (all assertions met)
    1 = FAIL (quality thresholds violated)
"""

import json
import sys
from pathlib import Path


FALSE_RECYCLE_RATE_THRESHOLD = 40.0
AVG_SWAP_EDGE_R_THRESHOLD = 0.00


def main():
    metrics_path = Path('outputs/metrics.json')
    
    if not metrics_path.exists():
        print("ERROR: outputs/metrics.json not found")
        print("Run a backtest first with --backtest-v2")
        return 1
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    eff = metrics.get('recycling_effectiveness', {})
    
    recycle_events = eff.get('recycle_events_count', 0)
    false_recycle_rate = eff.get('false_recycle_rate', 0)
    avg_swap_edge_r = eff.get('avg_swap_edge_r', 0)
    
    print("=" * 60)
    print("RECYCLING QUALITY CHECK")
    print("=" * 60)
    print(f"Recycle events:      {recycle_events}")
    print(f"False recycle rate:  {false_recycle_rate:.1f}% (threshold: <={FALSE_RECYCLE_RATE_THRESHOLD}%)")
    print(f"Avg swap edge (R):   {avg_swap_edge_r:+.3f} (threshold: >={AVG_SWAP_EDGE_R_THRESHOLD})")
    print("-" * 60)
    
    passed = True
    
    if recycle_events == 0:
        print("WARN: No recycling events detected (not a failure, but check config)")
        print("      This may be expected if no positions were recyclable")
    else:
        if false_recycle_rate > FALSE_RECYCLE_RATE_THRESHOLD:
            print(f"FAIL: false_recycle_rate {false_recycle_rate:.1f}% > {FALSE_RECYCLE_RATE_THRESHOLD}%")
            passed = False
        else:
            print(f"PASS: false_recycle_rate {false_recycle_rate:.1f}% <= {FALSE_RECYCLE_RATE_THRESHOLD}%")
        
        if avg_swap_edge_r < AVG_SWAP_EDGE_R_THRESHOLD:
            print(f"FAIL: avg_swap_edge_r {avg_swap_edge_r:+.3f} < {AVG_SWAP_EDGE_R_THRESHOLD}")
            passed = False
        else:
            print(f"PASS: avg_swap_edge_r {avg_swap_edge_r:+.3f} >= {AVG_SWAP_EDGE_R_THRESHOLD}")
    
    print("-" * 60)
    
    if passed:
        print("RESULT: PASS")
        return 0
    else:
        print("RESULT: FAIL")
        return 1


if __name__ == '__main__':
    sys.exit(main())
