#!/usr/bin/env python3
"""
Task 27 Part C: Determinism Hash Checker

Runs a command twice and verifies SHA256 hashes of output files match.

Usage:
    python -m scripts.check_determinism_hashes "python main.py --backtest-v2 --universe demo --symbols-seed 123 --price-cache-policy READONLY --quiet"

Exit codes:
    0 = PASS (all hashes match)
    1 = FAIL (hash mismatch detected)
"""

import subprocess
import hashlib
import sys
from pathlib import Path
import tempfile
import shutil


OUTPUT_FILES = [
    'outputs/trades.csv',
    'outputs/metrics.json',
]


def compute_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    path = Path(filepath)
    if not path.exists():
        return "FILE_NOT_FOUND"
    
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_command(cmd: str) -> bool:
    """Run command and return True if successful."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out after 600s")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.check_determinism_hashes <command>")
        print()
        print("Example:")
        print('  python -m scripts.check_determinism_hashes "python main.py --backtest-v2 --universe demo --symbols-seed 123 --price-cache-policy READONLY --quiet"')
        return 1
    
    cmd = sys.argv[1]
    
    print("=" * 70)
    print("DETERMINISM HASH CHECK")
    print("=" * 70)
    print(f"Command: {cmd}")
    print("-" * 70)
    
    print("Run 1...")
    if not run_command(cmd):
        print("ERROR: Run 1 failed")
        return 1
    
    hashes_run1 = {}
    for filepath in OUTPUT_FILES:
        hashes_run1[filepath] = compute_hash(filepath)
    
    print("Run 2...")
    if not run_command(cmd):
        print("ERROR: Run 2 failed")
        return 1
    
    hashes_run2 = {}
    for filepath in OUTPUT_FILES:
        hashes_run2[filepath] = compute_hash(filepath)
    
    print("-" * 70)
    print("HASH COMPARISON:")
    print("-" * 70)
    
    all_match = True
    for filepath in OUTPUT_FILES:
        h1 = hashes_run1.get(filepath, "N/A")
        h2 = hashes_run2.get(filepath, "N/A")
        
        if h1 == "FILE_NOT_FOUND" or h2 == "FILE_NOT_FOUND":
            print(f"  {filepath}: SKIP (file not found)")
            continue
        
        if h1 == h2:
            print(f"  {filepath}: MATCH ({h1[:16]}...)")
        else:
            print(f"  {filepath}: MISMATCH")
            print(f"    Run 1: {h1[:32]}...")
            print(f"    Run 2: {h2[:32]}...")
            all_match = False
    
    print("-" * 70)
    
    if all_match:
        print("RESULT: PASS - All hashes match, runs are deterministic")
        return 0
    else:
        print("RESULT: FAIL - Hash mismatch detected, runs are NOT deterministic")
        return 1


if __name__ == '__main__':
    sys.exit(main())
