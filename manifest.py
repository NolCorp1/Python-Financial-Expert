"""
Run Manifest & Reproducibility Module.

Task 14: Generate comprehensive run manifests for reproducibility and debugging.

Features:
- Unique run_id generation (timestamp + hash)
- Complete provenance tracking (versions, params, symbols)
- Price cache integrity reporting
- Run comparison utilities
"""

import os
import sys
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


def generate_run_id() -> str:
    """Generate unique run_id: timestamp + short hash."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_input = f"{timestamp}_{os.getpid()}_{id(object())}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"{timestamp}_{short_hash}"


def get_git_commit_hash() -> str:
    """Get current git commit hash, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def get_version_info() -> Dict[str, str]:
    """Collect all relevant package versions."""
    versions = {
        'python': sys.version.split()[0],
        'platform': f"{platform.system()} {platform.release()}",
    }
    
    try:
        import yfinance
        versions['yfinance'] = yfinance.__version__
    except:
        versions['yfinance'] = 'unknown'
    
    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except:
        versions['pandas'] = 'unknown'
    
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except:
        versions['numpy'] = 'unknown'
    
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except:
        versions['scipy'] = 'unknown'
    
    try:
        import pyarrow
        versions['pyarrow'] = pyarrow.__version__
    except:
        versions['pyarrow'] = 'unknown'
    
    return versions


def _convert_to_json_serializable(obj):
    """Convert numpy types and other non-JSON types to native Python."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        return obj


def create_manifest(
    run_id: str,
    mode: str,
    command_line: str,
    symbols_used: List[str],
    symbols_requested: int,
    config: Dict[str, Any],
    cache_stats: Optional[Dict[str, Any]] = None,
    date_range: Optional[tuple] = None,
    universe_type: str = "unknown",
    liquidity_config: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive run manifest.
    
    Args:
        run_id: Unique run identifier
        mode: 'optimize', 'validate', or 'backtest'
        command_line: Original command line used
        symbols_used: List of symbols actually used
        symbols_requested: Number of symbols requested
        config: Strategy configuration applied
        cache_stats: Price cache hit/miss stats
        date_range: Tuple of (start_date, end_date)
        universe_type: 'demo', 'nasdaq', 'custom'
        liquidity_config: Liquidity filter settings
        extra_info: Any additional run-specific info
    
    Returns:
        Complete manifest dictionary
    """
    config = _convert_to_json_serializable(config)
    cache_stats = _convert_to_json_serializable(cache_stats) if cache_stats else {}
    liquidity_config = _convert_to_json_serializable(liquidity_config) if liquidity_config else {}
    extra_info = _convert_to_json_serializable(extra_info) if extra_info else None
    
    manifest = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'command_line': command_line,
        'git_commit': get_git_commit_hash(),
        'versions': get_version_info(),
        
        'universe': {
            'type': universe_type,
            'symbols_requested': symbols_requested,
            'symbols_used': len(symbols_used),
        },
        
        'date_range': {
            'start': str(date_range[0]) if date_range else None,
            'end': str(date_range[1]) if date_range else None,
        },
        
        'liquidity_config': liquidity_config or {},
        
        'cache_stats': cache_stats or {},
        
        'strategy_config': config,
    }
    
    if extra_info:
        manifest['extra'] = extra_info
    
    return manifest


def save_manifest(
    manifest: Dict[str, Any],
    symbols_used: List[str],
    output_dir: str = "outputs/manifests"
) -> tuple:
    """
    Save manifest JSON and symbols list to files.
    
    Args:
        manifest: Manifest dictionary
        symbols_used: List of symbols
        output_dir: Directory for manifest files
    
    Returns:
        Tuple of (manifest_path, symbols_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    run_id = manifest['run_id']
    
    manifest_path = Path(output_dir) / f"{run_id}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    symbols_path = Path(output_dir) / f"{run_id}_symbols.txt"
    with open(symbols_path, 'w') as f:
        f.write(f"# Symbols for run: {run_id}\n")
        f.write(f"# Generated: {manifest['timestamp']}\n")
        f.write(f"# Mode: {manifest['mode']}\n")
        f.write(f"# Count: {len(symbols_used)}\n")
        f.write("\n")
        for symbol in sorted(symbols_used):
            f.write(f"{symbol}\n")
    
    return str(manifest_path), str(symbols_path)


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load a manifest from JSON file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def compare_manifests(
    manifest_a: Dict[str, Any],
    manifest_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two manifests and return differences.
    
    Returns:
        Dictionary with config_diff, symbols_diff, and other changes
    """
    diff = {
        'run_a': manifest_a.get('run_id', 'unknown'),
        'run_b': manifest_b.get('run_id', 'unknown'),
        'timestamp_a': manifest_a.get('timestamp'),
        'timestamp_b': manifest_b.get('timestamp'),
        'config_diff': {},
        'symbols_diff': {},
        'version_diff': {},
    }
    
    config_a = manifest_a.get('strategy_config', {})
    config_b = manifest_b.get('strategy_config', {})
    
    all_keys = set(config_a.keys()) | set(config_b.keys())
    for key in all_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            diff['config_diff'][key] = {'run_a': val_a, 'run_b': val_b}
    
    versions_a = manifest_a.get('versions', {})
    versions_b = manifest_b.get('versions', {})
    for key in set(versions_a.keys()) | set(versions_b.keys()):
        if versions_a.get(key) != versions_b.get(key):
            diff['version_diff'][key] = {
                'run_a': versions_a.get(key),
                'run_b': versions_b.get(key)
            }
    
    universe_a = manifest_a.get('universe', {})
    universe_b = manifest_b.get('universe', {})
    diff['symbols_diff'] = {
        'count_a': universe_a.get('symbols_used', 0),
        'count_b': universe_b.get('symbols_used', 0),
    }
    
    return diff


def save_diff_report(
    diff: Dict[str, Any],
    metrics_diff: Optional[Dict[str, Any]] = None,
    output_dir: str = "outputs"
) -> str:
    """
    Save a comparison report to file.
    
    Args:
        diff: Output from compare_manifests
        metrics_diff: Optional metrics comparison (return, dd, trades)
        output_dir: Output directory
    
    Returns:
        Path to saved diff file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    run_a = diff['run_a'][:16] if len(diff['run_a']) > 16 else diff['run_a']
    run_b = diff['run_b'][:16] if len(diff['run_b']) > 16 else diff['run_b']
    filename = f"diff_{run_a}_{run_b}.txt"
    filepath = Path(output_dir) / filename
    
    lines = []
    lines.append("=" * 70)
    lines.append("RUN COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append(f"\nRun A: {diff['run_a']}")
    lines.append(f"       {diff['timestamp_a']}")
    lines.append(f"\nRun B: {diff['run_b']}")
    lines.append(f"       {diff['timestamp_b']}")
    
    lines.append("\n" + "-" * 70)
    lines.append("SYMBOL COUNTS")
    lines.append("-" * 70)
    sd = diff['symbols_diff']
    lines.append(f"  Run A: {sd.get('count_a', 'N/A')} symbols")
    lines.append(f"  Run B: {sd.get('count_b', 'N/A')} symbols")
    
    if diff['config_diff']:
        lines.append("\n" + "-" * 70)
        lines.append("CONFIG DIFFERENCES")
        lines.append("-" * 70)
        for key, vals in diff['config_diff'].items():
            lines.append(f"  {key}:")
            lines.append(f"    Run A: {vals['run_a']}")
            lines.append(f"    Run B: {vals['run_b']}")
    else:
        lines.append("\n  No config differences.")
    
    if diff['version_diff']:
        lines.append("\n" + "-" * 70)
        lines.append("VERSION DIFFERENCES")
        lines.append("-" * 70)
        for key, vals in diff['version_diff'].items():
            lines.append(f"  {key}: {vals['run_a']} -> {vals['run_b']}")
    
    if metrics_diff:
        lines.append("\n" + "-" * 70)
        lines.append("METRICS DIFFERENCES")
        lines.append("-" * 70)
        for key, vals in metrics_diff.items():
            lines.append(f"  {key}:")
            lines.append(f"    Run A: {vals.get('run_a', 'N/A')}")
            lines.append(f"    Run B: {vals.get('run_b', 'N/A')}")
            if 'delta' in vals:
                lines.append(f"    Delta: {vals['delta']}")
    
    lines.append("\n" + "=" * 70)
    
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    
    return str(filepath)


def diff_runs(
    manifest_path_a: str,
    manifest_path_b: str,
    output_dir: str = "outputs"
) -> str:
    """
    Compare two runs by their manifest files and save report.
    
    Args:
        manifest_path_a: Path to first manifest
        manifest_path_b: Path to second manifest
        output_dir: Output directory
    
    Returns:
        Path to diff report
    """
    manifest_a = load_manifest(manifest_path_a)
    manifest_b = load_manifest(manifest_path_b)
    
    diff = compare_manifests(manifest_a, manifest_b)
    
    return save_diff_report(diff, output_dir=output_dir)


def print_manifest_summary(manifest: Dict[str, Any]) -> None:
    """Print a human-readable manifest summary."""
    print("\n" + "=" * 60)
    print("RUN MANIFEST")
    print("=" * 60)
    print(f"Run ID: {manifest.get('run_id', 'unknown')}")
    print(f"Mode: {manifest.get('mode', 'unknown')}")
    print(f"Timestamp: {manifest.get('timestamp', 'unknown')}")
    print(f"Git commit: {manifest.get('git_commit', 'unknown')}")
    
    universe = manifest.get('universe', {})
    print(f"\nUniverse: {universe.get('type', 'unknown')}")
    print(f"  Requested: {universe.get('symbols_requested', 'N/A')}")
    print(f"  Used: {universe.get('symbols_used', 'N/A')}")
    
    cache = manifest.get('cache_stats', {})
    if cache:
        print(f"\nCache: {cache.get('hits', 0)} hits, {cache.get('misses', 0)} misses")
    
    versions = manifest.get('versions', {})
    print(f"\nVersions: Python {versions.get('python', '?')}, "
          f"pandas {versions.get('pandas', '?')}, "
          f"yfinance {versions.get('yfinance', '?')}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Manifest utilities')
    parser.add_argument('--diff', nargs=2, metavar=('MANIFEST_A', 'MANIFEST_B'),
                       help='Compare two manifests')
    parser.add_argument('--show', metavar='MANIFEST',
                       help='Show manifest summary')
    parser.add_argument('--list', action='store_true',
                       help='List all manifests in outputs/manifests/')
    
    args = parser.parse_args()
    
    if args.diff:
        diff_path = diff_runs(args.diff[0], args.diff[1])
        print(f"Diff saved to: {diff_path}")
    
    elif args.show:
        manifest = load_manifest(args.show)
        print_manifest_summary(manifest)
    
    elif args.list:
        manifest_dir = Path('outputs/manifests')
        if manifest_dir.exists():
            manifests = sorted(manifest_dir.glob('*_manifest.json'))
            print(f"\nFound {len(manifests)} manifests:\n")
            for m in manifests[-10:]:
                print(f"  {m.name}")
        else:
            print("No manifests directory found.")
