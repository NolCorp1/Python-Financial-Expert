"""
Price Cache Module for Walk-Forward Optimization.

Task 12B: Download price data once and cache as parquet for fast reuse.

Key features:
- Batch download using yfinance.download with chunking
- Parquet persistence in data/price_cache/
- Single download per walk-forward run (no per-window re-downloads)
- CLI-configurable batch size and cache directory
"""
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm


def compute_required_date_range(
    train_bars: int = 504,
    test_bars: int = 126,
    step_bars: int = 126,
    n_windows: int = 5,
    buffer_bars: int = 50
) -> Tuple[datetime, datetime]:
    """
    Compute the full date range needed for walk-forward optimization.
    
    Args:
        train_bars: Training period bars
        test_bars: Testing period bars
        step_bars: Step size between windows
        n_windows: Estimated number of windows
        buffer_bars: Extra buffer for indicators/warmup
        
    Returns:
        Tuple of (start_date, end_date)
    """
    total_bars_needed = train_bars + test_bars + (step_bars * (n_windows - 1)) + buffer_bars
    
    calendar_days = int(total_bars_needed * 365 / 252 * 1.1) + 30
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=calendar_days)
    
    return start_date, end_date


def get_cache_path(symbol: str, cache_dir: str = "data/price_cache") -> Path:
    """Get the parquet cache path for a symbol."""
    return Path(cache_dir) / f"{symbol}.parquet"


def is_cache_valid(
    symbol: str,
    cache_dir: str,
    required_start: datetime,
    required_end: datetime,
    max_age_hours: int = 24
) -> bool:
    """
    Check if cached data is valid for the required date range.
    
    Returns True if:
    - Cache file exists
    - Cache is less than max_age_hours old
    - Cache covers the required date range
    """
    cache_path = get_cache_path(symbol, cache_dir)
    
    if not cache_path.exists():
        return False
    
    file_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
    if file_age_hours > max_age_hours:
        return False
    
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return False
        
        cache_start = pd.Timestamp(df.index.min())
        cache_end = pd.Timestamp(df.index.max())
        if cache_start.tzinfo is not None:
            cache_start = cache_start.tz_localize(None)
        if cache_end.tzinfo is not None:
            cache_end = cache_end.tz_localize(None)
        
        required_start_ts = pd.Timestamp(required_start)
        if required_start_ts.tzinfo is not None:
            required_start_ts = required_start_ts.tz_localize(None)
        
        if cache_start > required_start_ts + pd.Timedelta(days=60):
            return False
        
        if len(df) < 100:
            return False
        
        return True
        
    except Exception:
        return False


def load_from_cache(symbol: str, cache_dir: str) -> Optional[pd.DataFrame]:
    """Load price data from parquet cache."""
    cache_path = get_cache_path(symbol, cache_dir)
    
    if not cache_path.exists():
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Warning: Failed to load cache for {symbol}: {e}")
        return None


def save_to_cache(symbol: str, df: pd.DataFrame, cache_dir: str) -> bool:
    """Save price data to parquet cache (uses atomic write internally)."""
    return save_to_cache_atomic(symbol, df, cache_dir)


def batch_download_yfinance(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    batch_size: int = 50,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Download price data for multiple symbols using yfinance batching.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        batch_size: Number of symbols per batch (50-75 recommended)
        verbose: Show progress
        
    Returns:
        Dict of symbol -> OHLCV DataFrame
    """
    result = {}
    
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    iterator = tqdm(batches, desc="Downloading batches") if verbose else batches
    
    for batch in iterator:
        try:
            tickers_str = " ".join(batch)
            
            data = yf.download(
                tickers_str,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=False,
                threads=True,
                progress=False
            )
            
            if data.empty:
                continue
            
            if len(batch) == 1:
                symbol = batch[0]
                if not data.empty and len(data) > 50:
                    df = data.copy()
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    if 'Close' in df.columns:
                        result[symbol] = df
            else:
                for symbol in batch:
                    try:
                        if symbol in data.columns.get_level_values(0):
                            df = data[symbol].copy()
                            df = df.dropna(how='all')
                            if not df.empty and len(df) > 50 and 'Close' in df.columns:
                                result[symbol] = df
                    except Exception:
                        pass
                        
        except Exception as e:
            if verbose:
                print(f"Batch download error: {e}")
            
            for symbol in batch:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    if df is not None and len(df) > 50:
                        df = df.rename(columns={'Stock Splits': 'Stock_Splits'})
                        result[symbol] = df
                except Exception:
                    pass
    
    return result


def preload_price_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True,
    cache_dir: str = "data/price_cache",
    batch_size: int = 50,
    max_cache_age_hours: int = 24,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Preload price data for walk-forward optimization.
    
    This is the main entry point for Task 12B. It:
    1. Checks parquet cache for each symbol
    2. Downloads missing/stale symbols in batches
    3. Saves new downloads to cache
    4. Returns complete price data dict
    
    Args:
        symbols: List of ticker symbols
        start_date: Required start date
        end_date: Required end date
        use_cache: Whether to use parquet cache
        cache_dir: Directory for parquet files
        batch_size: Symbols per yfinance batch
        max_cache_age_hours: Max cache age before refresh
        verbose: Show progress
        
    Returns:
        Dict of symbol -> OHLCV DataFrame
    """
    if verbose:
        print(f"\nPreloading price data for {len(symbols)} symbols...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        if use_cache:
            print(f"Cache dir: {cache_dir}")
    
    result = {}
    symbols_to_download = []
    
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        for symbol in symbols:
            if is_cache_valid(symbol, cache_dir, start_date, end_date, max_cache_age_hours):
                df = load_from_cache(symbol, cache_dir)
                if df is not None:
                    result[symbol] = df
                else:
                    symbols_to_download.append(symbol)
            else:
                symbols_to_download.append(symbol)
        
        if verbose:
            print(f"Cache hits: {len(result)}, need download: {len(symbols_to_download)}")
    else:
        symbols_to_download = list(symbols)
    
    if symbols_to_download:
        downloaded = batch_download_yfinance(
            symbols_to_download,
            start_date,
            end_date,
            batch_size=batch_size,
            verbose=verbose
        )
        
        for symbol, df in downloaded.items():
            result[symbol] = df
            
            if use_cache:
                save_to_cache(symbol, df, cache_dir)
    
    if verbose:
        print(f"Total symbols loaded: {len(result)}")
    
    return result


def clear_price_cache(cache_dir: str = "data/price_cache", symbols: Optional[List[str]] = None):
    """
    Clear price cache.
    
    Args:
        cache_dir: Cache directory
        symbols: If provided, only clear these symbols. Otherwise clear all.
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return
    
    if symbols:
        for symbol in symbols:
            parquet_path = cache_path / f"{symbol}.parquet"
            if parquet_path.exists():
                parquet_path.unlink()
    else:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Cleared price cache: {cache_dir}")


def get_cache_stats(cache_dir: str = "data/price_cache") -> Dict:
    """Get statistics about the price cache."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {'exists': False, 'count': 0, 'size_mb': 0}
    
    parquet_files = list(cache_path.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    
    oldest_file = None
    newest_file = None
    if parquet_files:
        mtimes = [(f, f.stat().st_mtime) for f in parquet_files]
        oldest_file = min(mtimes, key=lambda x: x[1])[0].stem
        newest_file = max(mtimes, key=lambda x: x[1])[0].stem
    
    return {
        'exists': True,
        'count': len(parquet_files),
        'size_mb': total_size / (1024 * 1024),
        'oldest': oldest_file,
        'newest': newest_file,
    }


def validate_price_data(df: pd.DataFrame, symbol: str = "") -> Dict:
    """
    Validate price data integrity.
    
    Checks for:
    - Required OHLC columns exist and are numeric
    - NaN values in OHLC
    - Zero or negative prices
    - Large gaps in dates (>5 consecutive business days)
    - Duplicate dates
    
    Returns:
        Dict with validation results and issues found
    """
    issues = []
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return {'valid': False, 'issues': issues, 'nan_count': 0, 'duplicate_dates': 0}
    
    nan_count = df[required_cols].isna().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN values in OHLC: {nan_count}")
    
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Non-numeric column: {col}")
    
    negative_prices = (df[required_cols] <= 0).sum().sum()
    if negative_prices > 0:
        issues.append(f"Zero/negative prices: {negative_prices}")
    
    duplicate_dates = df.index.duplicated().sum()
    if duplicate_dates > 0:
        issues.append(f"Duplicate dates: {duplicate_dates}")
    
    if len(df) > 1:
        dates = pd.DatetimeIndex(df.index)
        business_days = pd.bdate_range(dates.min(), dates.max())
        missing_days = len(business_days) - len(dates)
        
        if len(dates) > 2:
            date_diffs = dates[1:] - dates[:-1]
            max_gap_days = date_diffs.max().days if len(date_diffs) > 0 else 0
            if max_gap_days > 10:
                issues.append(f"Large gap in dates: {max_gap_days} days")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'nan_count': nan_count,
        'duplicate_dates': duplicate_dates,
        'rows': len(df),
        'start': str(df.index.min()) if len(df) > 0 else None,
        'end': str(df.index.max()) if len(df) > 0 else None,
    }


def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price data for consistency.
    
    - Strip timezone info (convert to naive UTC)
    - Sort by date
    - Remove duplicate dates (keep first)
    - Ensure sorted ascending
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    if df.index.tzinfo is not None:
        df.index = df.index.tz_localize(None)
    
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]
    
    df = df.sort_index()
    
    return df


def save_to_cache_atomic(symbol: str, df: pd.DataFrame, cache_dir: str) -> bool:
    """
    Save price data to parquet cache with atomic write (temp file + rename).
    
    This prevents corruption if the process is interrupted during write.
    """
    if df is None or df.empty:
        return False
    
    cache_path = get_cache_path(symbol, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = cache_path.with_suffix('.parquet.tmp')
    
    try:
        df = normalize_price_data(df)
        
        df.to_parquet(temp_path)
        
        temp_path.rename(cache_path)
        return True
    except Exception as e:
        print(f"Warning: Failed to save cache for {symbol}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def repair_cache_entry(
    symbol: str,
    cache_dir: str,
    start_date: datetime,
    end_date: datetime,
    verbose: bool = True
) -> bool:
    """
    Repair a corrupt or invalid cache entry by re-downloading.
    
    Returns True if repair successful.
    """
    if verbose:
        print(f"Repairing cache for {symbol}...")
    
    cache_path = get_cache_path(symbol, cache_dir)
    if cache_path.exists():
        try:
            cache_path.unlink()
        except:
            pass
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df is not None and len(df) > 50:
            df = normalize_price_data(df)
            return save_to_cache_atomic(symbol, df, cache_dir)
    except Exception as e:
        if verbose:
            print(f"  Failed to repair {symbol}: {e}")
    
    return False


def generate_cache_report(
    symbols: List[str],
    cache_dir: str = "data/price_cache",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    auto_repair: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate integrity report for cached price data.
    
    Args:
        symbols: List of symbols to check
        cache_dir: Cache directory
        start_date: Required start date (for repair)
        end_date: Required end date (for repair)
        auto_repair: Automatically repair invalid entries
        verbose: Show progress
    
    Returns:
        DataFrame with columns: symbol, cached, downloaded, rows, start, end, 
                                nan_count, duplicate_dates, status
    """
    reports = []
    
    iterator = tqdm(symbols, desc="Checking cache integrity") if verbose else symbols
    
    for symbol in iterator:
        cache_path = get_cache_path(symbol, cache_dir)
        cached = cache_path.exists()
        
        if not cached:
            reports.append({
                'symbol': symbol,
                'cached': False,
                'downloaded': False,
                'rows': 0,
                'start': None,
                'end': None,
                'nan_count': 0,
                'duplicate_dates': 0,
                'status': 'missing',
            })
            continue
        
        try:
            df = pd.read_parquet(cache_path)
            validation = validate_price_data(df, symbol)
            
            status = 'ok' if validation['valid'] else 'invalid'
            
            if not validation['valid'] and auto_repair and start_date and end_date:
                if repair_cache_entry(symbol, cache_dir, start_date, end_date, verbose=False):
                    status = 'repaired'
                else:
                    status = 'repair_failed'
            
            reports.append({
                'symbol': symbol,
                'cached': True,
                'downloaded': False,
                'rows': validation.get('rows', 0),
                'start': validation.get('start'),
                'end': validation.get('end'),
                'nan_count': validation.get('nan_count', 0),
                'duplicate_dates': validation.get('duplicate_dates', 0),
                'status': status,
            })
            
        except Exception as e:
            status = 'corrupt'
            
            if auto_repair and start_date and end_date:
                if repair_cache_entry(symbol, cache_dir, start_date, end_date, verbose=False):
                    status = 'repaired'
                else:
                    status = 'repair_failed'
            
            reports.append({
                'symbol': symbol,
                'cached': True,
                'downloaded': False,
                'rows': 0,
                'start': None,
                'end': None,
                'nan_count': 0,
                'duplicate_dates': 0,
                'status': status,
            })
    
    report_df = pd.DataFrame(reports)
    
    os.makedirs('outputs', exist_ok=True)
    report_df.to_csv('outputs/price_cache_report.csv', index=False)
    
    if verbose:
        n_ok = len(report_df[report_df['status'] == 'ok'])
        n_invalid = len(report_df[report_df['status'].isin(['invalid', 'corrupt'])])
        n_repaired = len(report_df[report_df['status'] == 'repaired'])
        n_missing = len(report_df[report_df['status'] == 'missing'])
        print(f"\nCache report: {n_ok} ok, {n_invalid} invalid, {n_repaired} repaired, {n_missing} missing")
        print(f"Report saved to: outputs/price_cache_report.csv")
    
    return report_df
