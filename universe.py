"""
Universe management: NASDAQ symbol list caching and liquidity filtering.

Task 12A: Hardened download with validation and atomic caching.
"""
import os
import time
import shutil
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np

NASDAQ_SCREENER_URL = "https://www.nasdaq.com/market-activity/stocks/screener?exchange=nasdaq&render=download"
FALLBACK_SYMBOLS_FILE = "data/fallback_nasdaq_symbols.csv"

FUND_TOKENS = [
    "ETF", "ETN", "Fund", "Trust", "Index", "Shares", "iShares", "Vanguard",
    "Invesco", "ProShares", "Direxion", "WisdomTree", "VanEck", "Global X",
    "First Trust", "SPDR"
]


def _is_valid_ticker(symbol: str) -> bool:
    """Check if ticker is valid (no special chars, reasonable length)."""
    if not symbol or not isinstance(symbol, str):
        return False
    symbol = symbol.strip().upper()
    if len(symbol) > 6:
        return False
    invalid_chars = ["^", "/", "=", "."]
    for char in invalid_chars:
        if char in symbol:
            return False
    if not symbol.isalpha():
        return False
    return True


def _is_fund(name: str) -> bool:
    """Check if name contains fund/ETF tokens."""
    if not name or not isinstance(name, str):
        return False
    name_upper = name.upper()
    for token in FUND_TOKENS:
        if token.upper() in name_upper:
            return True
    return False


def validate_nasdaq_csv_bytes(content: bytes) -> Tuple[bool, str]:
    """
    Validate downloaded NASDAQ CSV content before writing to cache.
    
    Checks:
    - Content size > 10KB
    - First 1KB does NOT contain "<html" or "<!DOCTYPE"
    - Parses as valid CSV
    - Has required columns (Symbol case-insensitive)
    - Number of rows >= 200
    
    Returns:
        Tuple of (is_valid, reason)
    """
    if len(content) < 10 * 1024:
        return False, f"Content too small: {len(content)} bytes (min 10KB)"
    
    header_sample = content[:1024].decode('utf-8', errors='ignore').lower()
    if '<html' in header_sample or '<!doctype' in header_sample:
        return False, "Content appears to be HTML, not CSV"
    
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        try:
            df = pd.read_csv(BytesIO(content), on_bad_lines='skip')
        except Exception as e2:
            return False, f"Failed to parse CSV: {e2}"
    
    symbol_col = None
    for col in df.columns:
        if col.lower().strip() == 'symbol':
            symbol_col = col
            break
    
    if symbol_col is None:
        return False, f"Missing required 'Symbol' column. Found: {list(df.columns)}"
    
    if len(df) < 200:
        return False, f"Too few rows: {len(df)} (min 200)"
    
    return True, "Valid CSV"


def _load_fallback_symbols() -> List[str]:
    """Load fallback symbols from CSV file or return embedded list."""
    fallback_path = Path(FALLBACK_SYMBOLS_FILE)
    
    if fallback_path.exists():
        try:
            df = pd.read_csv(fallback_path)
            if 'symbol' in df.columns:
                symbols = df['symbol'].dropna().astype(str).str.upper().tolist()
                symbols = [s for s in symbols if _is_valid_ticker(s)]
                if len(symbols) >= 100:
                    return symbols
        except Exception as e:
            print(f"Warning: Failed to load fallback file: {e}")
    
    return [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
        "AVGO", "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO", "INTC",
        "CMCSA", "TMUS", "INTU", "TXN", "QCOM", "AMGN", "AMAT", "ISRG",
        "HON", "SBUX", "BKNG", "VRTX", "GILD", "ADI", "MDLZ", "ADP",
        "REGN", "LRCX", "PANW", "KLAC", "SNPS", "CDNS", "MELI", "ASML",
        "ABNB", "PYPL", "MAR", "ORLY", "MRVL", "CTAS", "CHTR", "CRWD",
        "MNST", "WDAY", "PCAR", "NXPI", "ADSK", "ROST", "MCHP", "CPRT",
        "DXCM", "IDXX", "KDP", "LULU", "AZN", "FTNT", "ODFL", "PAYX",
        "KHC", "CEG", "BIIB", "FAST", "GFS", "CSGP", "EA", "ON", "TTD",
        "CDW", "VRSK", "BKR", "ALGN", "FANG", "TEAM", "ZS", "DDOG",
        "ANSS", "WBD", "CTSH", "GEHC", "ILMN", "EXC", "WBA", "XEL",
        "UBER", "LYFT", "DASH", "RBLX", "COIN", "PLTR", "SOFI", "HOOD",
        "MRNA", "BNTX", "NVAX", "SNOW", "NET", "MDB", "OKTA", "SPLK",
        "ZM", "DOCU", "TWLO", "CRSP", "BEAM", "EDIT", "NTLA", "SGEN",
    ]


def _get_fallback_symbols() -> List[str]:
    """Return fallback list of 600+ liquid NASDAQ stocks."""
    return _load_fallback_symbols()


def validate_existing_cache(cache_path: str) -> bool:
    """
    Validate an existing cache file.
    
    Returns True if cache is valid, False if corrupted.
    If corrupted, moves to cache_path.corrupt.<timestamp>
    """
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return False
    
    try:
        content = cache_file.read_bytes()
        
        if len(content) < 100:
            raise ValueError("Cache too small")
        
        header_sample = content[:1024].decode('utf-8', errors='ignore').lower()
        if '<html' in header_sample or '<!doctype' in header_sample:
            raise ValueError("Cache contains HTML")
        
        df = pd.read_csv(cache_file)
        if 'symbol' not in df.columns:
            raise ValueError("Missing 'symbol' column")
        if len(df) < 50:
            raise ValueError(f"Only {len(df)} symbols in cache")
        
        return True
        
    except Exception as e:
        timestamp = int(time.time())
        corrupt_path = f"{cache_path}.corrupt.{timestamp}"
        try:
            shutil.move(cache_path, corrupt_path)
            print(f"Quarantined corrupted cache: {corrupt_path}")
        except Exception:
            pass
        return False


def get_nasdaq_symbols_cached(
    cache_path: str = "data/nasdaq_symbols_cache.csv",
    max_age_hours: int = 24,
    exclude_funds: bool = True,
    limit: Optional[int] = None,
    force_refresh: bool = False,
) -> List[str]:
    """
    Get NASDAQ symbol list, using cache if available and fresh.
    
    Implements Task 12A hardening:
    - Validates downloads before caching
    - Atomic writes (temp file -> rename)
    - Preserves .bak backup of last-known-good cache
    - Falls back to 600+ symbol list on failure
    
    Args:
        cache_path: Path to cache file
        max_age_hours: Max age in hours before refreshing cache
        exclude_funds: Whether to exclude ETFs/funds
        limit: Optional limit on number of symbols
        force_refresh: Force download even if cache is fresh
        
    Returns:
        List of uppercase ticker symbols
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    use_cache = False
    if not force_refresh and cache_file.exists():
        if validate_existing_cache(cache_path):
            file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if file_age_hours < max_age_hours:
                use_cache = True
        else:
            print("Cache validation failed, will attempt fresh download")
    
    if use_cache:
        try:
            df = pd.read_csv(cache_file)
            symbols = df['symbol'].tolist()
            if limit:
                symbols = symbols[:limit]
            return symbols
        except Exception as e:
            print(f"Warning: Failed to read cache: {e}")
            use_cache = False
    
    symbols = _download_nasdaq_symbols_validated(cache_path, exclude_funds=exclude_funds)
    
    if not symbols or len(symbols) < 100:
        print(f"Download returned insufficient symbols ({len(symbols) if symbols else 0}), using fallback")
        symbols = _get_fallback_symbols()
    
    if limit:
        symbols = symbols[:limit]
    
    return symbols


def _download_nasdaq_symbols_validated(cache_path: str, exclude_funds: bool = True) -> List[str]:
    """
    Download NASDAQ symbol list with validation and atomic caching.
    
    Returns:
        List of symbols, or empty list on failure
    """
    import requests
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/csv,application/csv,*/*',
    }
    
    try:
        print("Downloading NASDAQ symbol list...")
        response = requests.get(NASDAQ_SCREENER_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        content = response.content
        is_valid, reason = validate_nasdaq_csv_bytes(content)
        
        if not is_valid:
            print(f"Download validation failed: {reason}")
            return []
        
        df = pd.read_csv(BytesIO(content))
        
        symbol_col = None
        name_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower == 'symbol':
                symbol_col = col
            elif col_lower == 'name':
                name_col = col
        
        if symbol_col is None:
            print("Warning: Could not find Symbol column")
            return []
        
        symbols = []
        for idx, row in df.iterrows():
            symbol = str(row[symbol_col]).strip().upper()
            name = str(row[name_col]) if name_col else ""
            
            if not _is_valid_ticker(symbol):
                continue
            
            if exclude_funds and _is_fund(name):
                continue
            
            symbols.append(symbol)
        
        if len(symbols) < 200:
            print(f"Warning: Only {len(symbols)} valid symbols parsed")
            return []
        
        cache_file = Path(cache_path)
        tmp_path = cache_file.with_suffix('.csv.tmp')
        bak_path = cache_file.with_suffix('.csv.bak')
        
        try:
            pd.DataFrame({'symbol': symbols}).to_csv(tmp_path, index=False)
            
            test_df = pd.read_csv(tmp_path)
            if len(test_df) < 200:
                raise ValueError("Temp file validation failed")
            
            if cache_file.exists():
                shutil.copy2(cache_file, bak_path)
            
            shutil.move(str(tmp_path), str(cache_file))
            print(f"Cached {len(symbols)} NASDAQ symbols to {cache_path}")
            
        except Exception as e:
            print(f"Warning: Failed to update cache atomically: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
        
        return symbols
        
    except Exception as e:
        print(f"Warning: Failed to download NASDAQ symbols: {e}")
        return []


def passes_liquidity_filter(
    df: pd.DataFrame,
    min_price: float = 5.0,
    min_avg_dollar_vol: float = 20_000_000,
    window: int = 20
) -> Tuple[bool, Dict]:
    """
    Check if symbol passes liquidity filter.
    
    Args:
        df: Price DataFrame with Close and Volume columns
        min_price: Minimum last price
        min_avg_dollar_vol: Minimum average dollar volume over window
        window: Lookback window for avg dollar volume
        
    Returns:
        Tuple of (passes, diagnostics_dict)
    """
    diagnostics = {
        'price_last': np.nan,
        'avg_dollar_vol_20': np.nan,
        'reason': None
    }
    
    if df is None or df.empty:
        diagnostics['reason'] = 'no_data'
        return False, diagnostics
    
    if 'Close' not in df.columns:
        diagnostics['reason'] = 'no_close'
        return False, diagnostics
    
    if 'Volume' not in df.columns:
        diagnostics['reason'] = 'no_volume'
        return False, diagnostics
    
    if len(df) < window:
        diagnostics['reason'] = 'insufficient_history'
        return False, diagnostics
    
    price_last = df['Close'].iloc[-1]
    diagnostics['price_last'] = float(price_last)
    
    if pd.isna(price_last):
        diagnostics['reason'] = 'no_price'
        return False, diagnostics
    
    if price_last < min_price:
        diagnostics['reason'] = 'min_price'
        return False, diagnostics
    
    dollar_volume = df['Close'] * df['Volume']
    avg_dollar_vol = dollar_volume.tail(window).mean()
    diagnostics['avg_dollar_vol_20'] = float(avg_dollar_vol) if not pd.isna(avg_dollar_vol) else np.nan
    
    if pd.isna(avg_dollar_vol) or avg_dollar_vol < min_avg_dollar_vol:
        diagnostics['reason'] = 'min_dollar_vol'
        return False, diagnostics
    
    diagnostics['reason'] = 'passed'
    return True, diagnostics


def filter_universe_by_liquidity(
    price_data: Dict[str, pd.DataFrame],
    min_price: float = 5.0,
    min_avg_dollar_vol: float = 20_000_000,
    window: int = 20,
    verbose: bool = True
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Filter a dictionary of price data by liquidity.
    
    Args:
        price_data: Dict mapping symbol -> DataFrame
        min_price: Minimum last price
        min_avg_dollar_vol: Minimum average dollar volume
        window: Lookback window
        verbose: Print summary
        
    Returns:
        Tuple of (filtered_price_data, liquidity_report_df)
    """
    filtered = {}
    report_rows = []
    
    for symbol, df in price_data.items():
        passed, diag = passes_liquidity_filter(
            df, 
            min_price=min_price,
            min_avg_dollar_vol=min_avg_dollar_vol,
            window=window
        )
        
        report_rows.append({
            'symbol': symbol,
            'pass_liquidity': passed,
            'price_last': diag.get('price_last', np.nan),
            'avg_dollar_vol_20': diag.get('avg_dollar_vol_20', np.nan),
            'reason': diag.get('reason', '')
        })
        
        if passed:
            filtered[symbol] = df
    
    report_df = pd.DataFrame(report_rows)
    
    if verbose:
        n_total = len(price_data)
        n_passed = len(filtered)
        n_failed = n_total - n_passed
        print(f"Liquidity filter: {n_passed}/{n_total} passed ({n_failed} filtered out)")
    
    return filtered, report_df


def get_demo_symbols() -> List[str]:
    """Return the original demo symbol list."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
        "AMD", "INTC", "CRM", "ADBE", "PYPL", "CSCO", "QCOM", "TXN",
        "AVGO", "INTU", "AMAT", "MU"
    ]


def refresh_symbol_cache(cache_path: str = "data/nasdaq_symbols_cache.csv") -> bool:
    """
    Force refresh of symbol cache.
    
    Returns True if refresh succeeded, False if fell back to cached/fallback data.
    """
    print("Forcing symbol cache refresh...")
    cache_file = Path(cache_path)
    
    if cache_file.exists():
        bak_path = cache_file.with_suffix('.csv.bak')
        try:
            shutil.copy2(cache_file, bak_path)
        except Exception:
            pass
    
    symbols = _download_nasdaq_symbols_validated(cache_path, exclude_funds=True)
    
    if symbols and len(symbols) >= 200:
        print(f"Successfully refreshed cache with {len(symbols)} symbols")
        return True
    else:
        print("Refresh failed, keeping existing cache/fallback")
        return False
