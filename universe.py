"""
Universe management: NASDAQ symbol list caching and liquidity filtering.
"""
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

NASDAQ_SCREENER_URL = "https://www.nasdaq.com/market-activity/stocks/screener?exchange=nasdaq&render=download"

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


def get_nasdaq_symbols_cached(
    cache_path: str = "data/nasdaq_symbols_cache.csv",
    max_age_hours: int = 24,
    exclude_funds: bool = True,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Get NASDAQ symbol list, using cache if available and fresh.
    
    Args:
        cache_path: Path to cache file
        max_age_hours: Max age in hours before refreshing cache
        exclude_funds: Whether to exclude ETFs/funds
        limit: Optional limit on number of symbols
        
    Returns:
        List of uppercase ticker symbols
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    use_cache = False
    if cache_file.exists():
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if file_age_hours < max_age_hours:
            use_cache = True
    
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
    
    symbols = _download_nasdaq_symbols(exclude_funds=exclude_funds)
    
    if symbols and len(symbols) > 50:
        pd.DataFrame({'symbol': symbols}).to_csv(cache_file, index=False)
        print(f"Cached {len(symbols)} NASDAQ symbols to {cache_path}")
    elif len(symbols) <= 50:
        print(f"Download returned only {len(symbols)} symbols - using fallback instead")
        symbols = _get_fallback_symbols()
    
    if limit:
        symbols = symbols[:limit]
    
    return symbols


def _download_nasdaq_symbols(exclude_funds: bool = True) -> List[str]:
    """
    Download NASDAQ symbol list from NASDAQ screener.
    
    Falls back to a hardcoded list if download fails.
    """
    import requests
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/csv,application/csv,text/plain',
    }
    
    try:
        print("Downloading NASDAQ symbol list...")
        response = requests.get(NASDAQ_SCREENER_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(response.text))
        except Exception:
            df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')
        
        symbol_col = None
        name_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower == 'symbol':
                symbol_col = col
            elif col_lower == 'name':
                name_col = col
        
        if symbol_col is None:
            print("Warning: Could not find Symbol column, trying first column")
            symbol_col = df.columns[0]
        
        symbols = []
        for idx, row in df.iterrows():
            symbol = str(row[symbol_col]).strip().upper()
            name = str(row[name_col]) if name_col else ""
            
            if not _is_valid_ticker(symbol):
                continue
            
            if exclude_funds and _is_fund(name):
                continue
            
            symbols.append(symbol)
        
        print(f"Downloaded {len(symbols)} valid NASDAQ symbols")
        return symbols
        
    except Exception as e:
        print(f"Warning: Failed to download NASDAQ symbols: {e}")
        print("Using fallback symbol list")
        return _get_fallback_symbols()


def _get_fallback_symbols() -> List[str]:
    """Return a fallback list of 200+ liquid NASDAQ stocks."""
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
        "SIRI", "LCID", "RIVN", "SOFI", "PLTR", "COIN", "ROKU", "HOOD",
        "DOCU", "SPLK", "OKTA", "MDB", "NET", "SNOW", "BILL", "HUBS",
        "VEEV", "TWLO", "COUP", "ZM", "PINS", "SNAP", "UBER", "LYFT",
        "DASH", "RBLX", "U", "PATH", "SAMSARA", "IOT", "CFLT", "ESTC",
        "GTLB", "MNDY", "DOCN", "APP", "BRZE", "S", "CYBR", "TENB",
        "RPD", "SUMO", "NEWR", "DT", "PD", "BSY", "MTTR", "ASAN",
        "FIVN", "RNG", "TOST", "SQ", "AFRM", "UPST", "LMND", "ROOT",
        "OPEN", "OPENDOOR", "RDFN", "EXPI", "COUR", "DUOL", "GENI",
        "DKNG", "PENN", "RSI", "SKLZ", "SRAD", "EVBG", "MSTR", "CLSK",
        "MARA", "RIOT", "HUT", "BTBT", "SOS", "CAN", "GREE", "BITF",
        "NVAX", "MRNA", "BNTX", "VCNX", "IOVA", "SGEN", "EXAS", "NTRA",
        "RARE", "ALNY", "IONS", "SRPT", "BMRN", "JAZZ", "UTHR", "NBIX",
        "HZNP", "INCY", "TECH", "BIO", "HOLX", "ALGM", "SLAB", "SWKS",
        "MPWR", "OLED", "MKSI", "ENTG", "ONTO", "WOLF", "DIOD", "SYNA",
        "POWI", "CRUS", "AMBA", "SITM", "RMBS", "ACLS", "FORM", "ICHR",
        "AXTI", "CAMT", "UCTT", "VECO", "AEHR", "MTSI", "MACOM", "SMTC",
        "HIMX", "AOSL", "INDI", "SIMO", "GSIT", "QUIK", "VSH", "SGH",
        "ZBRA", "EPAM", "GLOB", "EXLS", "PRFT", "ASGN", "FICO", "PAYC",
        "PCTY", "TYL", "GWRE", "MANH", "NCNO", "APPF", "YEXT", "SPSC",
        "EVBG", "MODN", "PLMR", "QTWO", "ALTR", "ALKT", "NTCT", "CGNX",
        "OMCL", "MGNI", "PUBM", "DV", "APPS", "INMD", "PODD", "NVRO",
        "AXNX", "TNDM", "HALO", "XRAY", "MASI", "OFIX", "LNTH", "CAKE",
        "TXRH", "WING", "SHAK", "PLAY", "EAT", "DRI", "BLMN", "DIN",
        "BJRI", "CHUY", "KURA", "BROS", "LOCO", "FAT", "PZZA", "WEN",
        "JACK", "NDLS", "SONC", "ARCO", "DEL", "DNUT", "OLO", "PTLO",
        "FWRG", "LSCC", "ACAD", "EXEL", "MEDP", "IRTC", "QDEL", "FTRE",
        "OGN", "OMCL", "PRGO", "VRTV", "PETQ", "CHWY", "WOOF", "FRPT",
    ]


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
