#!/usr/bin/env python3
"""
NASDAQ Double Bottom Pattern Scanner

A comprehensive Python program that scans for double bottom ("W") chart patterns
in NASDAQ-listed stocks using historical price data from yfinance.

Features:
- Algorithmic detection of classic double bottom patterns
- Swing low detection using scipy.signal.find_peaks
- RSI divergence confirmation
- Volume analysis
- Configurable parameters
- CSV output and optional charting
- Progress tracking with tqdm

Author: Double Bottom Scanner
Version: 1.0.0
"""

import argparse
import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf
from scipy.signal import find_peaks
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_CONFIG = {
    'price_tolerance': 0.04,          # Max price difference between bottoms (4%)
    'min_peak_height': 0.06,          # Minimum peak height above bottoms (6%)
    'min_separation': 20,             # Minimum days between bottoms
    'max_separation': 200,            # Maximum days between bottoms
    'lookback_days': 504,             # ~2 years of trading data
    'swing_prominence': 0.02,         # Prominence for peak detection (2%)
    'swing_distance': 5,              # Minimum distance between swing points
    'rsi_period': 14,                 # RSI calculation period
    'volume_decrease_threshold': 0.8, # Volume on 2nd bottom should be < 80% of 1st
    'confirmation_days': 10,          # Days to check for neckline breakout
    'download_years': 3,              # Years of historical data to download
    'rate_limit_delay': 0.1,          # Delay between API calls
    'trigger_lookahead_days': 10,     # Days to look ahead for forming trigger level
}


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def get_nasdaq_symbols(max_symbols: Optional[int] = None, 
                       exclude_etfs: bool = True) -> List[str]:
    """
    Fetch list of NASDAQ stock symbols.
    
    Uses a reliable static list of common NASDAQ symbols as the NASDAQ API
    requires authentication. Falls back to a curated list of liquid stocks.
    
    Args:
        max_symbols: Maximum number of symbols to return (None for all)
        exclude_etfs: Whether to exclude ETF symbols
        
    Returns:
        List of stock symbols
    """
    nasdaq_symbols = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
        'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'INTC', 'CMCSA', 'TMUS', 'INTU', 'AMGN',
        'TXN', 'QCOM', 'HON', 'ISRG', 'BKNG', 'SBUX', 'AMAT', 'VRTX', 'GILD', 'ADP',
        'LRCX', 'MDLZ', 'REGN', 'ADI', 'PANW', 'SNPS', 'KLAC', 'MELI', 'ASML', 'CDNS',
        'PYPL', 'MNST', 'MRVL', 'CTAS', 'ORLY', 'MAR', 'CHTR', 'ABNB', 'FTNT', 'NXPI',
        'WDAY', 'KDP', 'KHC', 'DXCM', 'PAYX', 'MCHP', 'AEP', 'PCAR', 'IDXX', 'EXC',
        'LULU', 'CPRT', 'FAST', 'ODFL', 'ROST', 'VRSK', 'AZN', 'CTSH', 'CSGP', 'EA',
        'BKR', 'DLTR', 'XEL', 'WBD', 'DDOG', 'FANG', 'BIIB', 'TEAM', 'GEHC', 'GFS',
        'ANSS', 'ILMN', 'ALGN', 'MRNA', 'ZS', 'WBA', 'EBAY', 'SIRI', 'ENPH', 'LCID',
        'RIVN', 'OKTA', 'CRWD', 'ZM', 'DOCU', 'ROKU', 'SNOW', 'NET', 'MDB', 'COIN',
        'HOOD', 'RBLX', 'PLTR', 'PATH', 'U', 'DASH', 'PINS', 'SNAP', 'LYFT', 'UBER',
        'SQ', 'SHOP', 'TWLO', 'SPLK', 'TTD', 'SPOT', 'MTCH', 'BIDU', 'JD', 'PDD',
        'NTES', 'BILI', 'TME', 'VIPS', 'NIO', 'XPEV', 'LI', 'BABA', 'TCOM', 'WB',
        'PTON', 'CHWY', 'ETSY', 'SFIX', 'WISH', 'APPS', 'FUBO', 'SKLZ', 'SOFI', 'OPEN',
        'UPST', 'AFRM', 'HIMS', 'DNA', 'LAZR', 'IONQ', 'SMCI', 'ARM', 'CELH', 'AXON',
        'MARA', 'RIOT', 'HUT', 'BITF', 'CLSK', 'CORZ', 'IREN', 'SATS', 'BTDR', 'WULF',
        'CIFR', 'GREE', 'MIGI', 'ARBK', 'HIVE', 'BTBT', 'CAN', 'SOS', 'EBON', 'NCTY',
        'IMPP', 'ENVX', 'MVST', 'QS', 'LCID', 'FSR', 'FFIE', 'GOEV', 'RIDE', 'WKHS',
        'NKLA', 'HYLN', 'ARVL', 'PTRA', 'LEV', 'EVGO', 'CHPT', 'BLNK', 'VLTA', 'BEEM',
        'ACHR', 'JOBY', 'LILM', 'EVTL', 'BLDE', 'SKYH', 'SPCE', 'RDW', 'RKLB', 'ASTR',
    ]
    
    if exclude_etfs:
        etf_patterns = ['ETF', 'QQQ', 'SPY', 'IWM', 'VTI', 'VOO']
        nasdaq_symbols = [s for s in nasdaq_symbols if not any(e in s for e in etf_patterns)]
    
    unique_symbols = list(dict.fromkeys(nasdaq_symbols))
    
    if max_symbols:
        return unique_symbols[:max_symbols]
    return unique_symbols


def download_stock_data(symbol: str, years: int = 3) -> Optional[pd.DataFrame]:
    """
    Download historical OHLCV data for a stock using yfinance.
    
    Args:
        symbol: Stock ticker symbol
        years: Number of years of historical data to download
        
    Returns:
        DataFrame with OHLCV data or None if download fails
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty or len(df) < 100:
            return None
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.dropna()
        
        return df
        
    except Exception as e:
        return None


# ============================================================================
# PATTERN DETECTION FUNCTIONS
# ============================================================================

def find_swing_lows(prices: np.ndarray, distance: int = 5, 
                    prominence: float = 0.02) -> np.ndarray:
    """
    Find swing low points (local minima) in price series.
    
    Uses scipy.signal.find_peaks on inverted prices to find local minima.
    
    Args:
        prices: Array of price values (typically Low prices)
        distance: Minimum distance between peaks
        prominence: Minimum prominence of peaks
        
    Returns:
        Array of indices where swing lows occur
    """
    inverted = -prices
    prominence_value = prominence * np.mean(prices)
    
    peaks, properties = find_peaks(
        inverted,
        distance=distance,
        prominence=prominence_value
    )
    
    return peaks


def find_swing_highs(prices: np.ndarray, distance: int = 5,
                     prominence: float = 0.02) -> np.ndarray:
    """
    Find swing high points (local maxima) in price series.
    
    Args:
        prices: Array of price values (typically High prices)
        distance: Minimum distance between peaks
        prominence: Minimum prominence of peaks
        
    Returns:
        Array of indices where swing highs occur
    """
    prominence_value = prominence * np.mean(prices)
    
    peaks, properties = find_peaks(
        prices,
        distance=distance,
        prominence=prominence_value
    )
    
    return peaks


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with Close prices
        period: RSI calculation period
        
    Returns:
        Series with RSI values
    """
    try:
        rsi = ta.rsi(df['Close'], length=period)
        return rsi
    except Exception:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


def compute_forming_trigger_level(
    df: pd.DataFrame,
    bottom2_pos: int,
    trigger_lookahead_days: int = 10
) -> Tuple[Optional[float], str]:
    """
    Compute forming trigger level for early entry logic.
    
    v1 rule: forming_trigger_level = max(High[bottom2_pos+1 : bottom2_pos+1+trigger_lookahead_days])
    
    Args:
        df: DataFrame with OHLCV data
        bottom2_pos: Index position of the second bottom
        trigger_lookahead_days: Number of days to look ahead
        
    Returns:
        Tuple of (forming_trigger_level, reason)
    """
    start = bottom2_pos + 1
    end = min(bottom2_pos + 1 + trigger_lookahead_days, len(df))
    
    if start >= end:
        return None, "insufficient_bars_after_bottom2"
    
    level = float(df["High"].iloc[start:end].max())
    
    if not (level > 0):
        return None, "invalid_trigger_level"
    
    return round(level, 2), f"max_high_next_{end-start}_bars_after_bottom2"


def find_breakout_date(
    df: pd.DataFrame,
    bottom2_pos: int,
    neckline: float,
    max_lookahead: int = 60
) -> Optional[pd.Timestamp]:
    """
    Find the date when price first closed above the neckline after bottom2.
    
    Args:
        df: DataFrame with OHLCV data
        bottom2_pos: Index position of the second bottom
        neckline: Neckline price level
        max_lookahead: Maximum bars to look ahead
        
    Returns:
        Breakout date or None if not found
    """
    start = bottom2_pos + 1
    end = min(bottom2_pos + 1 + max_lookahead, len(df))
    
    for idx in range(start, end):
        if df['Close'].iloc[idx] > neckline:
            return df.index[idx]
    
    return None


def detect_double_bottom(df: pd.DataFrame, config: Dict) -> List[Dict]:
    """
    Detect double bottom patterns in stock price data.
    
    A valid double bottom pattern has:
    1. Two distinct swing lows at similar price levels
    2. A swing high (peak/neckline) between the lows
    3. The peak is significantly higher than the lows
    4. Proper time separation between the lows
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary with detection parameters
        
    Returns:
        List of detected pattern dictionaries
    """
    patterns = []
    
    if len(df) < config['min_separation'] * 2:
        return patterns
    
    low_prices = df['Low'].values
    high_prices = df['High'].values
    close_prices = df['Close'].values
    volumes = df['Volume'].values
    dates = df.index
    
    rsi = calculate_rsi(df, config['rsi_period'])
    rsi_values = rsi.values
    
    swing_lows = find_swing_lows(
        low_prices,
        distance=config['swing_distance'],
        prominence=config['swing_prominence']
    )
    
    swing_highs = find_swing_highs(
        high_prices,
        distance=config['swing_distance'],
        prominence=config['swing_prominence']
    )
    
    if len(swing_lows) < 2:
        return patterns
    
    for i in range(len(swing_lows) - 1):
        for j in range(i + 1, len(swing_lows)):
            idx1 = swing_lows[i]
            idx2 = swing_lows[j]
            
            separation = idx2 - idx1
            if separation < config['min_separation'] or separation > config['max_separation']:
                continue
            
            price1 = low_prices[idx1]
            price2 = low_prices[idx2]
            avg_bottom = (price1 + price2) / 2
            price_diff = abs(price1 - price2) / avg_bottom
            
            if price_diff > config['price_tolerance']:
                continue
            
            peaks_between = swing_highs[(swing_highs > idx1) & (swing_highs < idx2)]
            
            if len(peaks_between) == 0:
                continue
            
            peak_idx = peaks_between[np.argmax(high_prices[peaks_between])]
            peak_price = high_prices[peak_idx]
            
            peak_height = (peak_price - avg_bottom) / avg_bottom
            if peak_height < config['min_peak_height']:
                continue
            
            neckline = peak_price
            
            vol1 = volumes[max(0, idx1-2):idx1+3].mean() if idx1 >= 2 else volumes[idx1]
            vol2 = volumes[max(0, idx2-2):idx2+3].mean() if idx2 >= 2 else volumes[idx2]
            volume_decreased = vol2 < vol1 * config['volume_decrease_threshold']
            
            rsi1 = rsi_values[idx1] if not np.isnan(rsi_values[idx1]) else 50
            rsi2 = rsi_values[idx2] if not np.isnan(rsi_values[idx2]) else 50
            rsi_divergence = (price2 <= price1 and rsi2 > rsi1)
            
            status = 'FORMING'
            breakout_date = None
            
            breakout_date = find_breakout_date(df, idx2, neckline, max_lookahead=config['confirmation_days'])
            if breakout_date is not None:
                status = 'CONFIRMED'
            
            pattern_height = neckline - avg_bottom
            target_price = neckline + pattern_height
            
            trigger_lookahead = config.get('trigger_lookahead_days', 10)
            forming_trigger_level, forming_trigger_reason = compute_forming_trigger_level(
                df, idx2, trigger_lookahead
            )
            
            strength_score = 0
            strength_score += 25 if price_diff < 0.02 else (15 if price_diff < 0.03 else 5)
            strength_score += 25 if peak_height > 0.10 else (15 if peak_height > 0.07 else 5)
            strength_score += 20 if volume_decreased else 0
            strength_score += 20 if rsi_divergence else 0
            strength_score += 10 if status == 'CONFIRMED' else 0
            
            pattern = {
                'bottom1_date': dates[idx1],
                'bottom1_price': round(price1, 2),
                'bottom1_idx': idx1,
                'peak_date': dates[peak_idx],
                'peak_price': round(peak_price, 2),
                'neckline': round(neckline, 2),
                'peak_idx': peak_idx,
                'bottom2_date': dates[idx2],
                'bottom2_price': round(price2, 2),
                'bottom2_idx': idx2,
                'status': status,
                'breakout_date': breakout_date,
                'price_diff_pct': round(price_diff * 100, 2),
                'peak_height_pct': round(peak_height * 100, 2),
                'separation_days': separation,
                'height': round(pattern_height, 2),
                'avg_bottom': round(avg_bottom, 2),
                'volume_confirmation': volume_decreased,
                'rsi_divergence': rsi_divergence,
                'target_price': round(target_price, 2),
                'score': strength_score,
                'strength_score': strength_score,
                'forming_trigger_level': forming_trigger_level,
                'forming_trigger_reason': forming_trigger_reason,
                'current_price': round(close_prices[-1], 2),
            }
            
            patterns.append(pattern)
    
    patterns.sort(key=lambda x: (-x['strength_score'], -x['bottom2_idx']))
    
    if patterns:
        filtered = [patterns[0]]
        for p in patterns[1:]:
            is_overlapping = any(
                abs(p['bottom2_idx'] - existing['bottom2_idx']) < 10
                for existing in filtered
            )
            if not is_overlapping:
                filtered.append(p)
        patterns = filtered
    
    return patterns


# ============================================================================
# SCANNING AND OUTPUT FUNCTIONS
# ============================================================================

def scan_stocks(symbols: List[str], config: Dict, 
                verbose: bool = True,
                return_price_data: bool = False):
    """
    Scan multiple stocks for double bottom patterns.
    
    Args:
        symbols: List of stock symbols to scan
        config: Configuration dictionary
        verbose: Whether to show progress bar
        return_price_data: If True, also return price data dict
        
    Returns:
        DataFrame with all detected patterns
        If return_price_data=True: Tuple of (DataFrame, Dict[str, DataFrame])
    """
    all_patterns = []
    price_data = {}
    
    iterator = tqdm(symbols, desc="Scanning stocks") if verbose else symbols
    
    for symbol in iterator:
        try:
            df = download_stock_data(symbol, years=config['download_years'])
            
            if df is None:
                continue
            
            lookback = min(config['lookback_days'], len(df))
            df_recent = df.iloc[-lookback:]
            
            price_data[symbol] = df_recent
            
            patterns = detect_double_bottom(df_recent, config)
            
            for pattern in patterns:
                pattern['symbol'] = symbol
                bottom2_date = pattern['bottom2_date']
                date_str = pd.Timestamp(bottom2_date).strftime('%Y%m%d')
                pattern['pattern_id'] = f"{symbol}_{date_str}_{pattern['status']}"
                all_patterns.append(pattern)
            
            time.sleep(config['rate_limit_delay'])
            
        except Exception as e:
            continue
    
    if not all_patterns:
        if return_price_data:
            return pd.DataFrame(), price_data
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_patterns)
    
    column_order = [
        'pattern_id', 'symbol', 'status', 'score', 'strength_score',
        'bottom1_date', 'bottom1_price',
        'peak_date', 'peak_price', 'neckline',
        'bottom2_date', 'bottom2_price',
        'breakout_date',
        'current_price', 'target_price',
        'height', 'avg_bottom', 'separation_days',
        'forming_trigger_level', 'forming_trigger_reason',
        'price_diff_pct', 'peak_height_pct',
        'volume_confirmation', 'rsi_divergence',
        'bottom1_idx', 'peak_idx', 'bottom2_idx',
    ]
    
    available_cols = [c for c in column_order if c in results_df.columns]
    extra_cols = [c for c in results_df.columns if c not in column_order]
    results_df = results_df[available_cols + extra_cols]
    
    results_df = results_df.sort_values(
        ['status', 'score', 'bottom2_date'],
        ascending=[False, False, False]
    )
    
    if return_price_data:
        return results_df, price_data
    return results_df


def save_results_to_csv(df: pd.DataFrame, filename: str = 'double_bottom_results.csv'):
    """
    Save scan results to a CSV file.
    
    Args:
        df: DataFrame with scan results
        filename: Output filename
    """
    if df.empty:
        print("No patterns found. CSV not saved.")
        return
    
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")


def print_summary(df: pd.DataFrame):
    """
    Print a summary of scan results.
    
    Args:
        df: DataFrame with scan results
    """
    if df.empty:
        print("\n" + "="*60)
        print("DOUBLE BOTTOM SCAN RESULTS")
        print("="*60)
        print("No double bottom patterns detected.")
        return
    
    print("\n" + "="*80)
    print("DOUBLE BOTTOM SCAN RESULTS")
    print("="*80)
    
    confirmed = df[df['status'] == 'CONFIRMED']
    forming = df[df['status'] == 'FORMING']
    
    print(f"\nTotal patterns found: {len(df)}")
    print(f"  - Confirmed: {len(confirmed)}")
    print(f"  - Forming: {len(forming)}")
    
    if len(confirmed) > 0:
        print("\n" + "-"*80)
        print("TOP CONFIRMED PATTERNS:")
        print("-"*80)
        display_cols = ['symbol', 'strength_score', 'bottom1_date', 'neckline', 
                       'bottom2_date', 'current_price', 'target_price']
        available = [c for c in display_cols if c in confirmed.columns]
        print(confirmed.head(10)[available].to_string(index=False))
    
    if len(forming) > 0:
        print("\n" + "-"*80)
        print("TOP FORMING PATTERNS:")
        print("-"*80)
        display_cols = ['symbol', 'strength_score', 'bottom1_date', 'neckline',
                       'bottom2_date', 'current_price', 'target_price']
        available = [c for c in display_cols if c in forming.columns]
        print(forming.head(10)[available].to_string(index=False))
    
    print("\n" + "="*80)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_double_bottom(symbol: str, df: pd.DataFrame, pattern: Dict,
                       save_path: Optional[str] = None, show: bool = True):
    """
    Plot a stock chart with double bottom pattern marked.
    
    Args:
        symbol: Stock ticker symbol
        df: DataFrame with OHLCV data
        pattern: Pattern dictionary with detection results
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    start_idx = max(0, pattern['bottom1_idx'] - 20)
    end_idx = min(len(df), pattern['bottom2_idx'] + 40)
    
    plot_df = df.iloc[start_idx:end_idx]
    dates = plot_df.index
    
    ax1.plot(dates, plot_df['Close'], color='blue', linewidth=1.5, label='Close Price')
    ax1.fill_between(dates, plot_df['Low'], plot_df['High'], 
                     alpha=0.2, color='gray', label='High-Low Range')
    
    b1_date = pattern['bottom1_date']
    b2_date = pattern['bottom2_date']
    peak_date = pattern['peak_date']
    
    ax1.scatter([b1_date], [pattern['bottom1_price']], 
                color='green', s=200, marker='^', zorder=5, label='Bottom 1')
    ax1.scatter([b2_date], [pattern['bottom2_price']], 
                color='lime', s=200, marker='^', zorder=5, label='Bottom 2')
    ax1.scatter([peak_date], [pattern['neckline']], 
                color='red', s=200, marker='v', zorder=5, label='Peak/Neckline')
    
    ax1.axhline(y=pattern['neckline'], color='red', linestyle='--', 
                linewidth=1, alpha=0.7, label=f"Neckline: ${pattern['neckline']}")
    
    avg_bottom = (pattern['bottom1_price'] + pattern['bottom2_price']) / 2
    ax1.axhline(y=avg_bottom, color='green', linestyle=':', 
                linewidth=1, alpha=0.7, label=f"Support: ${avg_bottom:.2f}")
    
    ax1.axhline(y=pattern['target_price'], color='purple', linestyle='--', 
                linewidth=1, alpha=0.7, label=f"Target: ${pattern['target_price']}")
    
    ax1.plot([b1_date, peak_date, b2_date],
             [pattern['bottom1_price'], pattern['neckline'], pattern['bottom2_price']],
             color='orange', linewidth=2, linestyle='-', alpha=0.8, marker='o')
    
    status_color = 'green' if pattern['status'] == 'CONFIRMED' else 'orange'
    status_text = pattern['status'].upper()
    
    ax1.set_title(f"{symbol} - Double Bottom Pattern ({status_text})\n"
                  f"Strength Score: {pattern['strength_score']}/100 | "
                  f"Current: ${pattern['current_price']} | Target: ${pattern['target_price']}",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    colors = ['green' if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i] 
              else 'red' for i in range(len(plot_df))]
    ax2.bar(dates, plot_df['Volume'], color=colors, alpha=0.7, width=0.8)
    
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_top_patterns(results_df: pd.DataFrame, config: Dict,
                      num_plots: int = 5, save_dir: str = 'charts'):
    """
    Generate charts for top detected patterns.
    
    Args:
        results_df: DataFrame with scan results
        config: Configuration dictionary
        num_plots: Number of charts to generate
        save_dir: Directory to save charts
    """
    if results_df.empty:
        print("No patterns to plot.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    top_patterns = results_df.head(num_plots)
    
    for _, row in tqdm(top_patterns.iterrows(), total=len(top_patterns), 
                       desc="Generating charts"):
        symbol = row['symbol']
        
        df = download_stock_data(symbol, years=config['download_years'])
        
        if df is None:
            continue
        
        lookback = min(config['lookback_days'], len(df))
        df_recent = df.iloc[-lookback:]
        
        patterns = detect_double_bottom(df_recent, config)
        
        if patterns:
            pattern = patterns[0]
            save_path = os.path.join(save_dir, f"{symbol}_double_bottom.png")
            plot_double_bottom(symbol, df_recent, pattern, 
                             save_path=save_path, show=False)
        
        time.sleep(config['rate_limit_delay'])
    
    print(f"\nCharts saved to: {save_dir}/")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for the double bottom scanner."""
    
    parser = argparse.ArgumentParser(
        description='NASDAQ Double Bottom Pattern Scanner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Specific stock symbols to scan (space-separated)')
    parser.add_argument('--max-stocks', type=int, default=50,
                       help='Maximum number of NASDAQ stocks to scan')
    parser.add_argument('--price-tolerance', type=float, default=0.04,
                       help='Max price difference between bottoms (decimal)')
    parser.add_argument('--min-peak-height', type=float, default=0.06,
                       help='Minimum peak height above bottoms (decimal)')
    parser.add_argument('--min-separation', type=int, default=20,
                       help='Minimum days between bottoms')
    parser.add_argument('--max-separation', type=int, default=200,
                       help='Maximum days between bottoms')
    parser.add_argument('--lookback-days', type=int, default=504,
                       help='Days of history to analyze')
    parser.add_argument('--output', type=str, default='double_bottom_results.csv',
                       help='Output CSV filename')
    parser.add_argument('--plot', action='store_true',
                       help='Generate charts for top patterns')
    parser.add_argument('--num-plots', type=int, default=5,
                       help='Number of charts to generate')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config['price_tolerance'] = args.price_tolerance
    config['min_peak_height'] = args.min_peak_height
    config['min_separation'] = args.min_separation
    config['max_separation'] = args.max_separation
    config['lookback_days'] = args.lookback_days
    
    print("\n" + "="*60)
    print("NASDAQ DOUBLE BOTTOM PATTERN SCANNER")
    print("="*60)
    print(f"Price tolerance: {config['price_tolerance']*100:.1f}%")
    print(f"Min peak height: {config['min_peak_height']*100:.1f}%")
    print(f"Separation range: {config['min_separation']}-{config['max_separation']} days")
    print(f"Lookback period: {config['lookback_days']} days")
    print("="*60 + "\n")
    
    if args.symbols:
        symbols = args.symbols
        print(f"Scanning {len(symbols)} specified symbol(s)...")
    else:
        symbols = get_nasdaq_symbols(max_symbols=args.max_stocks)
        print(f"Scanning {len(symbols)} NASDAQ stocks...")
    
    results = scan_stocks(symbols, config, verbose=not args.quiet)
    
    print_summary(results)
    
    save_results_to_csv(results, args.output)
    
    if args.plot and not results.empty:
        print(f"\nGenerating {args.num_plots} chart(s)...")
        plot_top_patterns(results, config, num_plots=args.num_plots)
    
    return results


if __name__ == '__main__':
    main()
