# NASDAQ Double Bottom Pattern Scanner

## Overview
A comprehensive Python program that scans for double bottom ("W") chart patterns in NASDAQ-listed stocks using historical price data from yfinance. The scanner uses algorithmic pattern detection with scipy, RSI divergence confirmation, and volume analysis.

## Project Structure
```
.
├── main.py                    # Main entry point (demo mode or CLI)
├── double_bottom_scanner.py   # Core scanner module with all functions
├── double_bottom_results.csv  # Output: detected patterns (generated)
├── charts/                    # Output: generated pattern charts (generated)
├── pyproject.toml            # Python dependencies
└── .gitignore                # Git ignore rules
```

## Features
- Algorithmic detection of classic double bottom patterns
- Swing low/high detection using scipy.signal.find_peaks
- RSI divergence confirmation (bullish divergence)
- Volume analysis (decreased volume on 2nd bottom)
- Pattern strength scoring (0-100)
- Configurable parameters via command-line arguments
- CSV output with all detected patterns
- Chart generation with matplotlib

## Usage

### Quick Demo (20 stocks)
```bash
python main.py
```

### Custom Scans
```bash
# Scan 100 stocks
python main.py --max-stocks 100

# Scan specific symbols
python main.py --symbols AAPL MSFT GOOGL NVDA

# Generate charts for top 5 patterns
python main.py --plot --num-plots 5

# Adjust detection parameters
python main.py --price-tolerance 0.03 --min-peak-height 0.08

# See all options
python main.py --help
```

## Configuration Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| price_tolerance | 4% | Max price difference between bottoms |
| min_peak_height | 6% | Minimum peak height above bottoms |
| min_separation | 20 days | Minimum days between bottoms |
| max_separation | 200 days | Maximum days between bottoms |
| lookback_days | 504 | Days of history to analyze (~2 years) |

## Pattern Detection Logic
1. Find swing lows using inverted peak detection
2. Identify pairs of lows with similar prices (within tolerance)
3. Confirm a swing high exists between the lows
4. Calculate pattern strength based on:
   - Price similarity between bottoms
   - Peak height above support
   - Volume confirmation (decreasing)
   - RSI bullish divergence
   - Neckline breakout confirmation

## Output Format
The CSV contains:
- symbol: Stock ticker
- status: 'confirmed' or 'forming'
- strength_score: Pattern quality (0-100)
- bottom1_date, bottom1_price: First bottom details
- peak_date, neckline: Peak/neckline details
- bottom2_date, bottom2_price: Second bottom details
- current_price: Latest close price
- target_price: Projected price target (neckline + pattern height)

## Dependencies
- yfinance: Stock data download
- pandas: Data manipulation
- scipy: Peak detection algorithms
- matplotlib: Chart generation
- pandas-ta: Technical indicators (RSI)
- tqdm: Progress bars
- requests: HTTP requests

## Recent Changes
- 2025-12-23: Initial implementation with full feature set

## Architecture Decisions
- Using scipy.signal.find_peaks for reliable swing detection
- Pattern strength scoring provides quality ranking
- Rate limiting (0.1s delay) prevents API blocks
- Modular design allows easy extension
