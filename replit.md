# NASDAQ Double Bottom Pattern Scanner

## Overview
A comprehensive Python program that scans for double bottom ("W") chart patterns in NASDAQ-listed stocks using historical price data from yfinance. The scanner uses algorithmic pattern detection with scipy, RSI divergence confirmation, and volume analysis. Includes a full backtesting engine to evaluate strategy performance.

## Project Structure
```
.
├── main.py                    # Main entry point (demo mode, CLI, or backtest)
├── double_bottom_scanner.py   # Core scanner module with all functions
├── strategy.py                # Trading strategy and signal generation
├── backtester.py              # Backtesting engine
├── metrics.py                 # Performance metrics calculation
├── alerts.py                  # Alert system for pattern breakouts
├── double_bottom_results.csv  # Output: detected patterns (generated)
├── charts/                    # Output: charts and equity curves (generated)
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
- **Backtesting engine** with bar-by-bar simulation
- **Performance metrics** (win rate, Sharpe ratio, drawdown, etc.)
- **Alert system** for confirmed pattern breakouts

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

### Backtesting
```bash
# Run backtest with default settings
python main.py --backtest

# Custom backtest configuration
python main.py --backtest --symbols AAPL NVDA AMD --initial-capital 10000

# Full configuration
python main.py --backtest --max-stocks 50 \
  --initial-capital 25000 \
  --position-size 0.15 \
  --stop-loss-buffer 0.02 \
  --max-hold-days 60 \
  --trailing-stop
```

### Backtest Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| --initial-capital | 10000 | Starting capital for simulation |
| --position-size | 0.1 | Fraction of capital per trade (10%) |
| --stop-loss-buffer | 0.02 | Buffer below support level (2%) |
| --max-hold-days | 60 | Maximum holding period |
| --trailing-stop | off | Enable trailing stop loss |
| --trailing-stop-pct | 0.05 | Trailing stop percentage (5%) |

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
- pattern_id: Unique identifier (SYMBOL_YYYYMMDD_STATUS format)
- symbol: Stock ticker
- status: 'FORMING' or 'CONFIRMED' (uppercase)
- strength_score: Pattern quality (0-100)
- bottom1_date, bottom1_price: First bottom details
- peak_date, neckline: Peak/neckline details
- bottom2_date, bottom2_price: Second bottom details
- breakout_date: Date when price closed above neckline (CONFIRMED only)
- current_price: Latest close price
- target_price: Projected price target (neckline + pattern height)
- height: Pattern height (neckline - avg_bottom)
- avg_bottom: Average of both bottom prices
- separation_days: Trading days between bottoms
- forming_trigger_level: Price level that would confirm FORMING patterns
- forming_trigger_reason: How trigger level was computed

## Dependencies
- yfinance: Stock data download
- pandas: Data manipulation
- scipy: Peak detection algorithms
- matplotlib: Chart generation
- pandas-ta: Technical indicators (RSI)
- tqdm: Progress bars
- requests: HTTP requests

## Recent Changes
- 2025-12-23: Strategy Signal Generation (Task 2)
  - TradeSignal now includes: pattern_id, entry_kind, trigger_level, risk_per_share, meta
  - generate_signals() supports both CONFIRMED and FORMING entry types
  - CONFIRMED: enter next bar open after neckline breakout close
  - FORMING: enter next bar open after trigger level break (green candle required)
  - Signal deduplication: one signal per pattern, CONFIRMED preferred
  - scan_stocks() now has return_price_data option for shared data access
- 2025-12-23: Pattern Output Hardening
  - Standardized status values to uppercase (FORMING/CONFIRMED)
  - Added pattern_id in SYMBOL_YYYYMMDD_STATUS format
  - Added new fields: height, avg_bottom, separation_days, forming_trigger_level
  - CONFIRMED patterns now always have breakout_date populated
  - Aligned confirmation logic with closing breakout detection
- 2025-12-23: Added complete backtesting strategy system
  - Created strategy.py with TradeSignal dataclass and generate_signals()
  - Created backtester.py with Backtest class for bar-by-bar simulation
  - Created metrics.py with comprehensive performance metrics
  - Created alerts.py with AlertManager for pattern breakout alerts
  - Updated main.py with --backtest mode and CLI arguments
- 2025-12-23: Initial implementation with full feature set

## Architecture Decisions
- Using scipy.signal.find_peaks for reliable swing detection
- Pattern strength scoring provides quality ranking
- Rate limiting (0.1s delay) prevents API blocks
- Modular design allows easy extension
