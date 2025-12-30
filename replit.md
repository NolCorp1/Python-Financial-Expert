# NASDAQ Double Bottom Pattern Scanner

## Overview
This project is a Python program designed to identify "Double Bottom" (W) chart patterns in NASDAQ-listed stocks. It leverages algorithmic pattern detection, RSI divergence, and volume analysis on historical price data. The system includes a robust backtesting engine, a pattern quality scoring mechanism, and tools for optimizing and validating trading strategies. Its purpose is to provide a comprehensive solution for traders looking to identify and capitalize on double bottom patterns, offering capabilities for signal generation, performance analysis, and risk management.

## User Preferences
I prefer iterative development, so please propose changes and explain them thoroughly. I appreciate detailed explanations of complex concepts. Do not make changes to the `charts/` folder. Do not modify the `outputs/` folder directly; all generated outputs should be placed there by the scripts.

## System Architecture

### UI/UX Decisions
- **Chart Generation**: Uses Matplotlib for visualizing detected patterns and equity curves.
- **Console Output**: Provides formatted reports for metrics, feature attribution, and summary statistics.

### Technical Implementations
- **Pattern Detection**: Utilizes `scipy.signal.find_peaks` for identifying swing lows and highs, incorporating RSI divergence and volume analysis.
- **Data Acquisition**: Employs `yfinance` for historical stock price data.
- **Strategy & Backtesting**: Includes modules for defining trading strategies, a no-lookahead bar-by-bar simulation engine with risk-based sizing, slippage, and commissions, and comprehensive performance metrics calculation.
- **Optimization**: Facilitates walk-forward optimization with rolling train/test windows, parameter grid search, and objective-based scoring. Includes `BacktestContext` speedups through precomputation of window-invariant artifacts and correlation/cluster caching.
- **Robustness Features**:
    - **Run Manifests**: Generates manifests for provenance (command, git commit, package versions).
    - **Deterministic Runs**: Supports reproducible symbol selection and fixes for floating-point drift, ensuring deterministic trade outputs.
    - **Price Cache Policies**: Provides flexible price data caching policies (`AUTO`, `READONLY`, `REFRESH`, `OFF`) for reproducible runs, with data normalization and provenance tracking.
- **Pattern Quality Scoring**: A 0-100 score based on pre-entry data, incorporating symmetry, neckline, separation, breakout strength, volume, and trend context. Includes score inversion fixes and policy selection.
- **Score-Based Risk Scaling**: Scales position sizes based on pattern quality scores using a smooth multiplier function, with an absolute capital cap.
- **Portfolio Capital Allocation Engine**: Implements signal competition for a finite daily/weekly risk budget, ranking signals by allocation score and scaling proportionally.
- **Capital Recycling Engine**: Implements opportunity-cost exits, recycling lower-quality open positions to free budget for higher-quality signals. Includes various trigger modes, quality gates (`min_expected_edge_r`, `replace_only_if_improves_score`), and validation tools for robustness.
- **Market Regime Filter**: Incorporates MA-based trend and volatility detection to adjust risk.
- **Portfolio Risk Management**: Implements correlation and cluster caps, position sizing, and concurrent position limits.
- **Exit Strategy**: Includes partial take profit, ATR-based trailing stops, and no-progress rules.
- **Universe Management**: Handles NASDAQ symbol caching and liquidity filtering.

### Feature Specifications
- **Algorithmic Double Bottom Detection**: Identifies "W" patterns with configurable parameters.
- **RSI Divergence Confirmation**: Integrates bullish RSI divergence.
- **Volume Analysis**: Assesses volume signatures at key pattern points.
- **Configurable Parameters**: All key parameters are configurable via command-line arguments.
- **Output Generation**: Produces CSVs for patterns, trade blotters, equity curves, and performance reports.

### System Design Choices
- **Modular Design**: Structured into distinct Python modules for clarity and extensibility.
- **No-Lookahead Principle**: Ensures all trading decisions are based on available information at the time.
- **Risk-Based Sizing**: Positions are sized based on defined risk per trade.
- **Parameter Optimization**: Designed for comprehensive parameter optimization via walk-forward analysis.

## Paper Trading Workflow

The paper trading system supports daily operations for forward testing strategies without risking capital.

### Quick Start

```bash
# Run paper trading for today (or specify a date)
python main.py --paper-daily --universe demo --max-stocks 20

# Run for a specific date
python main.py --paper-daily --paper-date 2024-02-28 --universe demo

# Use cached price data only (for reproducibility)
python main.py --paper-daily --price-cache-policy READONLY

# Force rerun if already executed today
python main.py --paper-daily --force
```

### Key Features

- **Idempotent Runs**: Duplicate runs for the same date are automatically skipped (manifest-based detection)
- **Portfolio State Management**: Tracks positions, cash, equity, and history in `data/paper/portfolio_state.json`
- **Order Generation**: Creates orders with pattern metadata, stop prices, and take profits
- **Position Caps**: Enforces max positions (total and forming) per configuration
- **Audit Trail**: Every run generates a manifest with provenance (command, git commit, packages)

### Output Files

Each daily run creates outputs in `outputs/paper/YYYYMMDD/`:

| File | Description |
|------|-------------|
| `orders.csv` | Machine-readable order list |
| `orders.json` | Order details with metadata |
| `report.md` | Human-readable daily summary |
| `paper_manifest.json` | Provenance and reproducibility data |

### Portfolio State

The portfolio state file (`data/paper/portfolio_state.json`) contains:

- `equity`: Current portfolio value
- `cash`: Available cash balance
- `open_positions`: Active positions with entry/stop/metadata
- `closed_positions`: Historical closed trades
- `history`: Daily equity snapshots

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--paper-date` | today | Target date for signals |
| `--paper-force` | false | Override idempotence check |
| `--max-positions-total` | 15 | Maximum concurrent positions |
| `--max-positions-forming` | 5 | Maximum forming-stage positions |
| `--daily-risk-budget` | 0.01 | Daily risk allocation (1%) |

## External Dependencies
- **yfinance**: For downloading historical stock market data.
- **pandas**: For data manipulation and analysis.
- **scipy**: Specifically `scipy.signal.find_peaks` for peak detection.
- **matplotlib**: For generating charts and visualizations.
- **pandas-ta**: For technical indicators like RSI.
- **tqdm**: For displaying progress bars.
- **requests**: For making HTTP requests (e.g., for symbol lists).