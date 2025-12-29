# NASDAQ Double Bottom Pattern Scanner

## Overview
This project is a Python program designed to identify "Double Bottom" (W) chart patterns in NASDAQ-listed stocks. It leverages algorithmic pattern detection, RSI divergence, and volume analysis on historical price data from yfinance. The system includes a robust backtesting engine for strategy evaluation, a pattern quality scoring mechanism, and tools for optimizing and validating trading strategies. Its purpose is to provide a comprehensive solution for traders looking to identify and capitalize on double bottom patterns, offering capabilities for signal generation, performance analysis, and risk management within various market regimes.

## User Preferences
I prefer iterative development, so please propose changes and explain them thoroughly. I appreciate detailed explanations of complex concepts. Do not make changes to the `charts/` folder. Do not modify the `outputs/` folder directly; all generated outputs should be placed there by the scripts.

## System Architecture

### UI/UX Decisions
- **Chart Generation**: Uses Matplotlib to generate visual representations of detected patterns and equity curves for analysis.
- **Console Output**: Provides clear, formatted reports for metrics, feature attribution, and summary statistics in the console.

### Technical Implementations
- **Pattern Detection**: Utilizes `scipy.signal.find_peaks` for accurate identification of swing lows and highs.
- **Data Acquisition**: Employs `yfinance` for downloading historical stock price data.
- **Algorithmic Core**: `double_bottom_scanner.py` houses the core pattern detection logic, including RSI divergence and volume analysis.
- **Strategy & Backtesting**:
    - `strategy.py`: Defines the trading strategy and signal generation, supporting both "FORMING" and "CONFIRMED" pattern entries.
    - `backtester.py`: Implements a no-lookahead, bar-by-bar simulation engine with risk-based sizing, slippage, and commissions.
    - `metrics.py`: Calculates comprehensive performance metrics (win rate, Sharpe ratio, drawdown, etc.) and equity statistics.
- **Optimization**: `optimize.py` facilitates walk-forward optimization with rolling train/test windows, parameter grid search, and objective-based scoring.
- **BacktestContext Speedup** (Task 18): Precomputes window-invariant artifacts (ATR series, returns matrix, date calendar, regime data) once per walk-forward window and reuses across all grid iterations. Achieves ~2.6-3.2x speedup over the previous per-combo computation approach. Includes a correlation/cluster cache keyed by `(first_signal_date, lookback_days, n_clusters)` to avoid redundant portfolio-level computations.
- **Robustness Features**:
    - **Run Manifests**: Every execution generates a manifest for full provenance, including command, git commit, and package versions.
    - **Deterministic Runs**: Supports `--symbols-seed` for reproducible symbol selection across runs.
    - **Parity Check**: Verifies identical results between runs using a given configuration.
    - **Price Cache Integrity**: Tools to generate and repair price cache reports.
- **Pattern Quality Scoring**: A 0-100 score based on pre-entry data, incorporating symmetry, neckline, separation, breakout strength, volume, and trend context.
- **Score Inversion Fix**: Diagnostic and adjustment tools (`--score-policy INVERT`, `--trend-score-mode NEUTRAL`) for scenarios where high scores underperform.
- **Score Policy Selection & Production Defaults**: Walk-forward aggregation by scoring knobs (`compute_score_policy_wf_summary`), stability-first selector (`select_best_scoring_defaults`), and enhanced validation mode with dual backtests.
  - **Outputs**: `score_policy_wf_summary.csv` (aggregated results by scoring config), `chosen_scoring_defaults.json` (selected defaults with reasoning), `validation_scoring_recommendation.txt` (comparison report).
  - **Monotonicity Check**: `check_score_monotonicity()` in `metrics.py` uses Spearman correlation to detect score inversion issues.
- **Score-Based Risk Scaling** (Task 19): Scales position sizes based on pattern quality scores. Higher-quality patterns receive larger positions, lower-quality patterns receive smaller positions. Uses a smooth multiplier function: `score_mult = min_mult + (max_mult - min_mult) * (score/100)^alpha`. Default settings: `min_mult=0.60`, `max_mult=1.20`, `alpha=1.0`. Applies to both FORMING and CONFIRMED entries. Includes an absolute cap (`max_risk_fraction_per_trade=0.02`) to prevent excessive risk. No lookahead: only uses pre-entry pattern score.
- **Market Regime Filter**: Incorporates a market regime filter with soft-gating using MA-based trend and volatility detection to adjust risk.
- **Correlation & Cluster Caps**: Implements portfolio-level risk management by limiting positions based on symbol correlation and cluster membership.
- **Exit Strategy**: Includes partial take profit, ATR-based trailing stops, and no-progress rules for "FORMING" patterns.
- **Portfolio Rules Engine**: Manages position sizing, concurrent position limits, and specific exit rules for different pattern "kinds" (FORMING/CONFIRMED).
- **Universe Management**: `universe.py` handles NASDAQ symbol caching and liquidity filtering (min price, min dollar volume).

### Feature Specifications
- **Algorithmic Double Bottom Detection**: Identifies "W" patterns with configurable price tolerance, peak height, and separation days.
- **RSI Divergence Confirmation**: Integrates bullish RSI divergence as a pattern quality factor.
- **Volume Analysis**: Assesses volume signatures at key pattern points, particularly decreased volume on the second bottom.
- **Configurable Parameters**: All key detection, backtesting, and portfolio parameters are configurable via command-line arguments.
- **Output Generation**: Produces CSV outputs for detected patterns, trade blotters, equity curves, and various performance reports.
- **Alert System**: Designed to alert on confirmed pattern breakouts (implementation details are modular).

### System Design Choices
- **Modular Design**: Structured into distinct Python modules (`main.py`, `double_bottom_scanner.py`, `strategy.py`, `backtester.py`, `metrics.py`, `alerts.py`) for clarity and extensibility.
- **No-Lookahead Principle**: Ensures all trading decisions in the backtester are based solely on information available at that point in time.
- **Risk-Based Sizing**: Positions are sized based on a defined risk per trade (e.g., a percentage of equity).
- **Parameter Optimization**: Designed for comprehensive parameter optimization through walk-forward analysis.

## External Dependencies
- **yfinance**: For downloading historical stock market data.
- **pandas**: For data manipulation and analysis.
- **scipy**: Specifically `scipy.signal.find_peaks` for peak detection.
- **matplotlib**: For generating charts and visualizations.
- **pandas-ta**: For technical indicators like RSI.
- **tqdm**: For displaying progress bars during long operations.
- **requests**: For making HTTP requests, likely used in `universe.py` for symbol lists.