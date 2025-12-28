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
- 2025-12-28: Market Regime Filter (Task 11)
  - Added BacktestConfig fields: use_regime_filter, regime_symbol, regime_trend_fast_ma, regime_trend_slow_ma, regime_vol_lookback, regime_vol_high_threshold, regime_disable_forming_in_downtrend, regime_disable_forming_in_high_vol, regime_reduce_risk_in_high_vol, regime_high_vol_risk_multiplier
  - compute_regime() helper: MA-based trend (50/200) + ATR% volatility detection
  - 1-bar shift for no-lookahead guarantee
  - Entry gating: skip FORMING signals in downtrend or high-vol
  - Dynamic risk: 70% risk multiplier in high volatility
  - Regime symbol (QQQ/SPY) excluded from pattern detection/trading
  - Skip counters: skipped_regime_forming_downtrend, skipped_regime_forming_highvol
  - CLI: --use-regime-filter, --regime-symbol, --regime-fast-ma, --regime-slow-ma, --regime-vol-lookback, --regime-vol-high-threshold, --regime-disable-forming-in-downtrend, --regime-disable-forming-in-high-vol, --regime-reduce-risk-in-high-vol, --regime-high-vol-risk-multiplier
  - Results: Max DD improved from -5.73% to -5.22% with regime filter enabled
- 2025-12-28: Correlation & Cluster Caps (Task 10)
  - Added BacktestConfig fields: use_correlation_caps, corr_lookback_days, max_corr_to_existing, use_cluster_caps, n_clusters, max_positions_per_cluster
  - Hierarchical clustering (scipy) for symbol grouping
  - Entry caps: skip if correlation >= 0.8 or cluster already at limit
  - Skip counters: skipped_corr_cap, skipped_cluster_cap
  - No-lookahead: correlation computed from data strictly before first signal date
  - CLI: --use-correlation-caps, --corr-lookback-days, --max-corr-to-existing, --use-cluster-caps, --n-clusters, --max-positions-per-cluster
- 2025-12-24: Walk-Forward v2 (Task 9)
  - Expanded parameter grid: portfolio/exit knobs (risk_fraction_forming, max_positions_forming, FORMING exit rules, CONFIRMED partial TP/trailing)
  - Composite scoring: trade-count penalty, exposure penalty, max DD constraint
  - Enhanced diagnostics: train_trade_count, train_total_return_pct, test_exposure_days in CSV
  - Stability summary v2: priority parameter win counts, aggregate stats
  - New CLI: --objective composite, --min-trades-test 20, --min-exposure-days-test 20, --grid-size-limit 250
- 2025-12-24: Exit Upgrades (Task 8)
  - CONFIRMED partial take profit: sell 50% at +1R, continue with remainder
  - ATR-based trailing stop: 2x ATR distance, activates after +1R or partial TP
  - FORMING TIGHTEN_STOP mode: alternative to EXIT for no-progress rule
  - New CLI flags: --confirmed-partial-tp-enabled, --confirmed-trailing-enabled, etc.
  - No-lookahead: trailing uses prior day close/ATR, partial TP uses current day OHLC
  - Results: marginal improvement with partial TP (7.62% vs 7.61% baseline)
- 2025-12-24: Portfolio Rules Engine (Task 7)
  - Kind-specific risk: CONFIRMED 1% vs FORMING 0.6% per trade
  - Position caps: max 3 concurrent FORMING positions, 10 total
  - FORMING exit rules: TIME (60 days), NO_PROGRESS (<0.5R after 20 days)
  - CONFIRMED breakeven stop at +0.5R MFE
  - Skip counters track rejected signals by reason
  - CLI flags: --risk-confirmed, --risk-forming, --max-positions-forming, etc.
  - Results: Max DD improved from -13% to -5%, FORMING hold days 84→30
- 2025-12-23: Universe Expansion & Liquidity Filter (Task 6)
  - New universe.py module with NASDAQ symbol caching and liquidity filtering
  - get_nasdaq_symbols_cached() downloads and caches NASDAQ symbol list
  - passes_liquidity_filter() checks min price ($5) and avg dollar volume (20M)
  - CLI: --universe demo|nasdaq|custom, --min-price, --min-dollar-vol
  - Outputs liquidity_filter_report.csv with per-symbol diagnostics
- 2025-12-23: Walk-Forward Optimization (Task 5)
  - optimize.py with rolling train/test windows
  - Parameter grid search (~50-150 combinations)
  - OOS scoring with drawdown constraints
  - Stability summary tracking which params win most often
  - Outputs: walkforward_results.csv, walkforward_best_params.json, walkforward_summary.txt
  - CLI: python optimize.py --symbols AAPL NVDA AMD --train-bars 504 --test-bars 126
- 2025-12-23: Metrics & Diagnostics (Task 4)
  - enrich_trades() adds derived columns (win, abs_r, capped_r, year, month)
  - compute_trade_metrics() for core stats (win_rate, expectancy, profit_factor)
  - compute_equity_metrics() for equity stats (returns, drawdown, Sharpe)
  - compute_split_metrics() splits by ALL/FORMING/CONFIRMED
  - print_metrics_report() for formatted console output
  - --backtest-v2 flag in main.py outputs metrics.json
- 2025-12-23: No-Lookahead Backtester (Task 3)
  - BacktestConfig with risk-based sizing (1% equity risk per trade)
  - run_backtest() returns (trades_df, equity_df)
  - Gap-aware exits: gap-through stop/target handled, conservative conflict resolution
  - Slippage (5 bps) + commission applied to both entry and exit
  - Trade blotter with: pattern_id, entry_kind, pnl_r_multiple, meta_json
  - Daily equity curve with drawdown_pct
  - Overlap control: one_position_per_symbol, max_positions
  - No lookahead: uses prior close for position valuation during sizing
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
