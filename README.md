# Smart Order Router Backtest

## Description
Backtests a Smart Order Router by implementing the Cont & Kukanov static cost model with an Almgren–Chriss inspired queue-impact term. It splits a large buy order across multiple venues to minimize cost and benchmarks performance against Best Ask, TWAP, and VWAP strategies.

## Dependencies
- Python 3.8+  
- numpy  
- pandas  
- matplotlib  

Install with:
```bash
pip install numpy pandas matplotlib
```

## File Structure
```
backtest.py                        # Main implementation
l1_day.csv                         # Level-1 market data feed
results.png    # Generated cost plot
README.md                          # Project documentation
```

## Key Functions

- **cont_kukanov_execution(df, order_size, λ_over, λ_under, θ_queue, step=100)**  
  Static allocator with queue-impact penalty:
  1. Defines `Venue` (ask, size, fees, rebates)  
  2. Enumerates splits (step granularity)  
  3. Computes cost = cash + under/over-fill penalties + θ_queue·(under+over)  

- **run_best_ask(df, order_size=5000)**  
  Always takes the lowest ask until filling the order.

- **run_vwap(df, order_size=5000)**  
  Computes running VWAP of displayed liquidity; executes when best ask ≤ VWAP.

- **run_twap(df, order_size=5000)**  
  Evenly slices the order into minute buckets, filling at the best asks.

- **plot_all_strategies(curves)**  
  Plots and saves cumulative cost curves to `results.png`.

- **format_output(best_params, cont_kuk, best_ask, twap, vwap, runtime)**  
  Prints and JSON dumps tuned parameters, execution metrics, savings (bps), and runtime.

## Main Script Workflow
1. Load and parse `ts_event` from `l1_day.csv`.  
2. Filter `action ∈ {A,M,C,R}` and dedupe on (`ts_event`,`publisher_id`).  
3. Compute `θ_queue = σ_mid_returns·√N` per Almgren–Chriss (2001).  
4. Grid-search over λ_over, λ_under ∈ [0.01–1.00] (4 points), θ_queue ∈ [0.5–2×σ√N] (4 points).  
5. Run baselines: Best Ask, TWAP, VWAP.  
6. Plot cost curves and output JSON summary.

## Usage
```bash
python backtest.py
```

## Suggested Improvements
- Simulate partial fills and queue dynamics for realistic slippage.  
- Use Bayesian or random search for parameter tuning.  
- Model permanent impact by adjusting mid-price post-trade.  
- Add real-time streaming and asynchronous order submissions.  
- Integrate unit tests and CI for maintainability.

## References
- Cont, R., & Kukanov, A. (2014). Optimal Order Placement in Limit Order Markets.  
- Almgren, R., & Chriss, N. (2001). Optimal Execution of Portfolio Transactions.  
