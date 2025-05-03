import json
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import sys
import time
#//=============================STRATEGIES=============================//

class Venue:
    def __init__(self, ask, ask_size, fee=0.0, rebate=0.0):
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate


def allocate_dp(order_size, venues, step, lambda_over, lambda_under, theta_queue):
    V = len(venues)
    M = order_size // step                   # number of buckets

    # 1) Precompute per-venue cost & execution tables
    cost_tables = []
    exec_tables = []
    for v in venues:
        cap = min(v.ask_size, order_size)
        K = cap // step
        c_tbl = [0.0] * (K + 1)
        e_tbl = [0] * (K + 1)
        for k in range(K + 1):
            q = k * step
            exe = min(q, v.ask_size)
            e_tbl[k] = exe
            cash = exe * (v.ask + v.fee)
            cash -= max(q - exe, 0) * v.rebate
            c_tbl[k] = cash
        cost_tables.append(c_tbl)
        exec_tables.append(e_tbl)

    # penalty shift constant
    C_pen = lambda_under + theta_queue
    INF = float('inf')

    # 2) Bottom-up DP
    dp = [INF] * (M + 1)
    dp[0] = 0.0
    back = [[0] * (M + 1) for _ in range(V + 1)]

    for i in range(V):
        new_dp = [INF] * (M + 1)
        K = len(cost_tables[i]) - 1
        for j in range(M + 1):
            if dp[j] == INF:
                continue
            for k in range(K + 1):
                nj = j + k
                if nj > M:
                    break
                val = dp[j] + cost_tables[i][k] - C_pen * exec_tables[i][k]
                if val < new_dp[nj]:
                    new_dp[nj] = val
                    back[i + 1][nj] = k
        dp = new_dp

    # 3) Reconstruct splits (in shares)
    splits = [0] * V
    rem = M
    for i in range(V, 0, -1):
        k = back[i][rem]
        splits[i - 1] = k * step
        rem -= k

    # 4) Recover true cost (undo shift + add penalties)
    executed = sum(min(splits[i], venues[i].ask_size) for i in range(V))
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    penalty = (
        lambda_under * underfill
        + lambda_over * overfill
        + theta_queue * (underfill + overfill)
    )

    # dp[M] = Σ[cash_i] − C_pen*executed  →  add back C_pen*order_size + λ_over*overfill
    best_cost = dp[M] + C_pen * order_size + lambda_over * overfill
    return splits, best_cost


def cont_kukanov_execution(df, order_size, lambda_over, lambda_under, theta_queue, step=100):
    remaining = order_size
    executed_total = 0
    cash = 0.0
    cumulative_costs = []

    for _, row in df.iterrows():
        if remaining <= 0:
            break

        # build venues
        venues = [
            Venue(row[f"ask_px_{i:02d}"], row[f"ask_sz_{i:02d}"])
            for i in range(10)
        ]
        ts = pd.to_datetime(row["ts_event"], unit="ns")

        # DP‐based allocation
        alloc_size = min(remaining, order_size)
        split, _ = allocate_dp(alloc_size, venues, step, lambda_over, lambda_under, theta_queue)

        # execute the split
        executed_now = 0
        cash_now = 0.0
        for q, v in zip(split, venues):
            exe = min(q, v.ask_size)
            executed_now += exe
            cash_now += exe * (v.ask + v.fee)
            cash_now -= max(q - exe, 0) * v.rebate

        cash += cash_now
        executed_total += executed_now
        remaining -= executed_now
        cumulative_costs.append((ts, cash))

    avg_price = cash / executed_total if executed_total else float("nan")
    return {
        "cash_spent": cash,
        "avg_fill_price": avg_price,
        "executed_shares": executed_total,
        "remaining_shares": order_size - executed_total,
        "cumulative_costs": cumulative_costs,
    }

def run_best_ask(df, order_size=5000):
    remaining = order_size
    cash = 0.0
    points = []
    for _, row in df.iterrows():
        if remaining <= 0:
            break
        venues = sorted([(row[f'ask_px_{i:02d}'], row[f'ask_sz_{i:02d}']) for i in range(10)], key=lambda x: x[0])
        ts = pd.to_datetime(row['ts_event'], unit='ns')
        for px, sz in venues:
            if remaining <= 0:
                break
            size = min(remaining, sz)
            cost = size * px
            cash += cost
            remaining -= size
            points.append((ts, cash))
    return cash, cash / order_size, points

def run_twap(df, order_size=5000):
    df = df.copy()
    times = pd.to_datetime(df['ts_event'], unit='ns')
    start = times.min()
    df['bucket'] = ((times - start).dt.total_seconds() // 60).astype(int)
    n_buckets = df['bucket'].nunique()
    target = order_size / n_buckets
    remaining = order_size
    cash = 0.0
    points = []
    for b in sorted(df['bucket'].unique()):
        if remaining <= 0:
            break
        bucket_df = df[df['bucket'] == b]
        if bucket_df.empty:
            continue
        first = bucket_df.iloc[0]
        ts = pd.to_datetime(first['ts_event'], unit='ns')
        venues = sorted([(first[f'ask_px_{i:02d}'], first[f'ask_sz_{i:02d}']) for i in range(10)], key=lambda x: x[0])
        alloc = min(remaining, target)
        for px, sz in venues:
            if alloc <= 0:
                break
            take = min(alloc, sz)
            cost = take * px
            cash += cost
            alloc -= take
            remaining -= take
            points.append((ts, cash))
    return cash, cash / order_size, points


def run_vwap(df, order_size=5000):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts_event'], utc=True)
    df = df.sort_values('ts').reset_index(drop=True)
    
    cumulative_volume = 0.0
    cumulative_dollar_volume = 0.0
    remaining = order_size
    total_cost = 0.0
    execution_records = []
    vwap_history = []
    
    for idx, row in df.iterrows():
        current_ts = row['ts']
        
        # 1. Aggregate best asks across all venues
        venue_data = []
        for i in range(10):
            px = row[f'ask_px_{i:02d}']
            sz = row[f'ask_sz_{i:02d}']
            if px > 0 and sz > 0:
                venue_data.append((px, sz))
        
        # 2. Check execution using PREVIOUS VWAP if available
        if vwap_history: 
            
            price_levels = {}
            for px, sz in venue_data:
                if px not in price_levels:
                    price_levels[px] = 0.0
                price_levels[px] += sz
            
            best_ask = min(price_levels.keys()) if price_levels else None
            best_size = price_levels.get(best_ask, 0) if best_ask else 0
            
            # Get previous VWAP
            prev_vwap = vwap_history[-1]
            
            # Execute only if valid comparison
            if prev_vwap and best_ask and best_ask <= prev_vwap:
                fill_qty = min(remaining, best_size)
                if fill_qty > 0:
                    fill_cost = fill_qty * best_ask
                    total_cost += fill_cost
                    remaining -= fill_qty
                    execution_records.append({
                        'timestamp': current_ts,
                        'quantity': fill_qty,
                        'price': best_ask,
                        'cumulative_cost': total_cost,
                        'venues_used': len([p for p, _ in venue_data if p == best_ask])
                    })
        
        # 3. Update VWAP with current data (AFTER execution check)
        current_volume = sum(sz for _, sz in venue_data)
        current_dollar = sum(px * sz for px, sz in venue_data)
        cumulative_volume += current_volume
        cumulative_dollar_volume += current_dollar
        
        current_vwap = (cumulative_dollar_volume / cumulative_volume 
                       if cumulative_volume > 0 else None)
        vwap_history.append(current_vwap)
        
        if remaining <= 0:
            break
    
    avg_price = total_cost / (order_size - remaining) if (order_size - remaining) > 0 else 0
    return (
        total_cost, 
        avg_price, 
        [(r['timestamp'], r['cumulative_cost']) for r in execution_records]
    )

def plot_all_strategies(curves):
    plt.figure(figsize=(12, 6))
    for label, points, color in curves:
        if points:
            ts, cs = zip(*points)
            plt.plot(ts, cs, label=label, color=color)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Cost ($)")
    plt.title("Cumulative Cost Comparison by Strategy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("execution_cost_comparison_1.png")
    plt.close()
    print("Saved plot: execution_cost_comparison.png")


def format_output(best_params, cont_kuk, best_ask, twap, vwap, runtime):
    savings = {
        "vs_best_ask": (best_ask[1] - cont_kuk['avg_fill_price']) / best_ask[1] * 1e4,
        "vs_twap": (twap[1] - cont_kuk['avg_fill_price']) / twap[1] * 1e4,
        "vs_vwap": (vwap[1] - cont_kuk['avg_fill_price']) / vwap[1] * 1e4
    }
    print(f"Best average price: {cont_kuk['avg_fill_price']:.2f}")
    output = {
        "best_parameters": best_params,
        "cont_kukanov": cont_kuk,
        "baselines": {
            "best_ask": {"cash_spent": best_ask[0], "avg_fill_price": best_ask[1]},
            "twap": {"cash_spent": twap[0], "avg_fill_price": twap[1]},
            "vwap": {"cash_spent": vwap[0], "avg_fill_price": vwap[1]}
        },
        "savings_bps": savings,
        "runtime_seconds": runtime
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    start_time = time.time()
    df = pd.read_csv('/Users/theboss/Library/Mobile Documents/com~apple~CloudDocs/Stevens/Internship Interview Process/Blockhouse/Quant Strategist Intern May 2025/l1_day.csv')
    df = df.sort_values('ts_event').reset_index(drop=True)

    all_bid_cols = [f'bid_px_{i:02d}' for i in range(10)]
    all_ask_cols = [f'ask_px_{i:02d}' for i in range(10)]
    bid_avg = df[all_bid_cols].mean(axis=1)
    ask_avg = df[all_ask_cols].mean(axis=1)
    mid = (bid_avg + ask_avg) / 2

    rets = mid.pct_change().dropna()
    theta_queue = rets.std() * np.sqrt(len(rets))

    MIN, MAX = 0.000001, 1.0

    lam_vals = np.linspace(MIN, MAX, 4)
    theta_vals = np.linspace(theta_queue * 0.5, theta_queue * 2, 4)
    best = {"cash_spent": float('inf')}

    for lam_over, lam_under, theta in product(lam_vals, lam_vals, theta_vals):
        result = cont_kukanov_execution(df, order_size=5000, lambda_over=lam_over, lambda_under=lam_under, theta_queue=theta)
        if result['cash_spent'] < best['cash_spent']:
            best.update(result)
            best.update({
                "lambda_over": lam_over,
                "lambda_under": lam_under,
                "theta_queue": theta
            })

    best_params = {
        "lambda_over": best['lambda_over'],
        "lambda_under": best['lambda_under'],
        "theta_queue": best['theta_queue']
    }
    cont_kuk = {
        "cash_spent": best['cash_spent'],
        "avg_fill_price": best['avg_fill_price'],
        "executed_shares": best['executed_shares'],
        "remaining_shares": best['remaining_shares']
    }

    best_ask = run_best_ask(df)
    twap = run_twap(df)
    vwap = run_vwap(df)
    runtime = time.time() - start_time

    plot_all_strategies([
        ("Cont-Kukanov", best['cumulative_costs'], 'tab:red'),
        ("Best Ask", best_ask[2], 'tab:blue'),
        ("TWAP", twap[2], 'tab:orange'),
        ("VWAP", vwap[2], 'tab:green')
    ])

    format_output(best_params, cont_kuk, best_ask, twap, vwap, runtime)
