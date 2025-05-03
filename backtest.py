import json
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import time

# 
def cont_kukanov_execution(df, order_size, lambda_over, lambda_under, theta_queue, step=100):
    class Venue:
        def __init__(self, ask, ask_size, fee=0.0, rebate=0.0):
            self.ask = ask
            self.ask_size = ask_size
            self.fee = fee
            self.rebate = rebate

    def allocate(order_size, venues):
        splits = [[]]
        for v in range(len(venues)):
            new_splits = []
            for alloc in splits:
                used = sum(alloc)
                max_v = min(order_size - used, venues[v].ask_size)
                for q in range(0, int(max_v) + 1, step):
                    new_splits.append(alloc + [q])
            splits = new_splits

        best_cost = float('inf')
        best_split = []
        for alloc in splits:
            if sum(alloc) != order_size:
                continue
            cost = compute_cost(alloc, venues, order_size)
            if cost < best_cost:
                best_cost = cost
                best_split = alloc
        return best_split, best_cost

    def compute_cost(split, venues, order_size):
        executed = 0
        cash_spent = 0
        for i in range(len(venues)):
            exe = min(split[i], venues[i].ask_size)
            executed += exe
            cash_spent += exe * (venues[i].ask + venues[i].fee)
            rebate = max(split[i] - exe, 0) * venues[i].rebate
            cash_spent -= rebate

        underfill = max(order_size - executed, 0)
        overfill = max(executed - order_size, 0)
        penalty = (lambda_under * underfill + lambda_over * overfill) + theta_queue * (underfill + overfill)
        return cash_spent + penalty

    remaining = order_size
    executed_total = 0
    cash = 0.0
    cumulative_costs = []

    for _, row in df.iterrows():
        if remaining <= 0:
            break

        venues = [Venue(row[f'ask_px_{i:02d}'], row[f'ask_sz_{i:02d}']) for i in range(10)]
        ts = pd.to_datetime(row['ts_event'], unit='ns')

        alloc_size = min(remaining, order_size)
        split, _ = allocate(alloc_size, venues)
        if not split:
            continue

        executed_now = 0
        cash_now = 0
        for q, v in zip(split, venues):
            exe = min(q, v.ask_size)
            executed_now += exe
            cash_now += exe * (v.ask + v.fee)
            cash_now -= max(q - exe, 0) * v.rebate

        cash += cash_now
        executed_total += executed_now
        remaining -= executed_now
        cumulative_costs.append((ts, cash))

    avg_price = cash / executed_total if executed_total else float('nan')
    return {
        "cash_spent": cash,
        "avg_fill_price": avg_price,
        "executed_shares": executed_total,
        "remaining_shares": order_size - executed_total,
        "cumulative_costs": cumulative_costs
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


# ========== VWAP STRATEGY (Streaming + Impact) ========== #
def run_vwap(df, order_size=5000):
    remaining = order_size
    cash = 0.0
    executed_total = 0
    cumulative_costs = []

    cumulative_dollar_volume = 0.0
    cumulative_volume = 0.0

    for _, row in df.iterrows():
        if remaining <= 0:
            break

        ts = pd.to_datetime(row['ts_event'], unit='ns')

        # Gather visible prices and sizes
        prices = []
        sizes = []
        for i in range(10):
            px = row[f'ask_px_{i:02d}']
            sz = row[f'ask_sz_{i:02d}']
            if px > 0 and sz > 0:
                prices.append(px)
                sizes.append(sz)

        prices = np.array(prices)
        sizes = np.array(sizes)
        total_liquidity = sizes.sum()
        if total_liquidity == 0:
            continue

        # Update cumulative VWAP with visible data at this tick (up to this point)
        cumulative_dollar_volume += (prices * sizes).sum()
        cumulative_volume += total_liquidity

        current_vwap = cumulative_dollar_volume / cumulative_volume if cumulative_volume > 0 else None
        if current_vwap is None:
            continue

        # Check best ask
        best_ask = np.min(prices)
        if best_ask <= current_vwap:
            # Execute as much as possible at or below best ask
            alloc = 0
            venues = sorted(zip(prices, sizes), key=lambda x: x[0])
            for px, sz in venues:
                if px > current_vwap or remaining <= 0:
                    break
                trade = min(remaining, sz)
                alloc += trade
                cash += trade * px
                remaining -= trade
                executed_total += trade
                if remaining <= 0:
                    break
            if alloc > 0:
                cumulative_costs.append((ts, cash))

    avg_price = cash / executed_total if executed_total > 0 else float('nan')
    return cash, avg_price, cumulative_costs

# ========== TWAP STRATEGY (Distributed) ========== #
def run_twap(df, order_size=5000):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts_event'], unit='ns')
    df['bucket'] = ((df['ts'] - df['ts'].min()).dt.total_seconds() // 60).astype(int)
    buckets = df['bucket'].unique()
    n_buckets = len(buckets)
    per_bucket = order_size / n_buckets
    remaining = order_size
    cash = 0.0
    points = []

    for b in buckets:
        if remaining <= 0:
            break
        bucket_df = df[df['bucket'] == b]
        alloc = min(per_bucket, remaining)
        for _, row in bucket_df.iterrows():
            ts = pd.to_datetime(row['ts_event'], unit='ns')
            venues = sorted([(row[f'ask_px_{i:02d}'], row[f'ask_sz_{i:02d}']) for i in range(10)], key=lambda x: x[0])
            for px, sz in venues:
                if alloc <= 0:
                    break
                take = min(alloc, sz)
                cash += take * px
                alloc -= take
                remaining -= take
                points.append((ts, cash))
            if alloc <= 0:
                break
    return cash, cash / order_size, points

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
    plt.savefig("results.png")
    plt.close()
    print("Saved plot: results.png")


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
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')
    df = df.sort_values('ts_event').reset_index(drop=True)
    book_actions = ['A', 'M', 'C', 'R']  # only book-modifying messages
    df = df[df['action'].isin(book_actions)].copy()
    # deduplicate: keep first message per timestamp & publisher
    df = df.drop_duplicates(['ts_event', 'publisher_id'], keep='first').reset_index(drop=True)
    all_bid_cols = [f'bid_px_{i:02d}' for i in range(10)]
    all_ask_cols = [f'ask_px_{i:02d}' for i in range(10)]
    bid_avg = df[all_bid_cols].mean(axis=1)
    ask_avg = df[all_ask_cols].mean(axis=1)
    mid = (bid_avg + ask_avg) / 2

    rets = mid.pct_change().dropna()
    theta_queue = rets.std() * np.sqrt(len(rets))

    MIN, MAX = 0.01, 1.0

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
