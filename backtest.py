"""
Evaluate the simple BUY/HOLD/SELL rule on the held-out test window.

We compare a cautious long-only strategy (cash unless a BUY opens a position,
SELL flattens) against buying once at the first test close and holding through
the end — a sanity baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM

from data import PreparedData
from hmm_signal import SignalResult


CHART_PATH = Path(__file__).resolve().parent / "backtest_chart.png"


def _regime_color(bundle: Dict[str, Any], hmm_state: int) -> str:
    if hmm_state == bundle["bull_state"]:
        return "#c8f7c5"  # light green
    if hmm_state == bundle["bear_state"]:
        return "#f7c5c5"  # light red
    return "#e0e0e0"  # gray sideways


def simulate_long_only(
    closes: np.ndarray,
    signals: List[str],
) -> Tuple[float, int, int, int]:
    """
    Start with 1.0 cash, 0 shares.

    BUY: if flat, deploy all cash at today's close.
    SELL: if long, liquidate at today's close.
    HOLD: keep the current position.
    """
    cash = 1.0
    shares = 0.0
    n_buy = n_sell = n_hold = 0

    for price, sig in zip(closes, signals, strict=True):
        if sig == "BUY":
            n_buy += 1
            if shares == 0.0 and cash > 0.0:
                shares = cash / price
                cash = 0.0
        elif sig == "SELL":
            n_sell += 1
            if shares > 0.0:
                cash = shares * price
                shares = 0.0
        else:
            n_hold += 1

    terminal = cash + shares * float(closes[-1])
    return terminal, n_buy, n_sell, n_hold


def buy_and_hold(closes: np.ndarray) -> float:
    """Spend $1 on the first close; mark to market on the last close."""
    if len(closes) < 2:
        return 1.0
    shares = 1.0 / closes[0]
    return float(shares * closes[-1])


def run_backtest(
    prep: PreparedData,
    bundle: Dict[str, Any],
    sig: SignalResult,
    chart_path: Path = CHART_PATH,
) -> Dict[str, Any]:
    """
    Restrict performance metrics to the test split, but draw the chart across
    the full sample so you can see regimes in context.
    """
    model: GaussianHMM = bundle["model"]
    df = prep.df
    test_start = prep.test_start_idx

    dates = df.index
    closes = df["Close"].to_numpy(dtype=np.float64)

    # Test-window signals only (aligned with prep.X_test rows).
    test_dates = dates[test_start:]
    test_sig = sig.signals[test_start:]
    test_closes = closes[test_start:]

    strat_end = simulate_long_only(test_closes, test_sig)[0]
    bh_end = buy_and_hold(test_closes)

    strat_ret = strat_end - 1.0
    bh_ret = bh_end - 1.0

    n_buy = sum(1 for s in test_sig if s == "BUY")
    n_sell = sum(1 for s in test_sig if s == "SELL")
    n_hold = sum(1 for s in test_sig if s == "HOLD")

    print("\n[backtest] Test-window performance (last 30% of history)")
    print(f"  Strategy ending value (from $1): {strat_end:.4f}  (P/L: {strat_ret:+.2%})")
    print(f"  Buy & hold ending value (from $1): {bh_end:.4f}  (P/L: {bh_ret:+.2%})")
    print(f"  Signal counts - BUY: {n_buy}, HOLD: {n_hold}, SELL: {n_sell}")

    # Decode most likely state path for background shading.
    X_full = prep.X_scaled_full
    states = model.predict(X_full)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Colored spans for contiguous runs of the same decoded state.
    i0 = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[i0]:
            ax.axvspan(
                dates[i0],
                dates[i - 1],
                color=_regime_color(bundle, int(states[i0])),
                alpha=0.9,
                linewidth=0,
            )
            i0 = i

    ax.plot(dates, closes, color="black", linewidth=1.0, label="Close")
    ax.set_title("Close price with HMM regime shading (green=bull, red=bear, gray=sideways)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"[backtest] Saved chart to {chart_path}")

    return {
        "strategy_end": strat_end,
        "buy_hold_end": bh_end,
        "strategy_return": strat_ret,
        "buy_hold_return": bh_ret,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_hold": n_hold,
        "chart_path": str(chart_path),
    }
