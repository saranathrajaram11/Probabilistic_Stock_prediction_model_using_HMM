"""
Load the trained HMM and translate state probabilities into trading signals.

We use posterior probabilities (soft beliefs) for today's signal thresholds,
and multiply today's posterior by the transition matrix to peek at tomorrow's
most likely regime in expectation.

Note: This file is named hmm_signal.py (not signal.py) so it does not shadow
Python's standard library `signal` module when you run `python -m pip` from
this project folder on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def _posteriors(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=np.float64)
    _, post = model.score_samples(X)
    return np.asarray(post, dtype=np.float64)


def _regime_name(bundle: Dict[str, Any], state_idx: int) -> str:
    if state_idx == bundle["bull_state"]:
        return "BULL"
    if state_idx == bundle["bear_state"]:
        return "BEAR"
    if state_idx == bundle["sideways_state"]:
        return "SIDEWAYS"
    return f"STATE_{state_idx}"


@dataclass
class SignalResult:
    """Per-row outputs for backtests and dashboards."""

    dates: pd.DatetimeIndex
    proba: np.ndarray  # (n_days, 3) aligned to HMM state ids
    signals: List[str]
    tomorrow_regime: str
    tomorrow_proba: np.ndarray  # distribution from last day forward one step


def generate_signals(bundle: Dict[str, Any], X: np.ndarray, dates: pd.DatetimeIndex) -> SignalResult:
    """
    For each row of scaled features, compute P(state), then map to BUY/HOLD/SELL.

    Rules:
      - P(bull) > 0.55 → BUY
      - P(bear) > 0.55 → SELL
      - else → HOLD
    """
    model: GaussianHMM = bundle["model"]
    bull = int(bundle["bull_state"])
    bear = int(bundle["bear_state"])

    proba = _posteriors(model, X)
    p_bull = proba[:, bull]
    p_bear = proba[:, bear]

    signals: List[str] = []
    for i in range(len(proba)):
        if p_bull[i] > 0.55:
            signals.append("BUY")
        elif p_bear[i] > 0.55:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    # One-step lookahead: expected mixture tomorrow if today's posterior is correct.
    last = proba[-1]
    trans = np.asarray(model.transmat_, dtype=np.float64)
    next_proba = last @ trans
    tomorrow_state = int(np.argmax(next_proba))
    tomorrow_regime = _regime_name(bundle, tomorrow_state)

    return SignalResult(
        dates=dates,
        proba=proba,
        signals=signals,
        tomorrow_regime=tomorrow_regime,
        tomorrow_proba=next_proba,
    )


def print_latest_summary(bundle: Dict[str, Any], sig: SignalResult, df: pd.DataFrame) -> None:
    """Print today's regime (argmax state), signal, and tomorrow's predicted regime."""
    last_idx = -1
    today_state = int(np.argmax(sig.proba[last_idx]))
    today_regime = _regime_name(bundle, today_state)
    today_signal = sig.signals[last_idx]

    last_date = sig.dates[last_idx]
    print("\n[signal] Latest day summary")
    print(f"  As-of date: {last_date.date()}")
    print(f"  Today's most likely regime: {today_regime}")
    print(f"  Today's signal: {today_signal}")
    print(
        f"  Tomorrow's predicted regime (argmax): {sig.tomorrow_regime} "
        f"(P={sig.tomorrow_proba.max():.3f})"
    )
    print(
        "  Tomorrow's regime distribution over [BULL, SIDEWAYS, BEAR] "
        f"(HMM ids {bundle['bull_state']}, {bundle['sideways_state']}, {bundle['bear_state']}): "
        f"{np.array2string(sig.tomorrow_proba, precision=3)}"
    )

    # Optional: show last close for context
    if "Close" in df.columns:
        print(f"  Last close: {float(df['Close'].iloc[last_idx]):.2f}")


def attach_signals_to_frame(df: pd.DataFrame, bundle: Dict[str, Any], sig: SignalResult) -> pd.DataFrame:
    """Convenience for Streamlit tables: align signals with the feature dataframe."""
    out = df.copy()
    out = out.loc[sig.dates]
    out["signal"] = sig.signals
    bull = int(bundle["bull_state"])
    out["p_bull"] = sig.proba[:, bull]
    regimes = [_regime_name(bundle, int(row.argmax())) for row in sig.proba]
    out["regime"] = regimes
    return out
