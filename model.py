"""
Train a 3-state Gaussian HMM on scaled features, pick the best random restart,
label states by average log-return (bull / sideways / bear), and persist the bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from data import PreparedData


MODEL_PATH = Path(__file__).resolve().parent / "hmm_model.pkl"
N_COMPONENTS = 3
N_INIT_RUNS = 10
COVARIANCE_TYPE = "full"


def _posteriors(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """hmmlearn versions differ slightly; both APIs give per-row state probabilities."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=np.float64)
    _, post = model.score_samples(X)
    return np.asarray(post, dtype=np.float64)


def label_states_by_return(
    model: GaussianHMM,
    train_df: pd.DataFrame,
    X_train: np.ndarray,
) -> Tuple[int, int, int]:
    """
    After training, decode training states and rank them by mean log_return.

    Highest average return → bull, lowest → bear, middle → sideways.
    Returns (bull_state, bear_state, sideways_state) as HMM state indices 0..2.
    """
    states = model.predict(X_train)
    means: Dict[int, float] = {}
    for s in range(N_COMPONENTS):
        mask = states == s
        if not np.any(mask):
            means[s] = float("nan")
        else:
            means[s] = float(train_df["log_return"].to_numpy()[mask].mean())

    # If a state never appears, push it to the end so ranking still works.
    ranked = sorted(range(N_COMPONENTS), key=lambda k: (np.isnan(means[k]), means[k]))

    bear_state = ranked[0]
    sideways_state = ranked[1]
    bull_state = ranked[2]
    return bull_state, bear_state, sideways_state


def print_state_stats(
    model: GaussianHMM,
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    bull_state: int,
    bear_state: int,
    sideways_state: int,
) -> None:
    """Human-readable sanity check: each regime's average return and volatility."""
    states = model.predict(X_train)
    log_ret = train_df["log_return"].to_numpy()
    vol = train_df["volatility"].to_numpy()

    def summarize(name: str, state_id: int) -> None:
        m = states == state_id
        if not np.any(m):
            print(f"  {name} (state {state_id}): no training assignments")
            return
        print(
            f"  {name} (state {state_id}): "
            f"mean log_return={log_ret[m].mean():.6f}, "
            f"mean volatility={vol[m].mean():.6f}"
        )

    print("[model] Per-state training statistics (unscaled log_return / volatility):")
    summarize("BULL", bull_state)
    summarize("SIDEWAYS", sideways_state)
    summarize("BEAR", bear_state)


def train_best_hmm(
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    n_runs: int = N_INIT_RUNS,
) -> Tuple[GaussianHMM, int, int, int, float]:
    """
    Train several HMMs with different seeds and keep the one with highest train log-likelihood.
    """
    best_model: GaussianHMM | None = None
    best_score = -np.inf
    best_tuple: Tuple[int, int, int] | None = None

    for seed in range(n_runs):
        model = GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type=COVARIANCE_TYPE,
            random_state=seed,
            n_iter=2000,
            tol=1e-4,
            init_params="stmc",
        )
        try:
            model.fit(X_train)
        except Exception as exc:  # noqa: BLE001
            print(f"[model] Fit failed for seed={seed}: {exc}", file=sys.stderr)
            continue

        score = float(model.score(X_train))
        if score > best_score:
            best_score = score
            best_model = model
            bull_s, bear_s, side_s = label_states_by_return(model, train_df, X_train)
            best_tuple = (bull_s, bear_s, side_s)

    if best_model is None or best_tuple is None:
        raise RuntimeError("All HMM training attempts failed.")

    bull_state, bear_state, sideways_state = best_tuple
    return best_model, bull_state, bear_state, sideways_state, best_score


def save_bundle(
    path: Path,
    model: GaussianHMM,
    scaler: Any,
    bull_state: int,
    bear_state: int,
    sideways_state: int,
    ticker: str,
) -> None:
    """Persist everything inference needs (model + scaler + state semantics + ticker)."""
    bundle = {
        "model": model,
        "scaler": scaler,
        "bull_state": bull_state,
        "bear_state": bear_state,
        "sideways_state": sideways_state,
        "ticker": ticker,
    }
    joblib.dump(bundle, path)


def load_bundle(path: Path = MODEL_PATH) -> Dict[str, Any]:
    return joblib.load(path)


def train_and_save(prep: PreparedData, ticker: str, path: Path = MODEL_PATH) -> Dict[str, Any]:
    train_df = prep.df.iloc[: prep.test_start_idx]

    model, bull_state, bear_state, sideways_state, score = train_best_hmm(
        prep.X_train,
        train_df,
        n_runs=N_INIT_RUNS,
    )

    print(f"[model] Best train log-likelihood after {N_INIT_RUNS} runs: {score:.4f}")
    print(
        f"[model] State map -> bull={bull_state}, sideways={sideways_state}, bear={bear_state}"
    )
    print_state_stats(model, train_df, prep.X_train, bull_state, bear_state, sideways_state)

    save_bundle(path, model, prep.scaler, bull_state, bear_state, sideways_state, ticker)
    print(f"[model] Saved bundle to {path}")
    return load_bundle(path)


def ensure_model(
    prep: PreparedData,
    ticker: str,
    path: Path = MODEL_PATH,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Load an existing bundle when present; otherwise train and write hmm_model.pkl.

    `force_retrain` ignores the file and retrains from scratch.
    If a saved bundle belongs to a different ticker, we retrain automatically so
    states and scaling stay coherent with the symbol you requested.
    """
    if not force_retrain and path.is_file():
        bundle = load_bundle(path)
        saved = str(bundle.get("ticker") or "").upper()
        if saved and saved != ticker.upper():
            print(
                f"[model] Saved model is for {saved!r} but you asked for {ticker!r}; retraining..."
            )
            return train_and_save(prep, ticker, path=path)

        print(f"[model] Loading existing model from {path}")
        return bundle

    print("[model] No saved model (or retrain requested); training a new HMM...")
    return train_and_save(prep, ticker, path=path)
