"""
End-to-end script: download data, train or load the HMM, emit signals, backtest,
and print a compact narrative so you can follow each step in the console.
"""

from __future__ import annotations

import sys

from backtest import run_backtest
from data import DEFAULT_TICKER, prepare_stock_data
from model import MODEL_PATH, ensure_model
from hmm_signal import generate_signals, print_latest_summary


def main(ticker: str = DEFAULT_TICKER) -> None:
    print("=" * 72)
    print("HMM stock prototype - full pipeline")
    print("=" * 72)

    print(f"\n[main] Step 1/4 - Download & prepare data for {ticker!r} ...")
    try:
        prep = prepare_stock_data(ticker=ticker)
    except Exception as exc:  # noqa: BLE001
        print(f"[main] Data preparation failed: {exc}", file=sys.stderr)
        raise

    print(
        f"  Rows after cleaning: {len(prep.df)} "
        f"(train: {prep.test_start_idx}, test: {len(prep.df) - prep.test_start_idx})"
    )

    print("\n[main] Step 2/4 - Train or load Gaussian HMM ...")
    bundle = ensure_model(prep, ticker=ticker, path=MODEL_PATH, force_retrain=False)
    # When loading a disk artifact, reuse its scaler so scaled inputs match training.
    prep.rescale_with(bundle["scaler"])

    print("\n[main] Step 3/4 - Generate trading signals on full history ...")
    dates = prep.df.index
    sig = generate_signals(bundle, prep.X_scaled_full, dates)
    print_latest_summary(bundle, sig, prep.df)

    print("\n[main] Step 4/4 - Backtest on the held-out test window ...")
    stats = run_backtest(prep, bundle, sig)

    print("\n[main] Done.")
    print(f"  Model file: {MODEL_PATH.resolve()}")
    print(f"  Chart file: {stats['chart_path']}")
    print("=" * 72)


if __name__ == "__main__":
    # Allow `python main.py MSFT` style invocation; default remains AAPL.
    t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    main(ticker=t.upper().strip())
