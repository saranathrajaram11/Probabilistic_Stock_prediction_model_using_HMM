"""
Streamlit front-end: pick a ticker, refresh data, load/retrain the HMM, and
inspect regimes, signals, and the saved backtest chart.

Run from the project folder:
  streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from backtest import CHART_PATH, run_backtest
from data import DEFAULT_TICKER, prepare_stock_data
from model import MODEL_PATH, ensure_model, train_and_save
from hmm_signal import attach_signals_to_frame, generate_signals, print_latest_summary


def regime_html(bundle: dict, state_idx: int) -> str:
    if state_idx == bundle["bull_state"]:
        return '<span style="color:white;background:#2ca02c;padding:6px 12px;border-radius:6px;font-weight:700;">BULL</span>'
    if state_idx == bundle["bear_state"]:
        return '<span style="color:white;background:#d62728;padding:6px 12px;border-radius:6px;font-weight:700;">BEAR</span>'
    return '<span style="color:white;background:#7f7f7f;padding:6px 12px;border-radius:6px;font-weight:700;">SIDEWAYS</span>'


def main() -> None:
    st.set_page_config(page_title="HMM Stock Prototype", layout="wide")
    st.title("Hidden Markov Model — regime & signal dashboard")

    ticker = st.text_input("Stock ticker", value=DEFAULT_TICKER).upper().strip()
    col_a, col_b = st.columns(2)
    with col_a:
        retrain = st.button("Force retrain model", help="Ignores hmm_model.pkl and fits a fresh HMM.")
    with col_b:
        run = st.button("Run pipeline", type="primary")

    if not (run or retrain):
        st.info("Set a ticker and click **Run pipeline** (or force a retrain).")
        return

    with st.spinner("Downloading and preparing data…"):
        try:
            prep = prepare_stock_data(ticker=ticker)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Data error: {exc}")
            return

    with st.spinner("Training or loading HMM…"):
        try:
            if retrain:
                bundle = train_and_save(prep, ticker=ticker, path=MODEL_PATH)
            else:
                bundle = ensure_model(prep, ticker=ticker, path=MODEL_PATH, force_retrain=False)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Model error: {exc}")
            return

    prep.rescale_with(bundle["scaler"])

    sig = generate_signals(bundle, prep.X_scaled_full, prep.df.index)

    # Echo the same console summary underneath for quick auditing.
    print_latest_summary(bundle, sig, prep.df)

    with st.spinner("Backtesting and refreshing chart…"):
        try:
            run_backtest(prep, bundle, sig, chart_path=CHART_PATH)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Backtest/chart issue: {exc}")

    last_state = int(sig.proba[-1].argmax())
    st.markdown("### Current market regime (most likely today)")
    st.markdown(regime_html(bundle, last_state), unsafe_allow_html=True)

    sig_word = sig.signals[-1]
    color = {"BUY": "#2ca02c", "SELL": "#d62728", "HOLD": "#7f7f7f"}[sig_word]
    st.markdown("### Today's signal")
    st.markdown(
        f'<div style="font-size:56px;font-weight:800;color:{color};">{sig_word}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Tomorrow's predicted regime")
    order = [
        ("BULL", bundle["bull_state"]),
        ("SIDEWAYS", bundle["sideways_state"]),
        ("BEAR", bundle["bear_state"]),
    ]
    rows = []
    for name, sid in order:
        rows.append({"regime": name, "probability": float(sig.tomorrow_proba[int(sid)])})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        f"Argmax regime: **{sig.tomorrow_regime}** "
        f"(max P = {float(sig.tomorrow_proba.max()):.3f})"
    )

    st.markdown("### Backtest chart")
    chart_file = Path(CHART_PATH)
    if chart_file.is_file():
        st.image(str(chart_file), use_container_width=True)
    else:
        st.warning("Chart not found yet — run the full pipeline once.")

    st.markdown("### Last 10 days of signals")
    table = attach_signals_to_frame(prep.df, bundle, sig).tail(10).copy()
    if "Close" in table.columns:
        show = table[["Close", "regime", "signal", "p_bull"]].copy()
    else:
        show = table[["regime", "signal", "p_bull"]].copy()
    st.dataframe(show, use_container_width=True)


if __name__ == "__main__":
    # When executed as a script outside Streamlit, point users to the right command.
    if "streamlit" not in sys.modules:  # pragma: no cover
        print("Launch with: streamlit run dashboard.py", file=sys.stderr)
    main()
