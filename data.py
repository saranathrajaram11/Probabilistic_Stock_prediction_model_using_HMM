"""
Download stock history with yfinance, build HMM input features, scale them,
and split into train/test slices in chronological order.

This module keeps time order everywhere (no shuffling) so the model only
learns from the past when we evaluate on the test window.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# Default symbol — change this one constant to switch the demo ticker.
DEFAULT_TICKER = "AAPL"
YEARS_OF_HISTORY = 7
TRAIN_FRACTION = 0.70


@dataclass
class PreparedData:
    """Everything downstream code needs after preparation."""

    # Full history after feature engineering and NaN removal (unscaled features).
    df: pd.DataFrame
    # Raw feature matrix (same row order as df) — use the saved model scaler when loading.
    X_unscaled: np.ndarray
    # Observation matrix for the HMM, scaled the same way train/test were built.
    X_scaled_full: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    # Row index in df where the test split starts (inclusive).
    test_start_idx: int
    scaler: StandardScaler

    def rescale_with(self, scaler: StandardScaler) -> None:
        """
        Rebuild scaled train/test matrices using an external scaler (for example the
        one stored next to a previously trained HMM). This keeps online inference
        aligned with the saved model when yfinance returns a longer history than
        the artifact was trained on.
        """
        X = self.X_unscaled
        n_train = self.test_start_idx
        self.scaler = scaler
        self.X_train = scaler.transform(X[:n_train])
        self.X_test = scaler.transform(X[n_train:])
        self.X_scaled_full = np.vstack([self.X_train, self.X_test])


def download_ohlcv(ticker: str, years: int = YEARS_OF_HISTORY) -> pd.DataFrame:
    """
    Pull daily OHLCV from Yahoo Finance.

    If the download fails (network, delisted symbol, etc.), we surface a clear
    message instead of a raw stack trace.
    """
    try:
        raw = yf.download(
            ticker,
            period=f"{years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001 — yfinance can raise many types
        print(
            f"[data] yfinance could not download '{ticker}': {exc}\n"
            "Check the ticker spelling and your internet connection, then try again.",
            file=sys.stderr,
        )
        raise

    if raw is None or raw.empty:
        print(
            f"[data] No rows returned for '{ticker}'. "
            "The symbol may be invalid or Yahoo Finance may have no data.",
            file=sys.stderr,
        )
        raise ValueError(f"Empty dataframe for ticker {ticker!r}")

    # yfinance sometimes returns a MultiIndex when downloading multiple tickers;
    # flatten so the rest of the pipeline always sees simple column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    return raw


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    From raw OHLCV, compute log return, short-horizon volatility, and volume
    pressure vs its recent average — three signals an HMM can use to cluster
    market regimes.
    """
    df = ohlcv.copy()
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # 1) Log return: symmetric percentage move in log space.
    df["log_return"] = np.log(close / close.shift(1))

    # 2) Volatility: rolling std of log returns (5 trading days).
    df["volatility"] = df["log_return"].rolling(window=5).std()

    # 3) Volume ratio: today's volume vs its 20-day average.
    vol_ma20 = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / vol_ma20

    return df


def prepare_stock_data(
    ticker: str = DEFAULT_TICKER,
    years: int = YEARS_OF_HISTORY,
    train_fraction: float = TRAIN_FRACTION,
) -> PreparedData:
    """
    End-to-end: download → features → drop NaNs → scale (fit on train only) → split.

    Returns the processed dataframe (unscaled feature columns for interpretation)
    plus numpy matrices ready for hmmlearn.
    """
    raw = download_ohlcv(ticker, years=years)
    df = build_features(raw)

    feature_cols = ["log_return", "volatility", "volume_ratio"]
    df = df.dropna(subset=feature_cols).copy()

    n = len(df)
    if n < 50:
        raise ValueError(f"Not enough clean rows after dropna (got {n}).")

    # Time-ordered split: first 70% train, last 30% test.
    n_train = int(np.floor(n * train_fraction))
    n_train = max(n_train, 1)
    n_train = min(n_train, n - 1)

    X_unscaled = df[feature_cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train_raw = X_unscaled[:n_train]
    X_test_raw = X_unscaled[n_train:]
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_scaled_full = np.vstack([X_train, X_test])

    test_start_idx = n_train

    return PreparedData(
        df=df,
        X_unscaled=X_unscaled,
        X_scaled_full=X_scaled_full,
        X_train=X_train,
        X_test=X_test,
        test_start_idx=test_start_idx,
        scaler=scaler,
    )


def main_cli() -> None:
    """Tiny manual smoke test when you run `python data.py`."""
    prep = prepare_stock_data(DEFAULT_TICKER)
    print(f"Rows: {len(prep.df)}, train: {prep.test_start_idx}, test: {len(prep.df) - prep.test_start_idx}")


if __name__ == "__main__":
    main_cli()
