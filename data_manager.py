"""data_manager.py — Data loading for HJB Stochastic Control engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ALL_TICKERS = sorted(set(
    config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS
))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download master_data.parquet → (log_returns, macro_df)."""
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

    available   = [t for t in ALL_TICKERS if t in df.columns]
    prices      = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    macro_cols = [c for c in config.MACRO_COLS if c in df.columns]
    macro_df   = df[macro_cols].reindex(log_returns.index).ffill().fillna(0.0)

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs"
        f" | Macro: {macro_cols}"
    )
    return log_returns, macro_df


def realised_variance(
    returns: np.ndarray,
    window: int = config.VOL_WINDOW,
) -> float:
    """Compute current realised daily variance from close-to-close returns.

    Returns annualised variance v_t = (252 × rolling_std²).
    """
    if len(returns) < 2:
        return 0.01  # fallback: ~10% annual vol
    recent = returns[-window:] if len(returns) >= window else returns
    return float(np.var(recent, ddof=1) * 252)


def vix_to_variance(vix_value: float) -> float:
    """Convert VIX index value to annualised daily variance.

    VIX = 100 × √(annualised_variance) → v = (VIX/100)²
    """
    return float((max(vix_value, 0.01) / 100.0) ** 2)
