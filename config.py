"""config.py — HJB Stochastic Control Engine configuration.

Mathematical framework
----------------------
Investor maximises expected power utility E[W_T^(1-gamma)/(1-gamma)]
subject to Heston stochastic volatility dynamics.

HJB PDE separates via ansatz V(t,W,v) = W^(1-gamma)/(1-gamma) * f(t,v)
reducing to 2D PDE for f(t,v) solved backward on a finite-difference grid.

Optimal weights decompose into:
  w*(i) = myopic_demand(i) + hedging_demand(i)
         = mu_i/(gamma*sigma_ii*v)  -  (f_v/f)*rho_i*xi/(gamma*sigma_ii)

The hedging demand term is unique to this engine — no other engine in the
suite computes the explicit vol-hedging allocation.
"""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE   = "master_data.parquet"
HF_MODEL_REPO  = "P2SAMAPA/p2-etf-hjb-model"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-hjb-results"
HF_TOKEN       = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS       = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS":  EQUITY_SECTORS_TICKERS,
    "FI_COMMODITIES":  FI_COMMODITIES_TICKERS,
    "COMBINED":        COMBINED_TICKERS,
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Investor preferences ──────────────────────────────────────────────────────
GAMMA              = 3.0    # relative risk aversion (1=log utility, 3=moderate, 5=conservative)
HORIZON_DAYS       = 63     # investment horizon T in trading days (1 quarter)

# ── PDE grid parameters ───────────────────────────────────────────────────────
NV                 = 100    # number of variance grid points
NT                 = 63     # number of time steps (= HORIZON_DAYS for daily resolution)
V_MAX              = 0.25   # max annualised variance on grid (VIX ~158% — extreme upper bound)
V_MIN              = 1e-6   # min variance (strictly positive for Heston)

# ── Heston model calibration ──────────────────────────────────────────────────
HESTON_WINDOW      = 252    # rolling days for Heston parameter estimation
HESTON_REFIT_FREQ  = 21     # refit Heston params every N days (monthly)
# Heston parameter bounds for MLE optimisation
KAPPA_BOUNDS       = (0.1, 20.0)   # mean reversion speed
THETA_BOUNDS       = (1e-4, 0.50)  # long-run variance
XI_BOUNDS          = (0.01, 2.0)   # vol-of-vol
RHO_BOUNDS         = (-0.99, 0.99) # return-variance correlation

# ── Realised volatility estimator ─────────────────────────────────────────────
# Parkinson (1980) estimator uses High-Low range when available,
# falls back to close-to-close if OHLCV not available
VOL_ESTIMATOR      = "close_to_close"   # "parkinson" | "close_to_close"
VOL_WINDOW         = 21                  # rolling window for current vol estimate

# ── Scoring weights ───────────────────────────────────────────────────────────
MYOPIC_WT          = 0.60   # weight on myopic demand in composite score
HEDGING_WT         = 0.40   # weight on hedging demand in composite score
# Note: these are for composite z-score blending, not the raw HJB weights

# ── Stress regime conditioning ────────────────────────────────────────────────
VIX_STRESS_THRESHOLD = 25.0    # VIX above → stress regime
STRESS_CASH_BOOST    = 0.15    # extra CASH allocation in stress (reduce all weights by this)

# ── CASH threshold ────────────────────────────────────────────────────────────
CASH_THRESHOLD     = -0.40     # composite z-score below → recommend CASH
TOP_N              = 6

# ── OOS start ─────────────────────────────────────────────────────────────────
OOS_START          = "2010-01-01"   # first date scores published (need HESTON_WINDOW history)

# ── Output ────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")
