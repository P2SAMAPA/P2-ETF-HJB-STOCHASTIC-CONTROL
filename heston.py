"""heston.py — Heston stochastic volatility model calibration.

Heston (1993) model:
    dS_t = mu * S_t dt + sqrt(v_t) * S_t dW^S_t
    dv_t = kappa*(theta - v_t) dt + xi * sqrt(v_t) dW^v_t
    corr(dW^S, dW^v) = rho * dt

Parameters
----------
kappa : float  — mean reversion speed of variance
theta : float  — long-run variance (annualised)
xi    : float  — vol-of-vol (volatility of variance process)
rho   : float  — correlation between return and variance innovations

Calibration method
------------------
Euler-Maruyama discretisation of variance process:
    v_{t+1} ≈ v_t + kappa*(theta - v_t)*dt + xi*sqrt(max(v_t,0)*dt) * eps
    eps ~ N(0,1)

Log-likelihood of transition v_t → v_{t+1}:
    log p(v_{t+1}|v_t) = log N(v_{t+1}; mu_v, sigma_v²)
    mu_v    = v_t + kappa*(theta - v_t)*dt
    sigma_v = xi * sqrt(max(v_t,0) * dt)

Return-variance correlation rho estimated separately via:
    rho = corr(r_t, delta_v_t) where delta_v_t = v_t - v_{t-1}
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import config


@dataclass
class HestonParams:
    """Container for fitted Heston parameters per ETF."""
    kappa:    float   # mean reversion speed
    theta:    float   # long-run variance
    xi:       float   # vol-of-vol
    rho:      float   # return-variance correlation
    mu:       float   # annualised drift (expected return)
    sigma:    float   # average annualised vol = sqrt(theta)
    loglik:   float   # log-likelihood of fit
    n_obs:    int     # number of observations used


def _heston_loglik(
    params: np.ndarray,
    v_series: np.ndarray,
    dt: float = 1.0 / 252,
) -> float:
    """Negative log-likelihood of Heston variance path under Euler discretisation."""
    kappa, theta, xi = params
    if kappa <= 0 or theta <= 0 or xi <= 0:
        return 1e10

    ll = 0.0
    for t in range(1, len(v_series)):
        v_prev = max(v_series[t - 1], 1e-8)
        v_next = v_series[t]
        mu_v   = v_prev + kappa * (theta - v_prev) * dt
        sig_v  = xi * np.sqrt(v_prev * dt) + 1e-10
        ll    += norm.logpdf(v_next, mu_v, sig_v)

    return -ll


def calibrate_heston(
    returns: np.ndarray,
    var_series: np.ndarray,
) -> HestonParams:
    """Fit Heston parameters to realised variance path.

    Parameters
    ----------
    returns    : (T,) daily log returns
    var_series : (T,) daily realised variance (annualised)

    Returns
    -------
    HestonParams with fitted kappa, theta, xi, rho, mu, sigma
    """
    if len(var_series) < 20:
        return _fallback_params(returns, var_series)

    # ── Moment-based initial estimates ────────────────────────────────────────
    theta0 = float(np.mean(var_series))
    kappa0 = 2.0
    xi0    = float(np.std(np.diff(var_series)) * np.sqrt(252)) + 0.01
    xi0    = np.clip(xi0, 0.05, 1.5)

    x0     = np.array([kappa0, theta0, xi0])
    bounds = [
        config.KAPPA_BOUNDS,
        config.THETA_BOUNDS,
        config.XI_BOUNDS,
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _heston_loglik,
            x0,
            args=(var_series,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )

    if result.success and result.fun < 1e9:
        kappa, theta, xi = result.x
    else:
        kappa, theta, xi = kappa0, theta0, xi0

    # ── Return-variance correlation via sample correlation ────────────────────
    if len(returns) > 2 and len(var_series) > 2:
        min_len = min(len(returns), len(var_series) - 1)
        delta_v = np.diff(var_series[:min_len + 1])
        ret_ali = returns[:min_len]
        if len(delta_v) > 2 and np.std(delta_v) > 1e-10 and np.std(ret_ali) > 1e-10:
            rho = float(np.corrcoef(ret_ali, delta_v)[0, 1])
            rho = float(np.clip(rho, -0.99, 0.99))
        else:
            rho = -0.50
    else:
        rho = -0.50

    mu    = float(np.mean(returns) * 252)
    sigma = float(np.sqrt(max(theta, 1e-8)))

    return HestonParams(
        kappa=float(np.clip(kappa, *config.KAPPA_BOUNDS)),
        theta=float(np.clip(theta, *config.THETA_BOUNDS)),
        xi=float(np.clip(xi, *config.XI_BOUNDS)),
        rho=rho,
        mu=mu,
        sigma=sigma,
        loglik=float(-result.fun) if result.success else 0.0,
        n_obs=len(var_series),
    )


def _fallback_params(
    returns: np.ndarray,
    var_series: np.ndarray,
) -> HestonParams:
    """Fallback Heston params when data is insufficient."""
    mu    = float(np.mean(returns) * 252) if len(returns) > 0 else 0.05
    theta = float(np.mean(var_series)) if len(var_series) > 0 else 0.04
    return HestonParams(
        kappa=2.0, theta=theta, xi=0.30, rho=-0.50,
        mu=mu, sigma=float(np.sqrt(theta)),
        loglik=0.0, n_obs=len(var_series),
    )


def build_variance_series(
    returns: np.ndarray,
    window: int = config.VOL_WINDOW,
) -> np.ndarray:
    """Build rolling realised variance series (annualised) from log returns.

    Uses rolling window standard deviation, annualised by 252.
    """
    T   = len(returns)
    var = np.full(T, np.nan)
    for t in range(window, T):
        var[t] = np.var(returns[t - window: t], ddof=1) * 252
    # Backfill the initial NaNs with first valid value
    first_valid = window
    if first_valid < T:
        var[:first_valid] = var[first_valid]
    return np.nan_to_num(var, nan=0.04)
