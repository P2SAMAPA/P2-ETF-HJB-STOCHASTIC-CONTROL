"""hjb_solver.py — HJB PDE solver via Crank-Nicolson finite differences.

Mathematical setup
------------------
Value function ansatz for power utility U(W) = W^(1-gamma)/(1-gamma):

    V(t, W, v) = W^(1-gamma) / (1-gamma) * f(t, v)

Substituting into the HJB PDE and optimising over weights w gives the
reduced PDE for f(t, v):

    f_t + A(v)*f_vv + B(v)*f_v + C(v)*f = 0
    f(T, v) = 1   for all v

where (for single ETF with drift mu, idiosyncratic variance sigma_ii,
       Heston params kappa, theta, xi, rho):

    optimal myopic   = mu / (gamma * sigma_ii * v)
    optimal hedging  = -(f_v/f) * rho * xi / (gamma * sigma_ii)

    A(v) = 0.5 * xi^2 * v
    B(v) = kappa*(theta - v) - lambda_rp * xi * sqrt(v)   [market price of vol risk]
    C(v) = [sum of contributions from optimal weights]

For a universe of n ETFs:
    C(v) = r*(1-gamma) + (1-gamma)*[myopic_return_contribution]
         - 0.5*gamma*(1-gamma)*[myopic_variance_contribution]

Crank-Nicolson scheme
---------------------
Implicit-explicit split: A*f_vv + B*f_v terms treated implicitly,
C*f term treated explicitly. Results in tridiagonal system per time step.
Unconditionally stable for any time step size.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

import config
from heston import HestonParams


class HJBSolver:
    """Crank-Nicolson HJB PDE solver for Heston stochastic volatility model.

    Parameters
    ----------
    heston_params : list of HestonParams — one per ETF in universe
    mu_vec        : (n_etf,) annualised expected returns
    cov_matrix    : (n_etf, n_etf) annualised return covariance matrix
    risk_free     : float — annualised risk-free rate
    gamma         : float — relative risk aversion
    """

    def __init__(
        self,
        heston_params: list[HestonParams],
        mu_vec:        np.ndarray,
        cov_matrix:    np.ndarray,
        risk_free:     float = 0.04,
        gamma:         float = config.GAMMA,
    ) -> None:
        self.params     = heston_params
        self.mu         = mu_vec            # (n_etf,)
        self.cov        = cov_matrix        # (n_etf, n_etf)
        self.r          = risk_free
        self.gamma      = gamma
        self.n_etf      = len(heston_params)

        # Grid
        self.Nv         = config.NV
        self.Nt         = config.NT
        self.v_max      = config.V_MAX
        self.v_min      = config.V_MIN
        self.T          = config.HORIZON_DAYS / 252.0

        self.v_grid     = np.linspace(self.v_min, self.v_max, self.Nv)
        self.dv         = self.v_grid[1] - self.v_grid[0]
        self.dt         = self.T / self.Nt

        # Solution arrays
        self.f          = np.ones(self.Nv)       # f(t=T, v) = 1
        self.f_v        = np.zeros(self.Nv)      # df/dv — used for hedging demand

        # Aggregate Heston parameters (weighted mean across ETFs)
        kappas = np.array([p.kappa for p in heston_params])
        thetas = np.array([p.theta for p in heston_params])
        xis    = np.array([p.xi    for p in heston_params])
        rhos   = np.array([p.rho   for p in heston_params])

        self.kappa_agg  = float(np.mean(kappas))
        self.theta_agg  = float(np.mean(thetas))
        self.xi_agg     = float(np.mean(xis))
        self.rho_vec    = rhos     # (n_etf,) per-ETF return-variance correlation

        # Market price of vol risk (approximate as 0 — conservative)
        self.lambda_rp  = 0.0

    def _pde_coefficients(self, v: float) -> tuple[float, float, float]:
        """Compute A(v), B(v), C(v) at a single grid point.

        Returns (A, B, C) for the PDE: f_t + A*f_vv + B*f_v + C*f = 0
        """
        v   = max(v, 1e-8)
        g   = self.gamma
        r   = self.r

        # ── A(v): diffusion coefficient of variance process ────────────────
        A = 0.5 * self.xi_agg**2 * v

        # ── B(v): drift of variance process (risk-neutral) ────────────────
        B = (self.kappa_agg * (self.theta_agg - v)
             - self.lambda_rp * self.xi_agg * np.sqrt(v))

        # ── C(v): contribution from optimal portfolio choice ───────────────
        # Myopic demand per ETF: w_i* = mu_i / (gamma * sigma_ii * v)
        # where sigma_ii = cov[i,i] = idiosyncratic + systematic variance
        diag_cov = np.diag(self.cov)
        sigma_ii = np.maximum(diag_cov, 1e-8)

        w_myopic  = self.mu / (g * sigma_ii * v)
        # Clip to reasonable range
        w_myopic  = np.clip(w_myopic, -5.0, 5.0)
        # Normalise so weights sum to ≤ 1 (long-only constraint approximate)
        w_sum = np.sum(np.maximum(w_myopic, 0))
        if w_sum > 1.0:
            w_myopic = np.maximum(w_myopic, 0) / w_sum

        # Portfolio return contribution
        port_mu   = float(np.dot(w_myopic, self.mu))

        # Portfolio variance contribution
        port_var  = float(w_myopic @ self.cov @ w_myopic * v)

        # C(v) = (1-g)*r + (1-g)*port_mu - 0.5*g*(1-g)*port_var
        C = ((1.0 - g) * r
             + (1.0 - g) * port_mu
             - 0.5 * g * (1.0 - g) * port_var)

        return A, B, C

    def _build_tridiagonal(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build tridiagonal matrix coefficients for Crank-Nicolson step.

        Returns (lower, main, upper) diagonal arrays for banded solver.
        """
        N  = self.Nv
        dv = self.dv
        dt = self.dt

        lower = np.zeros(N)
        main  = np.zeros(N)
        upper = np.zeros(N)

        for i in range(1, N - 1):
            v   = self.v_grid[i]
            A, B, C = self._pde_coefficients(v)

            # Crank-Nicolson: 0.5 implicit + 0.5 explicit
            # Diffusion: A * f_vv ≈ A * (f_{i+1} - 2f_i + f_{i-1}) / dv²
            a_d = 0.5 * dt * A / dv**2

            # Convection: B * f_v ≈ B * (f_{i+1} - f_{i-1}) / (2*dv)
            a_c = 0.5 * dt * B / (2 * dv)

            # Reaction: C * f_i
            a_r = 0.5 * dt * C

            lower[i] = -(a_d - a_c)
            main[i]  =  1.0 + 2.0 * a_d - a_r
            upper[i] = -(a_d + a_c)

        # Boundary conditions
        # v = v_min: f ~ exp(r*(T-t)) (no vol → deterministic growth)
        main[0]  = 1.0
        upper[0] = 0.0

        # v = v_max: f = 0 (infinite vol → all wealth lost)
        main[-1]  = 1.0
        lower[-1] = 0.0

        return lower, main, upper

    def solve(self) -> np.ndarray:
        """Solve HJB PDE backward from T to 0.

        Returns f(t=0, v) array of shape (Nv,).
        """
        f     = np.ones(self.Nv)   # terminal condition f(T,v) = 1
        N     = self.Nv
        dv    = self.dv
        dt    = self.dt

        lower, main, upper = self._build_tridiagonal()

        # Banded matrix format for scipy.linalg.solve_banded
        # ab[0] = upper diagonal, ab[1] = main, ab[2] = lower
        ab = np.zeros((3, N))
        ab[0, 1:]  = upper[:-1]   # upper: shift up by 1
        ab[1, :]   = main
        ab[2, :-1] = lower[1:]    # lower: shift down by 1

        for step in range(self.Nt):
            # Explicit RHS: (I - 0.5*dt*L)*f_old
            rhs = f.copy()
            for i in range(1, N - 1):
                v = self.v_grid[i]
                A, B, C = self._pde_coefficients(v)
                a_d = 0.5 * dt * A / dv**2
                a_c = 0.5 * dt * B / (2 * dv)
                a_r = 0.5 * dt * C

                rhs[i] = (f[i]
                          + a_d * (f[i+1] - 2*f[i] + f[i-1])
                          + a_c * (f[i+1] - f[i-1])
                          + a_r * f[i])

            # Boundary conditions on RHS
            rhs[0]  = np.exp(self.r * (self.T - step * dt))
            rhs[-1] = 0.0

            # Implicit solve
            f = solve_banded((1, 1), ab, rhs)
            f = np.maximum(f, 1e-10)   # keep f positive

        self.f = f

        # Compute f_v = df/dv via central differences
        self.f_v = np.gradient(f, self.v_grid)

        return f

    def optimal_weights(self, v_current: float) -> dict:
        """Extract optimal portfolio weights at current variance v_current.

        Returns dict with myopic_demand, hedging_demand, total_weight per ETF.
        """
        v   = np.clip(v_current, self.v_min, self.v_max)
        g   = self.gamma

        # Interpolate f and f_v at current v
        f_now   = float(np.interp(v, self.v_grid, self.f))
        fv_now  = float(np.interp(v, self.v_grid, self.f_v))

        # f_v / f ratio (vol hedging multiplier)
        fv_over_f = fv_now / max(f_now, 1e-8)

        diag_cov = np.diag(self.cov)
        sigma_ii = np.maximum(diag_cov, 1e-8)

        # Myopic demand: mu_i / (gamma * sigma_ii * v)
        w_myopic  = self.mu / (g * sigma_ii * v)

        # Hedging demand: -(f_v/f) * rho_i * xi_agg / (gamma * sigma_ii)
        # Positive for ETFs with rho_i < 0 (negatively correlated with vol)
        w_hedging = (-fv_over_f * self.rho_vec * self.xi_agg
                     / (g * sigma_ii))

        # Total HJB optimal weight
        w_total   = w_myopic + w_hedging

        # Clip and normalise to long-only (approximate constraint)
        w_total   = np.clip(w_total, -2.0, 5.0)
        w_myopic  = np.clip(w_myopic, -2.0, 5.0)
        w_hedging = np.clip(w_hedging, -2.0, 2.0)

        return {
            "myopic":   w_myopic.copy(),
            "hedging":  w_hedging.copy(),
            "total":    w_total.copy(),
            "f_now":    f_now,
            "fv_over_f": fv_over_f,
            "v_current": v,
        }

    def value_function_surface(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (v_grid, f) for visualisation in dashboard."""
        return self.v_grid.copy(), self.f.copy()
