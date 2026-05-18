# 📐 P2-ETF-HJB-STOCHASTIC-CONTROL

**P2Quant Engine** · Hamilton-Jacobi-Bellman PDE · Heston Stochastic Volatility · Optimal Portfolio Allocation

[![HJB Daily Inference](https://github.com/P2SAMAPA/P2-ETF-HJB-STOCHASTIC-CONTROL/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HJB-STOCHASTIC-CONTROL/actions/workflows/daily_run.yml)

---

## What Is This?

This engine solves the **Hamilton-Jacobi-Bellman PDE numerically** to find the mathematically
optimal portfolio weights under Heston stochastic volatility. It is the only engine in the
P2Quant suite that computes an explicit **hedging demand** — extra allocation to ETFs that
hedge the investor's exposure to rising volatility.

---

## Mathematical Framework

**Investor problem:** maximise expected power utility `E[W_T^(1-γ)/(1-γ)]`

**Heston dynamics:**
```
dS_t = μ S_t dt + √v_t S_t dW^S
dv_t = κ(θ - v_t) dt + ξ √v_t dW^v
corr(dW^S, dW^v) = ρ
```

**Ansatz:** `V(t,W,v) = W^(1-γ)/(1-γ) × f(t,v)` reduces the 3D HJB to a 2D PDE for `f(t,v)`:

```
f_t + A(v)·f_vv + B(v)·f_v + C(v)·f = 0
f(T, v) = 1
```

**Optimal weights decompose into two economically meaningful components:**

```
w*(i) = myopic_demand(i)  +  hedging_demand(i)

myopic_demand(i)  = μᵢ / (γ · σᵢᵢ · v)          ← Merton ratio
hedging_demand(i) = -(fᵥ/f) · ρᵢ · ξ / (γ · σᵢᵢ)  ← vol-risk hedge
```

The **hedging demand** is positive for ETFs with `ρ < 0` (return negatively correlated
with variance — typically TLT, GLD). These receive extra allocation not because their
expected returns improved, but because they hedge the investor's exposure to vol spikes.

---

## Scoring Formula

```python
myopic_z   = cross_sectional_zscore(myopic_demand)
hedging_z  = cross_sectional_zscore(hedging_demand)
composite  = 0.60 × myopic_z + 0.40 × hedging_z

# Stress regime (VIX > 25): shrink all scores
if vix > VIX_STRESS_THRESHOLD:
    composite *= (1 - STRESS_CASH_BOOST)

score = cross_sectional_zscore(composite)
```

---

## Numerical Solution: Crank-Nicolson PDE Solver

```
Grid:  Nv=100 variance points (v ∈ [1e-6, 0.25])
       Nt=63  time steps (T = 63/252 years = 1 quarter)

Scheme: Implicit-explicit Crank-Nicolson
        - Diffusion A(v)·f_vv + convection B(v)·f_v: implicit (tridiagonal system)
        - Reaction C(v)·f: explicit
        - Boundary: f(v=0) = exp(r·(T-t)), f(v=vmax) = 0
        - Solved via scipy.linalg.solve_banded — O(N) per time step

Runtime: ~0.1s per solve. Heston calibration: ~0.5s per ETF.
Total daily runtime: ~35s for 23 ETFs. No GPU needed.
```

---

## Heston Calibration

Per ETF, rolling 252-day window, refitted every 21 days:

```python
# Euler-Maruyama MLE on realised variance path
# v_{t+1} ≈ v_t + κ(θ - v_t)dt + ξ√(v_t·dt) · ε
# log p(v_{t+1}|v_t) = log N(v_{t+1}; μ_v, σ_v²)
# Optimised via scipy L-BFGS-B
```

Return-variance correlation `ρ` estimated as `corr(r_t, Δv_t)`.

---

## How It Differs From All 132 Existing Engines

| Property | MERTONANN (closest) | HJB Stochastic Control |
|---|---|---|
| Value function | Neural network approx | Exact finite-difference grid |
| Vol model | GBM (constant vol) | Heston (stochastic vol) |
| Hedging demand | Not computed | Explicitly computed from f_v/f |
| Optimality | Approximate | Exact (to grid discretisation) |
| Convergence | Training-dependent | Provably converges as Δv,Δt→0 |
| Interpretability | Black box | Myopic + hedging decomposition |
| Runtime | Slow (training) | Fast (~35s total, no training) |

---

## Universes

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM |
| FI_COMMODITIES | TLT VCIT LQD HYG VNQ GLD SLV |
| COMBINED | All above |

---

## Output Files (per universe)

| File | Content |
|---|---|
| `hjb_YYYY-MM-DD_{slug}.json` | Latest scores, Heston params, value function surface |
| `daily_{slug}.csv` | Top pick, score, VIX, regime, fv/f ratio |
| `scores_{slug}.csv` | Composite score history |
| `myopic_{slug}.csv` | Myopic demand history |
| `hedging_{slug}.csv` | Hedging demand history |
| `rankings_{slug}.csv` | Rank history |

**Results:** `P2SAMAPA/p2-etf-hjb-results`

---

## Streamlit Dashboard — 5 Tabs

1. **Rankings & Scores** — composite bar + myopic vs hedging stacked bar, top-N cards
2. **Value Function f(v)** — f(v) curve, f_v/f hedging multiplier, w*(v) vs VIX slider per ETF
3. **Myopic vs Hedging** — time-series of both demand components, VIX regime overlay
4. **Score History** — composite score time-series + heatmap, top-pick frequency
5. **Full Heston Table** — κ, θ, ξ, ρ, μ per ETF + engine config + daily summary

---

## References

- Merton (1971) *Optimum Consumption and Portfolio Rules in a Continuous-Time Model*
- Heston (1993) *A Closed-Form Solution for Options with Stochastic Volatility*
- Fleming & Rishel (1975) *Deterministic and Stochastic Optimal Control*
- Liu (2007) *Portfolio Selection in Stochastic Environments*
- Crank & Nicolson (1947) *A Practical Method for Numerical Evaluation of PDEs*

---

*P2Quant Engine Suite · Built by P2SAMAPA*
