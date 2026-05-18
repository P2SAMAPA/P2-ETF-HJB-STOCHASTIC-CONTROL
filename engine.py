"""engine.py — HJB Stochastic Control walk-forward engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import data_manager
from heston import calibrate_heston, build_variance_series, HestonParams
from hjb_solver import HJBSolver


def zscore_cross(arr: np.ndarray) -> np.ndarray:
    mu  = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mu) / std


def run_engine(
    log_returns: pd.DataFrame,
    macro_df:    pd.DataFrame,
    universe_tickers: list[str],
    universe_name:    str,
) -> dict:
    avail  = [t for t in universe_tickers if t in log_returns.columns]
    n_etf  = len(avail)
    mac_c  = [c for c in config.MACRO_COLS if c in macro_df.columns]
    has_vix = "VIX" in mac_c

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({n_etf} ETFs)\n"
        f"Period: {log_returns.index[0].date()} → {log_returns.index[-1].date()}"
        f"  ({len(log_returns)} days)\n{'='*60}"
    )

    ret_arr = log_returns[avail].values.astype(np.float64)
    dates   = log_returns.index
    vix_arr = macro_df["VIX"].values if has_vix else np.full(len(dates), 20.0)
    tbill   = macro_df["TBILL_3M"].values if "TBILL_3M" in mac_c else np.full(len(dates), 4.0)

    oos_start      = pd.Timestamp(config.OOS_START)
    last_heston_t  = -config.HESTON_REFIT_FREQ
    heston_cache   : list[HestonParams | None] = [None] * n_etf
    solver         : HJBSolver | None = None

    score_records, myopic_records, hedging_records = [], [], []
    ranking_records, daily_records = [], []
    n_scored = 0

    for t in range(config.HESTON_WINDOW, len(ret_arr)):
        date = dates[t]
        if date < oos_start:
            continue

        # ── Refit Heston params if due ────────────────────────────────────────
        if (t - last_heston_t) >= config.HESTON_REFIT_FREQ or solver is None:
            win_rets = ret_arr[t - config.HESTON_WINDOW: t]  # (252, n_etf)
            new_params = []
            for i in range(n_etf):
                r_i   = win_rets[:, i]
                var_i = build_variance_series(r_i, window=config.VOL_WINDOW)
                hp    = calibrate_heston(r_i, var_i)
                new_params.append(hp)
            heston_cache = new_params

            # Build covariance matrix and mu vector from training window
            mu_vec   = win_rets.mean(axis=0) * 252
            cov_mat  = np.cov(win_rets.T) * 252
            cov_mat  = np.maximum(cov_mat, np.eye(n_etf) * 1e-6)

            rf = float(tbill[t] / 100.0) if t < len(tbill) else 0.04

            solver = HJBSolver(
                heston_params=heston_cache,
                mu_vec=mu_vec,
                cov_matrix=cov_mat,
                risk_free=rf,
                gamma=config.GAMMA,
            )
            solver.solve()
            last_heston_t = t
            print(
                f"  Heston refit @ {date.date()} | "
                f"avg_kappa={np.mean([p.kappa for p in heston_cache]):.2f}  "
                f"avg_theta={np.mean([p.theta for p in heston_cache]):.4f}  "
                f"rf={rf:.3f}"
            )

        # ── Current variance state ────────────────────────────────────────────
        vix_today = float(vix_arr[t]) if t < len(vix_arr) else 20.0
        v_today   = data_manager.vix_to_variance(vix_today)
        high_stress = vix_today > config.VIX_STRESS_THRESHOLD

        # ── Extract optimal weights ───────────────────────────────────────────
        weights_dict = solver.optimal_weights(v_today)
        w_myopic  = weights_dict["myopic"]    # (n_etf,)
        w_hedging = weights_dict["hedging"]   # (n_etf,)
        w_total   = weights_dict["total"]     # (n_etf,)

        # ── Composite score ───────────────────────────────────────────────────
        myopic_z  = zscore_cross(w_myopic)
        hedging_z = zscore_cross(w_hedging)
        composite = (config.MYOPIC_WT  * myopic_z
                     + config.HEDGING_WT * hedging_z)

        # Stress regime: shrink all weights toward zero
        if high_stress:
            composite = composite * (1.0 - config.STRESS_CASH_BOOST)

        composite_z = zscore_cross(composite)
        ranked_idx  = np.argsort(composite_z)[::-1]
        top_ticker  = avail[ranked_idx[0]]
        top_score   = float(composite_z[ranked_idx[0]])
        cash_flag   = top_score < config.CASH_THRESHOLD

        ds = date.strftime("%Y-%m-%d")
        n_scored += 1

        score_records.append({"date": ds,
            **{avail[i]: round(float(composite_z[i]), 6) for i in range(n_etf)}})
        myopic_records.append({"date": ds,
            **{avail[i]: round(float(w_myopic[i]), 6) for i in range(n_etf)}})
        hedging_records.append({"date": ds,
            **{avail[i]: round(float(w_hedging[i]), 6) for i in range(n_etf)}})
        ranking_records.append({"date": ds,
            **{avail[ranked_idx[r]]: r + 1 for r in range(n_etf)}})
        daily_records.append({
            "date":       ds,
            "top_ticker": "CASH" if cash_flag else top_ticker,
            "top_score":  round(top_score, 6),
            "cash_flag":  cash_flag,
            "vix":        round(vix_today, 2),
            "v_today":    round(v_today, 6),
            "regime":     "STRESS" if high_stress else "NORMAL",
            "fv_over_f":  round(weights_dict["fv_over_f"], 6),
        })

        if n_scored % 252 == 0 or t == len(ret_arr) - 1:
            top5 = [(avail[ranked_idx[r]],
                     round(float(composite_z[ranked_idx[r]]), 3))
                    for r in range(min(5, n_etf))]
            print(
                f"  {ds} [VIX={vix_today:.1f} v={v_today:.4f}] | "
                + "  ".join(f"{tk}({sc:+.2f})" for tk, sc in top5)
                + (" [CASH]" if cash_flag else "")
            )

    # ── Latest snapshot ───────────────────────────────────────────────────────
    latest_score   = score_records[-1]
    latest_myopic  = myopic_records[-1]
    latest_hedging = hedging_records[-1]
    latest_ranking = ranking_records[-1]
    latest_date    = daily_records[-1]["date"]

    latest_out: dict = {}
    for i, tkr in enumerate(avail):
        latest_out[tkr] = {
            "composite_score": latest_score[tkr],
            "myopic_demand":   latest_myopic[tkr],
            "hedging_demand":  latest_hedging[tkr],
            "rank":            int(latest_ranking[tkr]),
            "heston": {
                "kappa": round(heston_cache[i].kappa, 4),
                "theta": round(heston_cache[i].theta, 6),
                "xi":    round(heston_cache[i].xi, 4),
                "rho":   round(heston_cache[i].rho, 4),
                "mu":    round(heston_cache[i].mu, 4),
            } if heston_cache[i] else {},
        }

    latest_ranked = sorted(
        latest_out.items(),
        key=lambda x: x[1]["composite_score"], reverse=True,
    )

    # Value function surface for dashboard
    v_grid, f_surface = solver.value_function_surface()

    print(
        f"\n  Latest ({latest_date}) top-{config.TOP_N}: "
        + "  ".join(
            f"{t}({v['composite_score']:+.3f})"
            for t, v in latest_ranked[:config.TOP_N]
        )
    )
    print(f"  Days scored (OOS): {n_scored}")

    return {
        "latest_date":   latest_date,
        "latest_scores": latest_out,
        "latest_ranked": latest_ranked,
        "daily_df":      pd.DataFrame(daily_records).set_index("date"),
        "score_df":      pd.DataFrame(score_records).set_index("date"),
        "myopic_df":     pd.DataFrame(myopic_records).set_index("date"),
        "hedging_df":    pd.DataFrame(hedging_records).set_index("date"),
        "ranking_df":    pd.DataFrame(ranking_records).set_index("date"),
        "v_grid":        v_grid.tolist(),
        "f_surface":     f_surface.tolist(),
        "universe":      universe_name,
        "n_etf":         n_etf,
    }
