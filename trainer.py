"""trainer.py — HJB engine orchestrator with HF push."""

from __future__ import annotations

import io
import json
import os

from huggingface_hub import HfApi

import config
import data_manager
from engine import run_engine


def push_results(result: dict, universe: str, token: str) -> None:
    slug = universe.lower().replace("_", "-")
    api  = HfApi(token=token)
    api.create_repo(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset",
                    exist_ok=True, private=False)

    output = {
        "run_date":      config.TODAY,
        "universe":      universe,
        "latest_date":   result["latest_date"],
        "latest_scores": result["latest_scores"],
        "latest_ranked": [{"ticker": t, **v} for t, v in result["latest_ranked"]],
        "v_grid":        result["v_grid"],
        "f_surface":     result["f_surface"],
        "config": {
            "gamma":               config.GAMMA,
            "horizon_days":        config.HORIZON_DAYS,
            "nv":                  config.NV,
            "nt":                  config.NT,
            "v_max":               config.V_MAX,
            "heston_window":       config.HESTON_WINDOW,
            "heston_refit_freq":   config.HESTON_REFIT_FREQ,
            "myopic_wt":           config.MYOPIC_WT,
            "hedging_wt":          config.HEDGING_WT,
            "vix_stress_threshold":config.VIX_STRESS_THRESHOLD,
            "cash_threshold":      config.CASH_THRESHOLD,
            "top_n":               config.TOP_N,
            "oos_start":           config.OOS_START,
        },
    }

    def _push(data: bytes, path: str, msg: str) -> None:
        api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=path,
                        repo_id=config.HF_OUTPUT_REPO, repo_type="dataset",
                        commit_message=msg)

    _push(json.dumps(output, indent=2, default=str).encode(),
          f"hjb_{config.TODAY}_{slug}.json",
          f"HJB results {config.TODAY} — {slug}")

    for name, df in [
        ("daily",    result["daily_df"]),
        ("scores",   result["score_df"]),
        ("myopic",   result["myopic_df"]),
        ("hedging",  result["hedging_df"]),
        ("rankings", result["ranking_df"]),
    ]:
        _push(df.to_csv().encode(), f"{name}_{slug}.csv",
              f"{name} {config.TODAY} — {slug}")

    print(f"  ✅ Pushed → {config.HF_OUTPUT_REPO}/hjb_{config.TODAY}_{slug}.json")


def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    target = os.environ.get("HJB_UNIVERSE", "ALL").upper()
    log_returns, macro_df = data_manager.load_data(token=token)

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue
        result = run_engine(log_returns=log_returns, macro_df=macro_df,
                            universe_tickers=tickers, universe_name=universe_name)
        push_results(result, universe_name, token)

    print("\n✅ HJB Stochastic Control engine complete.")


if __name__ == "__main__":
    main()
