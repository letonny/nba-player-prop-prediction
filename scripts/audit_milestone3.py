"""Reproduce final audit from saved predictions; no fitting or model selection on test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_player_prop_prediction.probabilistic.evaluation import confidence_report
from nba_player_prop_prediction.probabilistic.experiment import verify
from nba_player_prop_prediction.utils.io import write_json

ROOT = Path("artifacts/probabilistic/comparison-v1")


def bucket_interval(frame, mask):
    temp = frame.assign(selected=mask.astype(int), wins=(mask & frame.hit).astype(int))
    groups = temp.groupby("game_id")[["selected", "wins"]].sum().to_numpy()
    rng = np.random.default_rng(42)
    rates = []
    for _ in range(2000):
        n, w = groups[rng.integers(len(groups), size=len(groups))].sum(axis=0)
        if n:
            rates.append(w / n)
    return np.quantile(rates, [0.025, 0.975]).tolist() if rates else None


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    audit = {"comparison": {}, "selected": {}, "buckets": {}, "hit_rate_audit": {}}
    for target, models in summary.items():
        selected = json.loads((ROOT / target / "validation_selection.json").read_text())["selected"]
        audit["selected"][target] = selected
        baseline = min(
            (n for n in models if n.startswith("baseline")),
            key=lambda n: models[n]["validation"]["nll"],
        )
        xgb = min(
            (n for n in models if n.startswith("xgblss")),
            key=lambda n: models[n]["validation"]["nll"],
        )
        audit["comparison"][target] = {}
        for name in [baseline, xgb, "catboost", "catboost_direct", "ngboost"]:
            r = models[name]
            audit["comparison"][target][name] = {
                **r["test"],
                **{k: r["probabilities"]["sigmoid"][k] for k in ["brier", "log_loss", "ece"]},
            }
        f = pd.read_parquet(ROOT / target / selected / "thresholds_sigmoid.parquet")
        assert not f.duplicated(["player_game_id", "line"]).any()
        audit["buckets"][target] = []
        for low, high in [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.00001)]:
            mask = f.probability.ge(low) & f.probability.lt(high)
            rows = f.loc[mask]
            audit["buckets"][target].append(
                {
                    "bucket": f"{low:.0%}–{min(high, 1):.0%}",
                    "n": len(rows),
                    "expected": float(rows.probability.mean()) if len(rows) else None,
                    "observed": float(rows.hit.mean()) if len(rows) else None,
                    "game_cluster_95_ci": bucket_interval(f, mask),
                }
            )
        if target == "threes_made":
            selected_rows = f.loc[f.probability.ge(0.9)]
            one = f.sort_values(["probability", "line"], ascending=[False, True]).drop_duplicates(
                "player_game_id"
            )
            audit["hit_rate_audit"] = {
                "all": confidence_report(f, replicates=2000)["0.9"],
                "one_per_player_game": confidence_report(one, replicates=2000)["0.9"],
                "duplicate_candidates": int(f.duplicated(["player_game_id", "line"]).sum()),
                "extra_selected_lines": len(selected_rows) - selected_rows.player_game_id.nunique(),
                "lines": sorted(f.line.unique().tolist()),
            }
        # All candidates must be scored against identical event identities and outcomes.
        expected = f.sort_values(["player_game_id", "line"])[
            ["player_game_id", "line", "hit"]
        ].reset_index(drop=True)
        for name in models:
            if name == "catboost_quantile":
                continue
            other = pd.read_parquet(ROOT / target / name / "thresholds_sigmoid.parquet")
            pd.testing.assert_frame_equal(
                expected,
                other.sort_values(["player_game_id", "line"])[
                    ["player_game_id", "line", "hit"]
                ].reset_index(drop=True),
            )
    for report in ROOT.glob("*/*/report.json"):
        verify(report.parent)
    audit["periods"] = summary["points"]["catboost"]["periods"]
    oof = pd.read_parquet("artifacts/probabilistic/oof-v2/minutes.parquet")
    assert (oof.minutes_trained_through < oof.game_date).all()
    assert (oof.prediction_asof == oof.game_date).all()
    for models in summary.values():
        for r in models.values():
            assert "minutes" not in r["columns"]
            assert not any(c.startswith("label_") for c in r["columns"])
    audit["oof"] = {
        "rows": len(oof),
        "folds": json.loads(Path("artifacts/probabilistic/oof-v2/manifest.json").read_text())[
            "folds"
        ],
    }
    write_json(ROOT / "final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
