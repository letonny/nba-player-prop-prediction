"""Exploratory reuse of frozen M3 test forecasts; never a fresh holdout."""

import hashlib
import json

import numpy as np
import pandas as pd

from nba_player_prop_prediction.selection.policy import Policy, rank
from nba_player_prop_prediction.utils.io import write_json, write_parquet


def load_candidates(root, history_path, features_path):
    history = pd.read_parquet(history_path).set_index("player_game_id")
    features = pd.read_parquet(features_path).set_index("player_game_id")
    frames = []
    for target in ["points", "rebounds", "assists", "threes_made"]:
        name = json.loads((root / target / "validation_selection.json").read_text())["selected"]
        folder = root / target / name
        report = json.loads((folder / "report.json").read_text())
        f = pd.read_parquet(folder / "thresholds_sigmoid.parquet")
        raw = pd.read_parquet(folder / "thresholds_identity.parquet")
        f = f.merge(
            raw[["player_game_id", "line", "probability"]].rename(
                columns={"probability": "raw_probability"}
            ),
            on=["player_game_id", "line"],
            validate="one_to_one",
        )
        actual = f.player_game_id.map(history[target])
        if not (f.hit == (actual >= f.line)).all():
            raise ValueError("Saved outcomes disagree with cleaned game logs")
        f["target"] = target
        f["model"] = name
        f["model_version"] = report["version"]
        f["event"] = "at_least"
        f["calibration_n"] = report["periods"]["calibration"]["rows"]
        for col in ["player_id", "player_name", "team_id", "opponent", "opponent_id"]:
            f[col] = f.player_game_id.map(history[col])
        f["prior_games"] = f.player_game_id.map(features.prior_games)
        f["back_to_back"] = f.player_game_id.map(features.back_to_back)
        f["minutes_change"] = f.player_game_id.map(features.minutes_change)
        f["market_available"] = False
        f["availability"] = "unknown"
        f["minutes_restricted"] = False
        f["conditional_on_appearance"] = True
        f["mode"] = "exploratory_backtest"
        f["warnings"] = "No historical market, injury, or lineup snapshot; availability unknown"
        f["candidate_id"] = [
            hashlib.sha256(f"{p}|{target}|{line}|at_least".encode()).hexdigest()
            for p, line in zip(f.player_game_id, f.line, strict=True)
        ]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def metrics(f):
    n = len(f)
    if not n:
        return {
            "n": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": None,
            "average_probability": None,
            "gap": None,
            "game_cluster_95_ci": None,
            "unique_player_games": 0,
            "unique_games": 0,
        }
    groups = f.groupby("game_id").hit.agg(["count", "sum"]).to_numpy()
    rng = np.random.default_rng(42)
    rates = []
    for _ in range(1000):
        count, wins = groups[rng.integers(len(groups), size=len(groups))].sum(axis=0)
        rates.append(wins / count)
    return {
        "n": n,
        "hits": int(f.hit.sum()),
        "misses": int(n - f.hit.sum()),
        "hit_rate": float(f.hit.mean()),
        "average_probability": float(f.probability.mean()),
        "gap": float(f.hit.mean() - f.probability.mean()),
        "game_cluster_95_ci": np.quantile(rates, [0.025, 0.975]).tolist(),
        "unique_player_games": int(f.player_game_id.nunique()),
        "unique_games": int(f.game_id.nunique()),
    }


def report(frame, folder, detail=True):
    selected = frame.loc[frame.selected]
    result = {
        "mode": "exploratory_backtest",
        "candidates": len(frame),
        "selected": metrics(selected),
        "selection_coverage": len(selected) / len(frame) if len(frame) else 0,
        "exclusions": frame.loc[~frame.selected, "reason"].value_counts().to_dict(),
    }
    from nba_player_prop_prediction.probabilistic.evaluation import probability_metrics

    result["calibration"] = (
        probability_metrics(selected.probability, selected.hit) if len(selected) else None
    )
    result["buckets"] = {}
    for lo, hi in [(0.6, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0001)]:
        result["buckets"][str(lo)] = metrics(
            selected.loc[selected.probability.ge(lo) & selected.probability.lt(hi)]
        )
    if detail:
        selected = selected.copy()
        dates = pd.to_datetime(selected.game_date)
        selected["week"] = dates.dt.to_period("W").astype(str)
        selected["month"] = dates.dt.to_period("M").astype(str)
        for col in [
            "target",
            "model",
            "player_id",
            "opponent",
            "game_id",
            "game_date",
            "week",
            "month",
        ]:
            result[col] = {
                str(k): {
                    "n": len(g),
                    "hits": int(g.hit.sum()),
                    "hit_rate": float(g.hit.mean()),
                    "average_probability": float(g.probability.mean()),
                }
                for k, g in selected.groupby(col)
            }
    write_json(folder / "report.json", result)
    write_parquet(folder / "candidates.parquet", frame)
    selected.to_csv(folder / "selections.csv", index=False)
    (folder / "report.md").write_text(
        "# Exploratory historical selections\n\n"
        "Not live performance or confirmed sportsbook bets.\n\n```json\n"
        + json.dumps(
            {
                k: result[k]
                for k in ["candidates", "selected", "selection_coverage", "exclusions", "buckets"]
            },
            indent=2,
        )
        + "\n```\n"
    )
    return result


def backtest(candidates, output):
    if output.exists():
        raise ValueError("Use a new report directory")
    policies = {
        "default85": Policy(),
        "unrestricted90": Policy(0.9, 1000000, 1000000, 1000000),
        "player_game90": Policy(0.9, 1000000, 1000000, 1),
        "one_game90": Policy(0.9, 1000000, 1, 1),
        "market_only90": Policy(0.9, 1000000, 1000000, 1, require_market=True),
        "conservative90": Policy(0.9, 1000000, 1, 1, require_available=True, min_history=20),
    }
    results = {}
    for name, policy in policies.items():
        # Outcomes are carried only to evaluation; rank never reads them.
        result = rank(candidates, policy)
        results[name] = report(result, output / name, detail=name == "default85")
    threes = candidates.loc[candidates.target.eq("threes_made")]
    for name in [
        "unrestricted90",
        "player_game90",
        "one_game90",
        "market_only90",
        "conservative90",
    ]:
        results["threes_" + name] = report(
            rank(threes, policies[name]), output / ("threes_" + name), False
        )
    write_json(
        output / "comparison.json",
        {
            k: {"selected": v["selected"], "coverage": v["selection_coverage"]}
            for k, v in results.items()
        },
    )
    return results
