"""Fixed threshold evaluation with game-cluster bootstrap uncertainty."""

import numpy as np
import pandas as pd


def probability_metrics(p, y):
    p, y = np.asarray(p).ravel(), np.asarray(y).ravel()
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    bins = np.minimum((p * 10).astype(int), 9)
    reliability = []
    for b in range(10):
        mask = bins == b
        if mask.any():
            reliability.append(
                {
                    "bucket": b,
                    "n": int(mask.sum()),
                    "predicted": float(p[mask].mean()),
                    "observed": float(y[mask].mean()),
                }
            )
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "ece": sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in reliability) / len(p),
        "reliability": reliability,
    }


def confidence_report(frame, replicates=300):
    """One candidate per player-game/stat/line; overlapping lines remain explicit.

    These are selection counts, not independent trials. Resample complete NBA
    games (including games with no selected candidates) for interval estimates.
    """
    if frame.duplicated(["player_game_id", "line"]).any():
        raise ValueError("Duplicate threshold candidates")
    reports = {}
    for cutoff in [0.6, 0.7, 0.8, 0.85, 0.9]:
        selected = frame.probability >= cutoff
        temp = frame.assign(selected=selected.astype(int), win=(selected & frame.hit).astype(int))
        clusters = temp.groupby("game_id")[["selected", "win"]].sum().to_numpy()
        rng = np.random.default_rng(42)
        estimates = []
        for _ in range(replicates):
            sample = clusters[rng.integers(0, len(clusters), len(clusters))].sum(axis=0)
            if sample[0]:
                estimates.append(sample[1] / sample[0])
        n, wins = int(selected.sum()), int(temp.win.sum())
        reports[str(cutoff)] = {
            "eligible_candidates": len(frame),
            "selected": n,
            "wins": wins,
            "losses": n - wins,
            "hit_rate": wins / n if n else None,
            "average_probability": float(frame.loc[selected, "probability"].mean()) if n else None,
            "coverage": n / len(frame),
            "unique_player_games": int(frame.loc[selected, "player_game_id"].nunique()),
            "unique_games": int(frame.loc[selected, "game_id"].nunique()),
            "game_cluster_95_interval": np.quantile(estimates, [0.025, 0.975]).tolist()
            if estimates
            else None,
        }
    return reports


def distribution_metrics(dist, actual):
    y = np.asarray(actual)
    error = dist.expected - y
    lo, hi = dist.quantile(0.05), dist.quantile(0.95)
    return {
        "n": len(y),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "nll": float(-dist.logpmf(y).mean()),
        "interval_90_coverage": float(((y >= lo) & (y <= hi)).mean()),
        "interval_mean_width": float((hi - lo).mean()),
    }


def candidate_frame(context, lines, probs, actual):
    rows = []
    for index, line in enumerate(lines):
        part = context[["player_game_id", "game_id", "game_date"]].copy()
        part["line"] = line
        part["probability"] = probs[:, index]
        part["hit"] = np.asarray(actual) >= line
        rows.append(part)
    return pd.concat(rows, ignore_index=True)
