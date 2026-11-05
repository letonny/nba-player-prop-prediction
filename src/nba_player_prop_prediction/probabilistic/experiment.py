"""Local immutable experiments with dedicated chronological calibration."""

import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nba_player_prop_prediction.features.pregame import VERSION, model_columns
from nba_player_prop_prediction.probabilistic.calibration import Calibrator
from nba_player_prop_prediction.probabilistic.data import enrich, partitions
from nba_player_prop_prediction.probabilistic.distributions import TARGETS, CountDistribution
from nba_player_prop_prediction.probabilistic.evaluation import (
    candidate_frame,
    confidence_report,
    distribution_metrics,
    probability_metrics,
)
from nba_player_prop_prediction.probabilistic.models import Candidate
from nba_player_prop_prediction.utils.io import write_json, write_parquet

NAMES = [
    "baseline_5",
    "baseline_10",
    "baseline_season",
    "baseline_rate",
    "baseline_global",
    "xgblss_Poisson",
    "xgblss_NegativeBinomial",
    "xgblss_ZIPoisson",
    "xgblss_ZINB",
    "catboost",
    "catboost_direct",
    "catboost_quantile",
    "ngboost",
]


def prepare(history_path, features_path, oof_path, output):
    h, f, o = [pd.read_parquet(p) for p in [history_path, features_path, oof_path]]
    data = enrich(h, f, o)
    write_parquet(output, data)
    write_json(
        output.with_suffix(".json"),
        {
            "feature_version": VERSION,
            "sources": [
                {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in [history_path, features_path, oof_path]
            ],
            "base_columns": model_columns(f),
            "rows": len(data),
        },
    )
    return data


def sample(frame, limit):
    if limit and len(frame) > limit:
        return frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=int)]
    return frame


def run(data_path: Path, output: Path, targets=None, names=None, rounds=60, train_limit=6000):
    targets, names = targets or list(TARGETS), names or NAMES
    if not set(targets) <= set(TARGETS) or not set(names) <= set(NAMES):
        raise ValueError("Unknown target or candidate")
    source = json.loads(data_path.with_suffix(".json").read_text())
    data = pd.read_parquet(data_path)
    config = {
        "targets": targets,
        "names": names,
        "rounds": rounds,
        "train_limit": train_limit,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": 42,
        "feature_version": VERSION,
        "calibrator_default": "sigmoid",
        "boundaries": ["2024-07-01", "2025-01-01", "2025-07-01"],
        "software": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "catboost",
                "xgboost",
                "xgboostlss",
                "ngboost",
                "torch",
            ]
        },
    }
    manifest = output / "experiment.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Existing experiment has different configuration; use a new directory")
    else:
        write_json(manifest, config)
    write_json(output / "provenance.json", source)
    summary, failures = {}, {}
    for target in targets:
        valid = data.appearance & data[f"label_{target}"].notna()
        parts = partitions(data.loc[valid])
        if any(len(p) < 500 or p.game_date.nunique() < 14 for p in parts.values()):
            raise ValueError(
                "Insufficient chronological coverage (500 rows / 14 dates per partition)"
            )
        train, validation = (
            sample(parts["train"], train_limit),
            sample(parts["validation"], train_limit),
        )
        calibration, test = parts["calibration"], parts["test"]
        periods = {
            k: {
                "rows": len(p),
                "players": int(p.player_id.nunique()),
                "games": int(p.game_id.nunique()),
                "start": str(p.game_date.min()),
                "end": str(p.game_date.max()),
            }
            for k, p in dict(
                train=train, validation=validation, calibration=calibration, test=test
            ).items()
        }
        summary[target] = {}
        for name in names:
            folder = output / target / name
            if (folder / "report.json").exists():
                summary[target][name] = json.loads((folder / "report.json").read_text())
                continue
            print(f"Fitting {target}: {name}", flush=True)
            try:
                columns = source["base_columns"] + [
                    f"season_mean_{target}",
                    f"rate_{target}",
                    f"prior_std_{target}",
                ]
                if name != "catboost_direct":
                    columns += ["predicted_minutes", "minutes_uncertainty"]
                model = Candidate(name, target, columns, rounds).fit(train, validation)
                vd, cd, td = [model.predict(p) for p in [validation, calibration, test]]
                report = {
                    "target": target,
                    "name": name,
                    "periods": periods,
                    "columns": columns,
                    "conditional_on_appearance": True,
                    "train_mean": float(train[f"label_{target}"].mean()),
                    "train_variance": float(train[f"label_{target}"].var()),
                    "train_zero_fraction": float(train[f"label_{target}"].eq(0).mean()),
                }
                calibrators = {}
                if name == "catboost_quantile":
                    y = test[f"label_{target}"].to_numpy()
                    report["test"] = {
                        "median_mae": float(np.abs(td[:, 1] - y).mean()),
                        "interval_90_coverage": float(((y >= td[:, 0]) & (y <= td[:, 2])).mean()),
                    }
                    saved = test[
                        ["player_game_id", "game_id", "game_date", f"label_{target}"]
                    ].copy()
                    saved[["q05", "q50", "q95"]] = td
                else:
                    report["validation"] = distribution_metrics(vd, validation[f"label_{target}"])
                    report["test"] = distribution_metrics(td, test[f"label_{target}"])
                    lines = TARGETS[target]
                    cp = np.column_stack([cd.probability(line) for line in lines])
                    tp = np.column_stack([td.probability(line) for line in lines])
                    cy = calibration[f"label_{target}"].to_numpy()[:, None] >= np.array(lines)
                    ty = test[f"label_{target}"].to_numpy()[:, None] >= np.array(lines)
                    report["probabilities"] = {}
                    for method in ["identity", "sigmoid", "isotonic"]:
                        cal = Calibrator(method).fit(cp, cy, calibration.game_date, "2025-07-01")
                        calibrators[method] = cal
                        probabilities = cal.predict(tp)
                        if (np.diff(probabilities, axis=1) > 1e-10).any():
                            raise ValueError("Nonmonotone calibrated thresholds")
                        metrics = probability_metrics(probabilities, ty)
                        candidates = candidate_frame(
                            test, lines, probabilities, test[f"label_{target}"]
                        )
                        metrics["confidence"] = confidence_report(candidates)
                        report["probabilities"][method] = metrics
                        write_parquet(folder / f"thresholds_{method}.parquet", candidates)
                    saved = test[
                        ["player_game_id", "game_id", "game_date", f"label_{target}"]
                    ].copy()
                    saved["mean"], saved["dispersion"], saved["gate"] = (
                        td.mean,
                        td.dispersion,
                        td.gate,
                    )
                write_parquet(folder / "predictions.parquet", saved)
                bundle = {
                    "model": model,
                    "calibrators": calibrators,
                    "feature_version": VERSION,
                    "target": target,
                    "trained_through": str(calibration.game_date.max()),
                    "config": config,
                }
                folder.mkdir(parents=True, exist_ok=True)
                payload = pickle.dumps(bundle)
                (folder / "bundle.pkl").write_bytes(payload)
                report["version"] = hashlib.sha256(payload).hexdigest()
                restored = pickle.loads(payload)["model"].predict(test.iloc[:3])
                np.testing.assert_allclose(
                    restored if isinstance(restored, np.ndarray) else restored.expected,
                    td[:3] if isinstance(td, np.ndarray) else td.expected[:3],
                    rtol=1e-5,
                    atol=1e-6,
                )
                write_json(folder / "report.json", report)
                summary[target][name] = report
            except Exception as exc:
                import traceback

                failures[f"{target}/{name}"] = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(output / "failures.json", failures)
                print(f"FAILED {target}/{name}: {exc}", flush=True)
        # Distribution/model selection uses validation NLL only, never test metrics.
        eligible = {
            n: r["validation"]["nll"] for n, r in summary[target].items() if "validation" in r
        }
        write_json(
            output / target / "validation_selection.json",
            {"nll": eligible, "selected": min(eligible, key=eligible.get) if eligible else None},
        )
    write_json(output / "summary.json", summary)
    write_json(output / "failures.json", failures)
    return summary, failures


def load_bundle(folder):
    """Load only trusted local pickle artifacts; hashes detect accidental corruption."""
    report = json.loads((folder / "report.json").read_text())
    payload = (folder / "bundle.pkl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != report["version"]:
        raise ValueError("Model checksum mismatch")
    bundle = pickle.loads(payload)
    if bundle["feature_version"] != VERSION:
        raise ValueError("Feature schema version mismatch")
    return bundle, report


def verify(folder):
    bundle, report = load_bundle(folder)
    saved = pd.read_parquet(folder / "predictions.parquet")
    if "mean" not in saved:
        return {"quantile_artifact_rows": len(saved)}
    dist = CountDistribution(
        saved["mean"].to_numpy(), saved.dispersion.to_numpy(), saved.gate.to_numpy()
    )
    computed = distribution_metrics(dist, saved[f"label_{bundle['target']}"])
    for key, val in computed.items():
        if not np.isclose(val, report["test"][key]):
            raise ValueError(f"Reproduction mismatch: {key}")
    for method in bundle["calibrators"]:
        table = pd.read_parquet(folder / f"thresholds_{method}.parquet")
        result = probability_metrics(table.probability, table.hit)
        for key in ["brier", "log_loss", "ece"]:
            if not np.isclose(result[key], report["probabilities"][method][key]):
                raise ValueError(f"Probability mismatch: {method}/{key}")
    return computed
