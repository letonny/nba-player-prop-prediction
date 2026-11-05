"""Chronological minutes model competition, conditional on an observed appearance."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from nba_player_prop_prediction.features.pregame import (
    CATEGORICAL,
    VERSION,
    build_features,
    model_columns,
)
from nba_player_prop_prediction.ingestion.expansion import coverage
from nba_player_prop_prediction.utils.io import write_json, write_parquet

BASELINES = {
    "mean_5": "prior_5_minutes_mean",
    "mean_10": "prior_10_minutes_mean",
    "median": "historical_minutes_median",
}


@dataclass(frozen=True)
class TrainConfig:
    validation_start: str = "2024-07-01"
    test_start: str = "2025-07-01"
    min_train: int = 5000
    min_validation: int = 1000
    min_test: int = 1000
    min_train_dates: int = 60
    min_eval_dates: int = 14
    min_players: int = 50
    iterations: int = 300
    depth: int = 6
    learning_rate: float = 0.05
    seed: int = 42
    threads: int = 2
    interval_alpha: float = 0.1

    def __post_init__(self):
        if pd.Timestamp(self.validation_start) >= pd.Timestamp(self.test_start):
            raise ValueError("Validation must precede test")
        if (
            min(
                self.min_train,
                self.min_validation,
                self.min_test,
                self.min_train_dates,
                self.min_eval_dates,
                self.min_players,
                self.iterations,
                self.depth,
                self.threads,
            )
            < 1
        ):
            raise ValueError("Counts must be positive")
        if not 0 < self.interval_alpha < 1 or not 0 < self.learning_rate <= 1:
            raise ValueError("Invalid alpha or learning rate")


def matrix(frame, columns):
    result = frame[columns].copy()
    for col in columns:
        if col in CATEGORICAL:
            result[col] = result[col].astype("string").fillna("unknown").astype(str)
        else:
            result[col] = pd.to_numeric(result[col], errors="raise").astype(float)
    return result


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        "n": len(error),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "bias_prediction_minus_actual": float(error.mean()),
        "error_quantiles": {
            str(q): float(np.quantile(error, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]
        },
    }


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    report = {}
    for name, group in predictions.groupby("model"):
        entry = metrics(group.actual, group.predicted)
        entry["groups"] = {}
        categories = {
            "actual_minutes": pd.cut(group.actual, [-np.inf, 10, 20, 30, np.inf]).astype(str),
            "pregame_role": pd.cut(group.prior_minutes, [-np.inf, 10, 20, 30, np.inf]).astype(str),
            "history": np.where(group.prior_games < 5, "limited_under_5", "at_least_5"),
            "long_rest": np.where(group.rest_days >= 7, "at_least_7", "other_or_unknown"),
            "back_to_back": group.back_to_back.fillna(-1).astype(str),
        }
        for key, labels in categories.items():
            entry["groups"][key] = {
                str(label): metrics(part.actual, part.predicted)
                for label, part in group.groupby(labels, observed=True)
            }
        if "lower" in group and group.lower.notna().all():
            entry["interval_coverage"] = float(
                ((group.actual >= group.lower) & (group.actual <= group.upper)).mean()
            )
            entry["mean_interval_width"] = float((group.upper - group.lower).mean())
        report[name] = entry
    return report


class MinutesModel:
    def __init__(self, model, metadata):
        self.model, self.metadata = model, metadata

    def predict(self, features):
        return np.maximum(0, self.model.predict(matrix(features, self.metadata["columns"])))

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(folder / "model.cbm"))
        write_json(folder / "metadata.json", self.metadata)

    @classmethod
    def load(cls, folder: Path):
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["feature_version"] != VERSION:
            raise ValueError("Incompatible feature version")
        model = CatBoostRegressor()
        model.load_model(str(folder / "model.cbm"))
        return cls(model, metadata)


def train(history: pd.DataFrame, output: Path, config: TrainConfig | None = None) -> dict:
    config = config or TrainConfig()
    if (output / "report.json").exists() or (output / "metadata.json").exists():
        raise ValueError("Use a new run directory; existing evaluation artifacts are immutable")
    features = build_features(history)
    labels = history.set_index("player_game_id")
    features["target"] = features.player_game_id.map(labels.minutes)
    appearance = features.player_game_id.map(labels.participation).eq("played")
    data = features.loc[appearance & features.target.notna()].copy()
    validation, test = pd.Timestamp(config.validation_start), pd.Timestamp(config.test_start)
    splits = {
        "train": data.loc[data.game_date < validation],
        "validation": data.loc[(data.game_date >= validation) & (data.game_date < test)],
        "test": data.loc[data.game_date >= test],
    }
    periods = {
        name: {
            "n": len(part),
            "dates": int(part.game_date.nunique()),
            "players": int(part.player_id.nunique()),
            "start": str(part.game_date.min()),
            "end": str(part.game_date.max()),
        }
        for name, part in splits.items()
    }
    reasons = []
    for name, minimum in [
        ("train", config.min_train),
        ("validation", config.min_validation),
        ("test", config.min_test),
    ]:
        dates = config.min_train_dates if name == "train" else config.min_eval_dates
        if periods[name]["n"] < minimum or periods[name]["dates"] < dates:
            reasons.append(f"{name}: requires {minimum} rows and {dates} distinct game dates")
        if periods[name]["players"] < config.min_players:
            reasons.append(f"{name}: requires {config.min_players} players")
    report = {
        "coverage": coverage(history),
        "periods": periods,
        "config": asdict(config),
        "feature_version": VERSION,
        "limitations": [
            "Conditional on appearing; no inactive/DNP risk model.",
            "Prior dates approximate availability; no publication timestamps.",
            "Walk-forward features may use earlier held-out game results.",
            "Validation residual intervals have no time-series coverage guarantee.",
        ],
    }
    if reasons:
        report.update(status="skipped", reasons=reasons)
        write_json(output / "report.json", report)
        (output / "report.md").write_text("# Training skipped\n\n" + "\n".join(reasons))
        return report
    training, val = splits["train"], splits["validation"]
    columns = model_columns(features.drop(columns="target"))
    model = CatBoostRegressor(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        random_seed=config.seed,
        thread_count=config.threads,
        loss_function="RMSE",
        eval_metric="MAE",
        has_time=True,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        matrix(training, columns),
        training.target,
        cat_features=[c for c in CATEGORICAL if c in columns],
        eval_set=(matrix(val, columns), val.target),
        early_stopping_rounds=40,
    )
    fallback = float(training.target.median())
    residual = np.abs(val.target - np.maximum(0, model.predict(matrix(val, columns))))
    level = min(1.0, np.ceil((len(val) + 1) * (1 - config.interval_alpha)) / len(val))
    radius = float(np.quantile(residual, level, method="higher"))
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(data, index=False).values.tobytes()
    ).hexdigest()
    metadata = {
        "feature_version": VERSION,
        "columns": columns,
        "fallback": fallback,
        "interval_radius": radius,
        "interval_alpha": config.interval_alpha,
        "trained_through": str(val.game_date.max().date()),
        "version": fingerprint[:12],
        "data_sha256": fingerprint,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
    }
    bundle = MinutesModel(model, metadata)
    bundle.save(output)
    predictions = []
    for split_name in ["validation", "test"]:
        part = splits[split_name]
        all_predictions = {
            name: part[col].fillna(fallback).to_numpy() for name, col in BASELINES.items()
        }
        all_predictions["catboost"] = bundle.predict(part)
        for name, predicted in all_predictions.items():
            rows = part[
                [
                    "player_game_id",
                    "game_date",
                    "player_id",
                    "prior_games",
                    "rest_days",
                    "back_to_back",
                ]
            ].copy()
            rows["prior_minutes"] = part.prior_10_minutes_mean
            rows["actual"], rows["predicted"] = part.target, predicted
            rows["split"], rows["model"] = split_name, name
            if name == "catboost":
                rows["lower"], rows["upper"] = np.maximum(0, predicted - radius), predicted + radius
            predictions.append(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    write_parquet(output / "predictions.parquet", predictions)
    report.update(
        status="trained",
        model_version=metadata["version"],
        metrics={
            split: evaluate_predictions(predictions.loc[predictions.split.eq(split)])
            for split in ["validation", "test"]
        },
        feature_importance=dict(
            sorted(
                zip(columns, model.feature_importances_.tolist(), strict=True), key=lambda x: -x[1]
            )
        ),
    )
    write_json(output / "report.json", report)
    text = [
        "# Minutes model comparison",
        "",
        "Conditional on appearing. No win-rate claim.",
        "",
        "| Split | Model | N | MAE | RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for split, models in report["metrics"].items():
        for name, m in models.items():
            text.append(f"| {split} | {name} | {m['n']} | {m['mae']:.3f} | {m['rmse']:.3f} |")
    text.extend(
        ["", "Full periods, subgroup errors and importance: report.json.", *report["limitations"]]
    )
    (output / "report.md").write_text("\n".join(text))
    plot_predictions(predictions.loc[predictions.split.eq("test")], output)
    return report


def plot_predictions(predictions, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, group in predictions.groupby("model"):
        axes[0].hist(group.predicted - group.actual, bins=40, alpha=0.35, label=name)
    axes[0].set(xlabel="Predicted minus actual minutes", ylabel="Observations")
    axes[0].legend()
    cat = predictions.loc[predictions.model.eq("catboost")]
    axes[1].scatter(cat.actual, cat.predicted, s=2, alpha=0.2)
    axes[1].plot([0, 60], [0, 60], color="black")
    axes[1].set(xlabel="Actual minutes", ylabel="CatBoost predicted minutes")
    fig.tight_layout()
    fig.savefig(output / "errors.png")
    plt.close(fig)
