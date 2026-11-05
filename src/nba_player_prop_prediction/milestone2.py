"""CLI: python -m nba_player_prop_prediction.milestone2 --help."""

import argparse
import json
from pathlib import Path

from nba_player_prop_prediction.config import Settings
from nba_player_prop_prediction.features.pregame import VERSION, build_features
from nba_player_prop_prediction.ingestion.expansion import assemble, expand
from nba_player_prop_prediction.utils.io import (
    configure_logging,
    read_table,
    write_json,
    write_parquet,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    collect.add_argument("--max-requests", type=int, default=4)
    collect.add_argument("--budget-seconds", type=float, default=180)
    collect.add_argument("--refresh", action="store_true")
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--output", type=Path, default=Path("data/processed/history.parquet"))
    feat = commands.add_parser("features")
    feat.add_argument("history", type=Path)
    feat.add_argument("--output", type=Path, default=Path("data/processed/features.parquet"))
    training = commands.add_parser("train")
    training.add_argument("history", type=Path)
    training.add_argument("--config", type=Path, default=Path("configs/minutes.json"))
    training.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("run", type=Path)
    prediction = commands.add_parser("predict")
    prediction.add_argument("history", type=Path)
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--context", type=Path, required=True)
    prediction.add_argument("--player-id", required=True)
    prediction.add_argument("--cutoff", required=True)
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    if args.command == "collect":
        report = expand(settings, args.years, args.max_requests, args.budget_seconds, args.refresh)
        print(json.dumps(report, indent=2))
        if report["failed"] or report["pending"]:
            raise SystemExit(1)
    elif args.command == "assemble":
        report = assemble(settings, args.output)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ["missing_percent", "cleaning", "sources"]
                },
                indent=2,
            )
        )
        if report["cleaning"]["rejected_rows"]:
            raise SystemExit(2)
    elif args.command == "features":
        features = build_features(read_table(args.history))
        write_parquet(args.output, features)
        write_json(
            args.output.with_suffix(".json"),
            {
                "version": VERSION,
                "rows": len(features),
                "columns": list(features),
                "missing_percent": features.isna().mean().mul(100).to_dict(),
            },
        )
        print(f"{len(features)} rows, {len(features.columns)} columns: {args.output}")
    elif args.command == "train":
        from nba_player_prop_prediction.models.minutes.training import TrainConfig, train

        report = train(
            read_table(args.history),
            args.output,
            TrainConfig(**json.loads(args.config.read_text())),
        )
        print(json.dumps({k: report[k] for k in ["status", "periods"]}, indent=2))
        if report["status"] == "skipped":
            print("\n".join(report["reasons"]))
            raise SystemExit(2)
    elif args.command == "evaluate":
        from nba_player_prop_prediction.models.minutes.training import evaluate_predictions

        predictions = read_table(args.run / "predictions.parquet")
        print(
            json.dumps(
                evaluate_predictions(predictions.loc[predictions.split.eq("test")]), indent=2
            )
        )
    else:
        from nba_player_prop_prediction.models.minutes.predict import predict_player_minutes
        from nba_player_prop_prediction.models.minutes.training import MinutesModel

        print(
            json.dumps(
                predict_player_minutes(
                    MinutesModel.load(args.model),
                    read_table(args.history),
                    args.player_id,
                    json.loads(args.context.read_text()),
                    args.cutoff,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
