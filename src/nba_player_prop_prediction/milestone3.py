"""Milestone 3 reproducible local commands."""

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_player_prop_prediction.probabilistic.experiment import NAMES, prepare, run, verify
from nba_player_prop_prediction.probabilistic.minutes import walk_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    oof = sub.add_parser("oof")
    oof.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--oof", type=Path, required=True)
    prep.add_argument("--output", type=Path, default=Path("data/processed/stats.parquet"))
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("data/processed/stats.parquet"))
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--targets", nargs="+")
    training.add_argument("--models", nargs="+", choices=NAMES)
    training.add_argument("--rounds", type=int, default=60)
    training.add_argument("--train-limit", type=int, default=6000)
    report = sub.add_parser("report")
    report.add_argument("folder", type=Path)
    report.add_argument("--data", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("folder", type=Path)
    infer = sub.add_parser("predict")
    infer.add_argument("folder", type=Path)
    infer.add_argument("--data", type=Path, default=Path("data/processed/stats.parquet"))
    infer.add_argument("--player-game-id", required=True)
    infer.add_argument("--line", type=float, required=True)
    infer.add_argument("--event", choices=["at_least", "over", "under"], default="at_least")
    args = parser.parse_args()
    if args.command == "oof":
        walk_forward(
            pd.read_parquet("data/processed/history.parquet"),
            pd.read_parquet("data/processed/features.parquet"),
            args.output,
        )
    elif args.command == "prepare":
        prepare(
            Path("data/processed/history.parquet"),
            Path("data/processed/features.parquet"),
            args.oof,
            args.output,
        )
    elif args.command == "train":
        _, failures = run(
            args.data, args.output, args.targets, args.models, args.rounds, args.train_limit
        )
        if failures:
            raise SystemExit(1)
    elif args.command == "report":
        from nba_player_prop_prediction.probabilistic.reporting import base_rate_reference, render

        render(args.folder)
        base_rate_reference(args.data, args.folder)
    elif args.command == "verify":
        print(json.dumps(verify(args.folder), indent=2))
    else:
        from nba_player_prop_prediction.probabilistic.prediction import predict

        frame = pd.read_parquet(args.data)
        frame = frame.loc[frame.player_game_id.eq(args.player_game_id)]
        if len(frame) != 1:
            raise ValueError("Expected one historical prediction context")
        print(json.dumps(predict(args.folder, frame, args.line, args.event), indent=2))


if __name__ == "__main__":
    main()
