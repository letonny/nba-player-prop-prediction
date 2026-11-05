"""Selection, exploratory backtesting and forward prediction tracking CLI."""

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_player_prop_prediction.selection.backtest import backtest, load_candidates
from nba_player_prop_prediction.selection.policy import Policy, rank
from nba_player_prop_prediction.selection.tracking import Ledger
from nba_player_prop_prediction.utils.io import write_json, write_parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bt = sub.add_parser("backtest")
    bt.add_argument("--output", type=Path, required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--contexts", type=Path, required=True)
    gen.add_argument("--cutoff", required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--date", required=True)
    gen.add_argument("--availability", type=Path)
    gen.add_argument("--markets", type=Path)
    gen.add_argument("--lines", type=Path)
    gen.add_argument("--ledger", type=Path, default=Path("artifacts/predictions.sqlite"))
    gen.add_argument("--dry-run", action="store_true")
    ranks = sub.add_parser("rank")
    ranks.add_argument("input", type=Path)
    ranks.add_argument("--output", type=Path, required=True)
    for p in [gen, ranks]:
        p.add_argument("--policy", type=Path)
    grade = sub.add_parser("grade")
    grade.add_argument("--ledger", type=Path, required=True)
    grade.add_argument("--outcomes", type=Path, required=True)
    export = sub.add_parser("report")
    export.add_argument("--ledger", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    odds = sub.add_parser("markets")
    odds.add_argument("--event-id", required=True)
    odds.add_argument(
        "--markets",
        nargs="+",
        default=["player_points", "player_rebounds", "player_assists", "player_threes"],
    )
    args = parser.parse_args()
    root = Path("artifacts/probabilistic/comparison-v1")
    if args.command == "backtest":
        c = load_candidates(
            root, Path("data/processed/history.parquet"), Path("data/processed/stats-v2.parquet")
        )
        results = backtest(c, args.output)
        print(json.dumps({k: v["selected"] for k, v in results.items()}, indent=2))
    elif args.command in ["generate", "rank"]:
        policy = Policy(**json.loads(args.policy.read_text())) if args.policy else Policy()
        if args.command == "generate":
            from nba_player_prop_prediction.selection.engine import generate

            contexts = json.loads(args.contexts.read_text())
            if any(c["game_date"] != args.date for c in contexts):
                raise ValueError("Context date mismatch")
            c = generate(
                pd.read_parquet("data/processed/history.parquet"),
                contexts,
                args.cutoff,
                root,
                Path("artifacts/probabilistic/oof-v2"),
                pd.read_json(args.availability) if args.availability else None,
                json.loads(args.markets.read_text()) if args.markets else None,
                json.loads(args.lines.read_text()) if args.lines else None,
            )
        else:
            c = pd.read_parquet(args.input)
        result = rank(c, policy)
        write_parquet(args.output, result)
        write_json(
            args.output.with_suffix(".json"),
            json.loads(result.to_json(orient="records", date_format="iso")),
        )
        lines = [
            "# Conditional statistical threshold selections",
            "",
            "Not confirmed bets or live-validated performance.",
            "",
            "| Player-game | Target | Event | Line | Probability | Market available |",
            "|---|---|---|---:|---:|---|",
        ]
        for row in result.loc[result.selected].itertuples():
            lines.append(
                f"| {row.player_game_id} | {row.target} | {row.event} | {row.line} | "
                f"{row.probability:.2%} | {row.market_available} |"
            )
        args.output.with_suffix(".md").write_text("\n".join(lines))
        if args.command == "generate" and not args.dry_run:
            ledger = Ledger(args.ledger)
            # Persist all candidates, including excluded forecasts; grading never edits payloads.
            for row in json.loads(result.to_json(orient="records", date_format="iso")):
                ledger.record(row)
        print(f"{int(result.selected.sum())} selections from {len(result)} candidates")
    elif args.command == "grade":
        ledger = Ledger(args.ledger)
        for row in json.loads(args.outcomes.read_text()):
            print(ledger.grade(**row))
    elif args.command == "markets":
        from nba_player_prop_prediction.selection.markets import OddsAPI

        print(
            json.dumps(
                OddsAPI(Path("data/raw/markets")).snapshot(args.event_id, args.markets), indent=2
            )
        )
    else:
        from nba_player_prop_prediction.selection.reports import forward_summary

        rows = Ledger(args.ledger).export()
        write_json(
            args.output,
            {
                "performance": forward_summary(rows),
                "mode": "forward_paper_tracking",
                "predictions": rows,
                "status_counts": pd.Series([r["grading_status"] for r in rows])
                .value_counts()
                .to_dict(),
            },
        )
        pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
        args.output.with_suffix(".md").write_text(
            "# Forward paper ledger\n\nNo monetary performance claim.\n\n"
            + str(len(rows))
            + " forecasts; see JSON/CSV for immutable forecasts and grades.\n"
        )


if __name__ == "__main__":
    main()
