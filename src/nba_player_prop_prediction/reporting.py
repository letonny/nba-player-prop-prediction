"""Reproducible JSON and Markdown quality reports, optionally with histograms."""

from pathlib import Path

import pandas as pd

from nba_player_prop_prediction.utils.io import atomic_write, write_json


def quality_report(
    frame: pd.DataFrame, output: Path, audit: dict | None = None, plots: bool = False
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    distributions = {}
    for col in ["minutes", "points", "rebounds", "assists"]:
        distributions[col] = {
            k: float(v) if pd.notna(v) else None for k, v in frame[col].describe().items()
        }
    report = {
        "observations": len(frame),
        "players": int(frame.player_id.nunique()),
        "teams": int(frame.team_id.nunique()),
        "seasons": sorted(frame.season.unique().tolist()),
        "date_start": str(frame.game_date.min()) if len(frame) else None,
        "date_end": str(frame.game_date.max()) if len(frame) else None,
        "missing_percent": frame.isna().mean().mul(100).fillna(0).to_dict(),
        "duplicate_rows": int(frame.duplicated(["player_id", "game_id"]).sum()),
        "participation": frame.participation.value_counts().to_dict(),
        "distributions": distributions,
        "games_per_player": frame.groupby("player_id").size().to_dict(),
        "preprocessing_audit": audit,
    }
    write_json(output / "quality.json", report)
    lines = ["# NBA data quality report", "", "Reports describe data, not model performance.", ""]
    for key, value in report.items():
        lines.extend(
            [
                f"## {key.replace('_', ' ').title()}",
                "",
                "```json",
                __import__("json").dumps(value, indent=2, allow_nan=False),
                "```",
                "",
            ]
        )
    atomic_write(output / "quality.md", lambda p: p.write_text("\n".join(lines)))
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, col in zip(axes.flat, distributions, strict=True):
            ax.hist(frame[col].dropna().astype(float), bins=30)
            ax.set(title=col, ylabel="Player-games")
        fig.tight_layout()
        fig.savefig(output / "distributions.png")
        plt.close(fig)
    return report
