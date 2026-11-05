"""Forward paper performance, separated by confirmed market status."""

import pandas as pd

from nba_player_prop_prediction.selection.backtest import metrics


def forward_summary(rows):
    if not rows:
        return {"mode": "forward_paper_tracking", "n": 0, "groups": {}}
    frame = pd.DataFrame(rows)
    frame["market_available"] = frame.get("market_available", False)
    if "selected" not in frame:
        frame["selected"] = False
    results = {}
    for backed, group in frame.groupby("market_available"):
        valid = group.loc[group.grading_status.isin(["hit", "miss"]) & group.selected].copy()
        valid["hit"] = valid.grading_status.eq("hit")
        valid["player_game_id"] = valid.player_id.astype(str) + ":" + valid.game_id.astype(str)
        if len(valid):
            valid["day"] = pd.to_datetime(valid.scheduled_tipoff, utc=True).dt.strftime("%Y-%m-%d")
            valid["week"] = pd.to_datetime(valid.day).dt.to_period("W").astype(str)
            valid["month"] = pd.to_datetime(valid.day).dt.to_period("M").astype(str)
        entry = {
            "forecasts": len(group),
            "selected": int(group.selected.sum()),
            "grading_status": group.grading_status.value_counts().to_dict(),
            "performance": metrics(valid),
        }
        for column in ["day", "week", "month", "target", "model_version", "policy"]:
            entry[column] = (
                {str(k): metrics(g) for k, g in valid.groupby(column)} if len(valid) else {}
            )
        results["market_backed" if backed else "hypothetical"] = entry
    return {"mode": "forward_paper_tracking", "n": len(frame), "groups": results}
