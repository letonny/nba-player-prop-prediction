"""Date-only historical interfaces: same-day games are always excluded."""

import pandas as pd

from nba_player_prop_prediction.preprocessing.clean import TARGETS


def select_games(
    frame: pd.DataFrame,
    player_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    selected = frame.copy()
    if player_id is not None:
        selected = selected.loc[selected.player_id == str(player_id)]
    if start is not None:
        selected = selected.loc[selected.game_date >= pd.Timestamp(start)]
    if end is not None:
        selected = selected.loc[selected.game_date <= pd.Timestamp(end)]
    return selected.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def prior_game_features(frame: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Return identity plus prior played-game means, never target-game box scores.

    Dates are not scheduled tipoff timestamps. Excluding all same-day records is
    conservative; production must additionally enforce source availability time.
    """
    if window < 1:
        raise ValueError("window must be positive")
    ordered = select_games(frame)
    rows = []
    for _, game in ordered.iterrows():
        history = ordered.loc[
            (ordered.player_id == game.player_id)
            & (ordered.game_date < game.game_date)
            & ordered.participation.eq("played")
        ].tail(window)
        row = {"player_game_id": game.player_game_id, "prior_games": len(history)}
        for col in ["minutes", *TARGETS]:
            mean = history[col].mean()
            row[f"prior_{window}_{col}_mean"] = float(mean) if pd.notna(mean) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def chronological_split(frame: pd.DataFrame, validation_start: str, test_start: str):
    validation, test = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if validation >= test:
        raise ValueError("validation_start must precede test_start")
    ordered = select_games(frame)
    return (
        ordered.loc[ordered.game_date < validation].copy(),
        ordered.loc[(ordered.game_date >= validation) & (ordered.game_date < test)].copy(),
        ordered.loc[ordered.game_date >= test].copy(),
    )
