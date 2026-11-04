from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nba_player_prop_prediction.config import Settings
from nba_player_prop_prediction.features.history import (
    chronological_split,
    prior_game_features,
    select_games,
)
from nba_player_prop_prediction.ingestion.collector import Collector, import_file
from nba_player_prop_prediction.preprocessing.clean import clean_games, parse_minutes
from nba_player_prop_prediction.reporting import quality_report
from nba_player_prop_prediction.utils.io import read_table, write_parquet

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_games.csv"


@pytest.fixture
def raw():
    return read_table(FIXTURE)


def clean(raw):
    return clean_games(raw, season_type="Regular Season")


@pytest.mark.parametrize("value,expected", [(30, 30), ("30:30", 30.5), ("0:01", 1 / 60)])
def test_minutes(value, expected):
    assert parse_minutes(value) == expected


@pytest.mark.parametrize("value", [None, "DNP", "INACTIVE", ""])
def test_missing_minutes(value):
    assert pd.isna(parse_minutes(value))


@pytest.mark.parametrize("value", ["12:60", "hello", "12:3"])
def test_bad_minutes(value):
    with pytest.raises(ValueError):
        parse_minutes(value)


def test_cleaning_identity_trades_and_nulls(raw):
    result = clean(raw)
    assert result.rejected.empty
    data = result.data
    assert data.game_date.iloc[0] == pd.Timestamp("2023-10-24")
    assert data.game_id.iloc[0] == "0022300001"
    assert data.player_game_id.is_unique
    assert set(data.loc[data.player_id.eq("1001"), "team_id"]) == {"10", "20"}
    assert data.opponent_id.iloc[0] == "20"
    assert data.home_away.iloc[0] == "home"
    assert data.participation.iloc[-1] == "dnp"
    assert pd.isna(data.points.iloc[-1])
    assert data.pra.iloc[0] == 31
    assert_frame_equal(clean_games(data).data, data)


def test_duplicates_and_conflicts(raw):
    identical = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    assert len(clean(identical).data) == 4
    assert clean(identical).audit["duplicate_rows"] == 1
    identical.loc[4, "PTS"] = "99"
    result = clean(identical)
    assert len(result.rejected) == 2
    assert result.rejected.validation_errors.str.contains("conflicting_player_game").all()


@pytest.mark.parametrize(
    "column,value",
    [
        ("PTS", "-1"),
        ("REB", "1.5"),
        ("MIN", "oops"),
        ("GAME_DATE", "bad"),
        ("PLAYER_ID", "bad"),
        ("GAME_ID", "0"),
        ("AST", "inf"),
        ("FG3M", "10"),
        ("WL", "T"),
        ("SEASON_YEAR", "2023-25"),
    ],
)
def test_invalid_records(raw, column, value):
    raw.loc[0, column] = value
    result = clean(raw)
    assert len(result.rejected) == 1


def test_schema_and_context(raw):
    with pytest.raises(ValueError, match="required"):
        clean(raw.drop(columns="PLAYER_ID"))
    with pytest.raises(ValueError, match="season_type"):
        clean_games(raw)
    assert clean_games(raw, season_type="Playoffs").data.season_type.eq("Playoffs").all()


def test_unknown_is_not_zero_or_dnp(raw):
    raw.loc[0, "MIN"] = None
    raw.loc[0, ["PTS", "REB", "AST", "FG3M", "FGA", "FG3A", "FTA", "TOV", "PF"]] = None
    data = clean(raw).data
    assert data.participation.iloc[0] == "unknown"
    assert pd.isna(data.points.iloc[0])


def test_cache_retry_refresh(raw, tmp_path):
    calls, waits = [], []

    def fetch(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("test")
        return raw

    collector = Collector(Settings(data_dir=tmp_path, retries=1, pacing=0), fetch, waits.append)
    args = ("2023-24", "Regular Season", "2023-10-24", "2023-10-24")
    first = collector.collect(*args)
    assert collector.collect(*args) == first
    assert len(calls) == 2
    second = collector.collect(*args, refresh=True)
    assert second != first and first.exists()
    assert len(calls) == 3
    assert first.with_suffix(".json").exists()
    assert 2 in waits


def test_failed_request_does_not_cache(tmp_path):
    def fetch(**kwargs):
        raise TimeoutError("offline")

    collector = Collector(Settings(data_dir=tmp_path, retries=0), fetch, lambda _: None)
    with pytest.raises(TimeoutError):
        collector.collect("2023-24", "Playoffs", "2024-04-01", "2024-04-02")
    assert not list(tmp_path.rglob("latest.json"))


def test_import_export_and_report(raw, tmp_path):
    settings = Settings(data_dir=tmp_path)
    path = import_file(FIXTURE, settings, "Synthetic test fixture; not NBA observations")
    assert path.read_bytes() == FIXTURE.read_bytes()
    assert import_file(FIXTURE, settings, "fixture") == path
    result = clean(read_table(path))
    output = tmp_path / "clean.parquet"
    write_parquet(output, result.data)
    assert_frame_equal(read_table(output), result.data)
    report = quality_report(result.data, tmp_path / "report", result.audit, plots=True)
    assert report["observations"] == 4
    assert (tmp_path / "report" / "distributions.png").exists()
    with pytest.raises(ValueError):
        read_table(tmp_path / "bad.json")


def test_history_excludes_target_and_future(raw):
    data = clean(raw).data
    features = prior_game_features(data)
    assert features.prior_games.tolist() == [0, 0, 1, 0]
    changed = data.copy()
    changed.loc[2, "minutes"] = 999
    assert_frame_equal(features, prior_game_features(changed))
    assert features.prior_5_minutes_mean.iloc[2] == 30.5
    assert len(select_games(data, "1001", start="2023-10-25")) == 1
    train, validation, test = chronological_split(data, "2023-10-25", "2023-10-26")
    assert (len(train), len(validation), len(test)) == (2, 0, 2)
    same_day = data.copy()
    same_day.loc[2, "game_date"] = same_day.game_date.iloc[0]
    assert prior_game_features(same_day).prior_games.eq(0).all()


def test_empty_data(raw, tmp_path):
    result = clean(raw.iloc[:0])
    assert result.data.empty
    quality_report(result.data, tmp_path)


def test_invalid_request_settings_and_ranges(tmp_path):
    with pytest.raises(ValueError):
        Settings(timeout=0)
    with pytest.raises(ValueError):
        Settings(pacing=float("nan"))
    collector = Collector(Settings(data_dir=tmp_path), lambda **_: pytest.fail("network"))
    for args in [
        ("2023", "Playoffs", "2023-01-01", "2023-01-02"),
        ("2023-25", "Playoffs", "2023-01-01", "2023-01-02"),
        ("2023-24", "Bad", "2023-01-01", "2023-01-02"),
        ("2023-24", "Playoffs", "2023-01-02", "2023-01-01"),
    ]:
        with pytest.raises(ValueError):
            collector.collect(*args)


def test_rate_limit_and_corrupt_cache(raw, tmp_path):
    from requests import HTTPError, Response

    calls, waits = [], []

    def fetch(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            response = Response()
            response.status_code = 429
            response.headers["Retry-After"] = "4"
            raise HTTPError(response=response)
        return raw

    collector = Collector(Settings(data_dir=tmp_path, retries=1), fetch, waits.append)
    args = ("2023-24", "Regular Season", "2023-10-24", "2023-10-24")
    path = collector.collect(*args)
    assert 4 in waits
    path.write_bytes(b"corrupt")
    assert collector.collect(*args) != path
    assert len(calls) == 3


def test_cli_round_trip(monkeypatch, tmp_path):
    import sys

    from nba_player_prop_prediction.cli import main

    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NBA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        sys, "argv", ["nba-props", "import", str(FIXTURE), "--source", "Synthetic fixture"]
    )
    main()
    source = next((tmp_path / "data" / "raw").rglob("*.csv"))
    monkeypatch.setattr(
        sys, "argv", ["nba-props", "preprocess", str(source), "--season-type", "Regular Season"]
    )
    main()
    output = tmp_path / "data" / "processed" / "games.parquet"
    assert len(read_table(output)) == 4
    monkeypatch.setattr(sys, "argv", ["nba-props", "report", str(output)])
    main()
    assert (tmp_path / "artifacts" / "quality" / "quality.md").exists()
