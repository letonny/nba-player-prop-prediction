"""Offline coverage for bounded collection and overlapping cache snapshots."""

from pathlib import Path

import pytest

from nba_player_prop_prediction.config import Settings
from nba_player_prop_prediction.ingestion.collector import Collector
from nba_player_prop_prediction.ingestion.expansion import assemble, expand
from nba_player_prop_prediction.utils.io import read_table


def test_assemble_latest_overlap_and_checksums(tmp_path):
    raw = read_table(Path(__file__).parent / "fixtures" / "synthetic_games.csv")
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    client = Collector(settings, lambda **_: raw.copy(), lambda _: None)
    first = client.collect("2023-24", "Regular Season", "2023-10-24", "2023-10-26")
    second = client.collect("2023-24", "Regular Season", "2023-10-24", "2023-10-27")
    output = tmp_path / "history.parquet"
    result = assemble(settings, output)
    assert result["rows"] == 4 and result["overlapping_rows_replaced"] == 4
    assert result["games"] == 2
    assert first.exists()
    second.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        assemble(settings, output)


def test_expansion_budget_and_failures(monkeypatch, tmp_path):
    calls = []

    def collect(self, *args):
        calls.append(args)
        if len(calls) == 1:
            raise TimeoutError("offline fixture")
        return Path("fixture.parquet")

    monkeypatch.setattr(Collector, "collect", collect)
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path)
    result = expand(settings, [2022, 2023, 2024], max_requests=2)
    assert len(result["failed"]) == 1
    assert len(result["completed"]) == 1
    assert result["pending"] == [2024]
    assert (tmp_path / "expansion.json").exists()
