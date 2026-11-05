"""Offline selection and immutable tracking checks."""

import pandas as pd
import pytest

from nba_player_prop_prediction.selection.backtest import metrics
from nba_player_prop_prediction.selection.markets import OddsAPI, confirmed_market
from nba_player_prop_prediction.selection.policy import Policy, rank
from nba_player_prop_prediction.selection.tracking import Ledger


@pytest.fixture
def candidates():
    return pd.DataFrame(
        [
            dict(
                candidate_id=str(i),
                player_game_id=f"p{i // 2}",
                game_id="g",
                game_date="2026-01-01",
                probability=p,
                target="points",
                line=10 + i,
                event="at_least",
                calibration_n=500,
                market_available=False,
                availability="unknown",
                prior_games=20,
                minutes_restricted=False,
                hit=bool(i % 2),
            )
            for i, p in enumerate([0.95, 0.9, 0.89, 0.7])
        ]
    )


def test_rank_limits_and_leakage(candidates):
    a = rank(candidates)
    assert a.loc[a.selected, "candidate_id"].tolist() == ["0", "2"]
    changed = candidates.copy()
    changed["hit"] = ~changed.hit
    assert rank(changed).selected.tolist() == a.selected.tolist()
    assert (
        rank(candidates.sample(frac=1, random_state=1)).candidate_id.tolist()
        == a.candidate_id.tolist()
    )
    assert rank(candidates, Policy(max_per_game=1)).selected.sum() == 1
    assert rank(candidates, Policy(max_total=1)).selected.sum() == 1
    assert rank(candidates, Policy(max_per_player_game=2)).selected.sum() == 3


@pytest.mark.parametrize(
    "policy",
    [
        Policy(minimum=0.99),
        Policy(require_market=True),
        Policy(require_available=True),
        Policy(min_history=21),
        Policy(min_calibration=501),
        Policy(max_total=0),
    ],
)
def test_abstain(candidates, policy):
    assert not rank(candidates, policy).selected.any()


def test_bad_candidates_and_risks(candidates):
    with pytest.raises(ValueError):
        rank(pd.concat([candidates, candidates]))
    bad = candidates.copy()
    bad.loc[0, "probability"] = 1.1
    with pytest.raises(ValueError):
        rank(bad)
    with pytest.raises(ValueError):
        rank(candidates.drop(columns="availability"))
    candidates.loc[0, "availability"] = "out"
    candidates.loc[2, "minutes_restricted"] = True
    assert rank(candidates).loc[lambda f: f.selected, "candidate_id"].tolist() == ["1"]
    assert metrics(candidates.iloc[:0])["hit_rate"] is None


def test_market_time_and_identity(tmp_path):
    snapshot = dict(
        source="fixture",
        snapshot_at="2026-01-01T12:00Z",
        sportsbook="test",
        market_id="m",
        player_id="1",
        game_id="g",
        target="points",
        line=19.5,
        event="over",
        odds=1.1,
        available=True,
    )
    assert confirmed_market(snapshot, "1", "g", "points", 19.5, "over", "2026-01-01T12:01Z")
    assert not confirmed_market(snapshot, "1", "g", "points", 19.5, "over", "2026-01-01T11:00Z")
    assert not confirmed_market(snapshot, "2", "g", "points", 19.5, "over", "2026-01-01T12:01Z")
    assert not OddsAPI(tmp_path, key="").snapshot("g", ["player_points"])["available"]


def payload():
    return dict(
        mode="forward",
        player_id="1",
        game_id="g",
        target="points",
        line=20,
        event="at_least",
        model_version="v",
        policy="fixed",
        probability=0.9,
        source_cutoff="2026-01-01T10:00Z",
        generated_at="2026-01-01T11:00Z",
        scheduled_tipoff="2026-01-01T19:00Z",
    )


@pytest.mark.parametrize(
    "actual,participation,status",
    [(20, "played", "hit"), (19, "played", "miss"), (None, "dnp", "unable_to_grade")],
)
def test_ledger_grading(monkeypatch, tmp_path, actual, participation, status):
    from nba_player_prop_prediction.selection import tracking

    monkeypatch.setattr(tracking, "_now", lambda: pd.Timestamp("2026-01-01T12:00Z"))
    ledger = Ledger(tmp_path / "p.sqlite")
    p = payload()
    key = ledger.record(p)
    assert ledger.record(p) == key
    assert len(ledger.export()) == 1 and ledger.export()[0]["grading_status"] == "pending"
    with pytest.raises(ValueError):
        ledger.record(dict(p, probability=0.95))
    monkeypatch.setattr(tracking, "_now", lambda: pd.Timestamp("2026-01-02T12:00Z"))
    with pytest.raises(ValueError):
        ledger.record(dict(p, game_id="x"))
    assert ledger.grade(key, actual, participation, "2026-01-01T22:00Z", "fixture") == status
    assert ledger.grade(key, actual, participation, "2026-01-01T22:00Z", "fixture") == status
    with pytest.raises(ValueError):
        ledger.grade(key, 30, "played", "2026-01-01T22:00Z", "changed")


def test_push_and_market_rules(monkeypatch, tmp_path):
    from nba_player_prop_prediction.selection import tracking

    monkeypatch.setattr(tracking, "_now", lambda: pd.Timestamp("2026-01-01T12:00Z"))
    ledger = Ledger(tmp_path / "p.sqlite")
    push = ledger.record(dict(payload(), event="over"))
    market = ledger.record(dict(payload(), game_id="market", market_available=True))
    monkeypatch.setattr(tracking, "_now", lambda: pd.Timestamp("2026-01-02T12:00Z"))
    assert ledger.grade(push, 20, "played", "2026-01-01T22:00Z", "fixture") == "void"
    with pytest.raises(ValueError):
        ledger.grade(market, None, "dnp", "2026-01-01T22:00Z", "fixture")
    assert (
        ledger.grade(market, None, "dnp", "2026-01-01T22:00Z", "book settlement", "void") == "void"
    )


def test_empty_and_forward_report(candidates):
    from nba_player_prop_prediction.selection.engine import generate
    from nba_player_prop_prediction.selection.reports import forward_summary

    assert rank(candidates.iloc[:0]).empty
    assert rank(generate(None, [], None, None, None)).empty
    assert forward_summary([])["n"] == 0
    row = dict(payload(), selected=True, grading_status="hit", actual=20)
    result = forward_summary([row])
    assert result["groups"]["hypothetical"]["performance"]["hit_rate"] == 1


def test_adapter_cache(monkeypatch, tmp_path):
    from nba_player_prop_prediction.selection import markets

    calls = []

    class Response:
        status_code = 200
        headers = {"x-requests-remaining": "499"}

        def json(self):
            return {"bookmakers": []}

    monkeypatch.setattr(markets.time, "sleep", lambda _: None)
    monkeypatch.setattr(markets.requests, "get", lambda *a, **k: calls.append(k) or Response())
    provider = OddsAPI(tmp_path, key="fixture")
    first = provider.snapshot("abc", ["player_points"])
    assert first == provider.snapshot("abc", ["player_points"])
    assert len(calls) == 1
    assert "fixture" not in str(first)


def test_optional_schedule_restrictions(candidates):
    assert not rank(candidates, Policy(exclude_back_to_back=True)).selected.any()
    candidates["back_to_back"] = 0
    candidates["minutes_change"] = -5
    assert not rank(candidates, Policy(minimum_minutes_change=-3)).selected.any()
