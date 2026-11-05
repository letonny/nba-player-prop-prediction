"""Offline count semantics, chronology, calibration, and adapters."""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from nba_player_prop_prediction.probabilistic.calibration import Calibrator
from nba_player_prop_prediction.probabilistic.data import enrich, partitions
from nba_player_prop_prediction.probabilistic.distributions import TARGETS, CountDistribution
from nba_player_prop_prediction.probabilistic.evaluation import confidence_report
from nba_player_prop_prediction.probabilistic.minutes import walk_forward
from nba_player_prop_prediction.probabilistic.models import Candidate


@pytest.mark.parametrize("alpha", [0, 0.2])
def test_threshold_semantics(alpha):
    d = CountDistribution(np.array([5.0, 15.0]), alpha)
    np.testing.assert_allclose(d.probability(20), d.probability(19.5, "over"))
    np.testing.assert_allclose(d.probability(19.5, "under") + d.probability(19.5, "over"), 1)
    np.testing.assert_allclose(d.probability(20, "over"), d.probability(21))
    assert (d.probability(0) == 1).all()
    assert (d.probability(0, "under") == 0).all()
    ps = np.column_stack([d.probability(t) for t in range(40)])
    assert (np.diff(ps, axis=1) <= 0).all()
    ref = (
        poisson.sf(19, d.mean) if not alpha else nbinom.sf(19, 1 / alpha, 1 / (1 + alpha * d.mean))
    )
    np.testing.assert_allclose(d.probability(20), ref)
    assert (d.quantile(0.05) >= 0).all()


def test_parameters_and_zero_inflation():
    with pytest.raises(ValueError):
        CountDistribution([-1])
    with pytest.raises(ValueError):
        CountDistribution([1], -0.1)
    with pytest.raises(ValueError):
        CountDistribution([1], 0, 1)
    d = CountDistribution([3], 0.2, 0.4)
    assert d.expected[0] == pytest.approx(1.8)
    assert d.cdf(0)[0] > 0.4
    assert np.exp(d.logpmf([0]))[0] == pytest.approx(d.cdf(0)[0])
    assert set(TARGETS) == {"points", "rebounds", "assists", "threes_made"}


@pytest.mark.parametrize("method", ["identity", "sigmoid", "isotonic"])
def test_calibration_monotonicity_separation(method):
    p = np.linspace(0.01, 0.99, 400)
    y = (np.arange(400) % 3 == 0).astype(int)
    c = Calibrator(method).fit(p, y, ["2025-01-01"] * 400, "2025-07-01")
    assert (np.diff(c.predict(p)) >= -1e-12).all()
    assert ((c.predict(p) >= 0) & (c.predict(p) <= 1)).all()
    with pytest.raises(ValueError, match="overlaps"):
        Calibrator(method).fit(p, y, ["2025-07-01"] * 400, "2025-07-01")
    with pytest.raises(ValueError, match="Insufficient"):
        Calibrator(method).fit(p[:2], y[:2], ["2025-01-01"] * 2, "2025-07-01")


@pytest.fixture
def small():
    # Simple feature rows; labels deliberately separate from design columns.
    n = 80
    dates = pd.date_range("2023-01-01", periods=n)
    features = pd.DataFrame(
        {
            "player_game_id": [f"1:{i}" for i in range(n)],
            "player_id": "1",
            "game_id": [str(i) for i in range(n)],
            "game_date": dates,
            "season": "2022-23",
            "team_id": "A",
            "opponent_id": "B",
            "home_away": "home",
            "season_type": "Regular Season",
            "prior_10_minutes_std": 2.0,
            "lag": np.arange(n) % 9,
        }
    )
    history = features.copy()
    history["minutes"] = 15 + np.arange(n) % 9
    history["participation"] = "played"
    for t in TARGETS:
        history[t] = np.arange(n) % 6
    return history, features


def test_oof_target_and_future_changes(small, tmp_path):
    h, f = small
    args = dict(
        boundaries=["2023-02-01", "2023-03-01", "2023-04-01"],
        iterations=3,
        min_train=20,
        min_dates=20,
    )
    a = walk_forward(h, f, tmp_path / "a", **args)
    changed = h.copy()
    changed.loc[changed.game_date >= "2023-02-01", "minutes"] = 500
    b = walk_forward(changed, f, tmp_path / "b", **args)
    mask = a.game_date < "2023-03-01"
    np.testing.assert_allclose(a.loc[mask, "predicted_minutes"], b.loc[mask, "predicted_minutes"])
    assert (a.minutes_trained_through < a.game_date).all()
    assert a.game_date.min() == pd.Timestamp("2023-02-01")
    data = enrich(h, f, a)
    assert (
        data.label_points.tolist()
        == h.set_index("player_game_id").loc[data.player_game_id, "points"].tolist()
    )
    assert "minutes" not in data
    bad = a.copy()
    bad["minutes_trained_through"] = bad.game_date
    with pytest.raises(ValueError, match="overlaps"):
        enrich(h, f, bad)
    with pytest.raises(ValueError):
        partitions(data, "2025-01-01", "2024-01-01", "2026-01-01")


@pytest.mark.parametrize(
    "name",
    [
        "catboost",
        "catboost_quantile",
        "ngboost",
        "xgblss_Poisson",
        "xgblss_NegativeBinomial",
        "xgblss_ZIPoisson",
        "xgblss_ZINB",
    ],
)
def test_adapter_roundtrip(name, small):
    if name.startswith("xgblss"):
        pytest.importorskip("xgboostlss")
    if name == "ngboost":
        pytest.importorskip("ngboost")
    _, f = small
    f["label_points"] = np.arange(len(f)) % 7
    columns = ["lag", "player_id", "team_id", "opponent_id", "home_away", "season_type"]
    c = Candidate(name, "points", columns, rounds=2).fit(f.iloc[:50], f.iloc[50:65])
    a = c.predict(f.iloc[65:])
    b = pickle.loads(pickle.dumps(c)).predict(f.iloc[65:])
    if isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b)
        assert (np.diff(a, axis=1) >= 0).all()
    else:
        np.testing.assert_allclose(a.expected, b.expected)
        assert np.isfinite(a.logpmf(f.label_points.iloc[65:])).all()
    with pytest.raises(ValueError, match="schema"):
        c.predict(f.drop(columns="lag"))


def test_confidence_counts_no_duplicates():
    f = pd.DataFrame(
        {
            "player_game_id": ["a", "a", "b", "b"],
            "game_id": ["1", "1", "2", "2"],
            "line": [10, 15, 10, 15],
            "probability": [0.9, 0.8, 0.9, 0.7],
            "hit": [True, False, False, True],
        }
    )
    r = confidence_report(f, replicates=20)["0.9"]
    assert r["selected"] == 2 and r["wins"] == 1 and r["losses"] == 1
    assert r["unique_player_games"] == 2 and r["unique_games"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        confidence_report(pd.concat([f, f]))


def test_enrichment_future_mutation_and_boundaries(small, tmp_path):
    h, f = small
    o = walk_forward(
        h,
        f,
        tmp_path / "o",
        boundaries=["2023-02-01", "2023-04-01"],
        iterations=2,
        min_train=20,
        min_dates=20,
    )
    before = enrich(h, f, o)
    changed = h.copy()
    changed.loc[changed.game_date >= "2023-02-15", "points"] = 100
    after = enrich(changed, f, o)
    columns = ["season_mean_points", "rate_points", "prior_std_points"]
    np.testing.assert_allclose(
        before.loc[before.game_date <= "2023-02-15", columns],
        after.loc[after.game_date <= "2023-02-15", columns],
        equal_nan=True,
    )
    parts = partitions(before, "2023-02-10", "2023-02-20", "2023-03-01")
    assert parts["train"].game_date.max() < parts["validation"].game_date.min()
    assert parts["validation"].game_date.max() < parts["calibration"].game_date.min()
    assert parts["calibration"].game_date.max() < parts["test"].game_date.min()


def test_certain_thresholds_survive_calibration():
    c = Calibrator("sigmoid").fit(
        np.linspace(0.01, 0.99, 400), np.arange(400) % 2, ["2025-01-01"] * 400, "2025-07-01"
    )
    np.testing.assert_array_equal(c.predict([0, 1]), [0, 1])
