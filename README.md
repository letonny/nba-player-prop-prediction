# nba-player-prop-prediction
Machine learning platform for NBA player prop predictions using probabilistic modeling, calibration, and historical data.

## Status and objectives

Milestone 1 implements local historical ingestion, validation, Parquet storage,
quality reporting, and conservative historical data interfaces. No models have
been trained and no predictive performance or betting hit rate is claimed.
The objective is reliable, empirically validated probabilities and hit rates,
not payout or expected value. A predicted 90% probability is not evidence of
90% accuracy or a guarantee of winning.

## Installation on macOS

Use Python 3.11 or newer (for example `brew install python@3.13`). From this
repository, with that Python on your PATH:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plots]'
cp .env.example .env
```

`plots` is optional. Core dependencies are pandas, NumPy, SciPy, scikit-learn,
nba_api, PyArrow and python-dotenv. pytest and Ruff are development extras.
All computation runs locally; no paid API, account, GPU, or database is needed.
Environment settings are optional and do not override already exported values.
Paths are relative to your current working directory; run commands at repo root.

## Repository layout

- `src/nba_player_prop_prediction/ingestion/`: NBA requests, cache, source imports.
- `src/nba_player_prop_prediction/preprocessing/`: canonical schema and validation.
- `src/nba_player_prop_prediction/features/`: chronological history interfaces.
- `src/nba_player_prop_prediction/reporting.py`: readable quality reports and plots.
- `src/nba_player_prop_prediction/config.py`, `utils/`: settings, logging, atomic writes.
- `data/raw/`, `data/processed/`: ignored original snapshots and derived outputs.
- `artifacts/`: ignored reports and future model outputs.
- `tests/fixtures/`: explicitly synthetic, trackable offline test data.
- `scripts/`, `configs/`, `notebooks/`: report wrapper, configuration guide, optional exploration.

Future model, calibration, evaluation and prediction packages will be introduced
when implemented, rather than populated with empty classes now.

## Data sources and collection

The source is NBA Stats through the open-source
[nba_api PlayerGameLogs endpoint](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playergamelogs.md).
The endpoint supplies player and game IDs, dates, game-specific team context,
matchup, result and traditional box-score statistics. Access can time out or be
blocked. This software does not bypass access controls. Dataset usage is subject
to the source's terms; the MIT license covers this code.

Start with one historical day:

```bash
nba-props ingest --start-season 2023 --start-date 2023-10-24 --end-date 2023-10-24
```

Only after verifying that sample, request full seasons (starting years inclusive):

```bash
nba-props ingest --start-season 2022 --end-season 2023
nba-props ingest --start-season 2023 --season-type Playoffs
```

Full seasons use monthly windows from July through the following October to
cover delayed seasons such as 2019-20. Requests default to 30-second timeouts,
1.5-second pacing and two retries with exponential backoff; numeric Retry-After
headers are respected when exposed by the HTTP exception. NBA API errors do not
always expose HTTP headers. Completed windows are reused on reruns, so interrupted
jobs resume. Failures are logged and saved to `artifacts/ingestion_failures.json`;
the command exits nonzero if any window fails. Empty responses with valid columns
are cached but logged; they are not proof that no games existed.

`--refresh` retrieves a new immutable snapshot, preserving previous snapshots.
Use it for recent games, corrections or empty results. Cache manifests include
request parameters, UTC retrieval time, row counts and SHA-256 checksums. Only
one writer per request window is supported. Use printed snapshot paths for
preprocessing; do not blindly glob every historical refresh version together.

If NBA endpoints are inaccessible, import your own legitimate CSV or Parquet:

```bash
nba-props import /path/to/player_logs.csv --source 'Source URL or dataset description'
```

Imports preserve original bytes under a content hash and record provenance.
No fallback production statistics are generated. The fixture is synthetic and
must never be presented as historical NBA data.

## Preprocessing and schema

Pass one or more raw snapshot paths printed by ingestion/import:

```bash
nba-props preprocess data/raw/nba_api/REQUEST_KEY/SNAPSHOT.parquet
# For external data without competition metadata:
nba-props preprocess /path/to/player_logs.csv --season 2023-24 --season-type 'Regular Season'
```

Output defaults to `data/processed/games.parquet`, accompanied by
`games.rejected.parquet` and `games.audit.json`. Every run rebuilds this output
from the explicitly provided inputs; include all desired partitions. Inputs are
never edited. Rejected rows have `validation_errors`; exit code 2 signals that
quarantine needs review even though accepted rows were exported. Missing required
columns or missing season context raise an error. Clean data can be reprocessed
idempotently.

CSV headers can use NBA names or canonical lowercase names. Required columns:
`player_id`, `player_name`, `team_id`, `game_id`, `game_date`, `minutes`, `points`,
`rebounds`, `assists`, `threes_made`. Season and competition type must be present
or supplied explicitly. API context supplies both. Optional fields include
`team_name`, `matchup`, `result`, `field_goal_attempts`, `three_point_attempts`,
`free_throw_attempts`, `turnovers`, `personal_fouls`, `plus_minus`.

IDs are strings; game IDs are padded to 10 digits. The stable identity is
`player_id:game_id`, independent of traded-player team changes. Identical records
are deduplicated; conflicting box scores/context for the same identity are all
quarantined. Regular season and playoff context is retained in `season_type`.
Game dates normalize to dates, not inferred tipoff times. Minutes support numeric
values and MM:SS. Negative, malformed, fractional count statistics and selected
cross-field inconsistencies are rejected. No arbitrary upper cap rejects overtime.
Optional missing statistics remain nullable integers; unknown values never become
zero. `pra` is null if any of its components is unknown.

Explicit DNP/DND/INACTIVE minute markers become `participation=dnp`. Positive
minutes or statistics indicate played; otherwise participation remains unknown.
A zero-minute line alone does not prove DNP (rounded sub-minute appearances exist).
PlayerGameLogs may omit DNPs altogether, so this dataset cannot establish roster
availability or a complete universe of scheduled player appearances.
`opponent` and `home_away` come from matchup; `opponent_id` is derived only when
both teams appear in the input for that game. Missing context remains null.

## Reports and offline tests

```bash
nba-props report data/processed/games.parquet --plots
# Equivalent standalone script:
python scripts/quality_report.py data/processed/games.parquet --plots
python -m pytest
ruff check .
ruff format --check .
```

Reports in `artifacts/quality/` include Markdown, JSON and optional PNG histograms:
row/player/team totals, seasons, date range, missing percentages, duplicates,
minutes/points/rebounds/assists distributions, participation, games per player,
and the preprocessing audit when available. Reports describe observed rows,
not necessarily complete NBA coverage. Tests use synthetic fixtures and injected
API clients and require no network.

## Milestone 2 and leakage prevention

`features.history.select_games` provides ordered player/date filtering;
`prior_game_features` returns prior played-game means and identities only;
`chronological_split` creates train/validation/test periods with dates kept together.
The initial history helper favors clarity over speed and should be optimized for
multi-season feature workloads. Targets are points, rebounds, assists, made threes,
and PRA. Historical final minutes can be labels or lagged values, never a current
game feature. All same-day games are excluded because scheduled tipoff timestamps
are unavailable in game logs.

Milestone 2 will add lagged 5/10/20-game and season-to-date features, rest and
back-to-back indicators, and the minutes model. Pregame starters, teammate injury
status, rotation news and matchup/blowout signals require additional timestamped
sources. Enforce `available_at < scheduled_tipoff` for every future external feature;
retrospective game logs alone do not prove real-time availability. Fit imputers,
scalers, feature selection and models on chronological training data only. Use
validation for tuning/calibration and an untouched later holdout for evaluation.
Never include current-game final minutes, starting status or box scores as predictors.

Planned subsequent stages: XGBoostLSS distribution models, CatBoost, NGBoost,
Optuna tuning and optionally PyMC/PyTorch Forecasting; calibration with Brier score,
log loss, reliability diagrams, probability buckets and coverage versus hit rate;
then configurable alternate-line probabilities and correlation-aware PRA/multi-leg
analysis. Planned thresholds include points 10–35 by 5, rebounds 4/6/8/10/12,
assists 3/5/7/10 and threes 1–5. Next.js/TypeScript/Tailwind, Supabase and Vercel
are future application work, not part of Milestone 1. None of these advanced
libraries or services are required or installed here.

## License

MIT; see [LICENSE](LICENSE).

## Milestone 1 verification (2026-11-04)

Verified in a fresh local virtual environment on macOS with Python 3.14.5:

- Editable installation including development/plot extras succeeded; `pip check` passed.
- A live NBA Stats request for 2023-10-24 returned 44 observations, 44 players,
  four teams and two games. Network access required running outside the sandbox.
- All 44 observations passed preprocessing, with zero duplicates or rejected rows.
  Parquet was read back, identities checked and idempotent reprocessing verified.
- Repeating the collection used the local cache without a network request.
- Markdown/JSON quality reports and PNG distributions were generated in `artifacts/quality/`.
- 32 offline pytest cases passed; Ruff lint and formatting checks passed.
- Package import, `pip check`, and `git diff --check` passed. Raw data, processed
  Parquet and generated reports are excluded from Git; no credentials were added.

The sandbox emitted nonfatal PyArrow CPU-information and initial Matplotlib font
cache warnings; file exports completed successfully. Full multi-season downloads,
other Python versions and completeness against an independent NBA schedule have
not been verified. Missing pregame availability/tipoff sources remain a constraint
for future models, not a claim of completed pregame prediction support.
