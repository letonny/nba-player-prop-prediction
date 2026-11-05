# nba-player-prop-prediction
Machine learning platform for NBA player prop predictions using probabilistic modeling, calibration, and historical data.

## Status and objectives

Milestone 1 provides local historical ingestion, validation, Parquet storage,
quality reporting, and conservative historical data interfaces. Milestone 2 adds
pregame features and evaluated minutes models; see its verification section below.
No betting hit rate is claimed.
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

Minutes training, evaluation and inference now live in `models/minutes/`.
Future probabilistic model and calibration packages will be added when implemented.

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

Milestone 2 adds lagged 3/5/10/20-game features, rest and
back-to-back appearance indicators, and the minutes model (details below). Pregame starters, teammate injury
status, rotation news and matchup/blowout signals require additional timestamped
sources. Enforce `available_at < scheduled_tipoff` for every future external feature;
retrospective game logs alone do not prove real-time availability. Fit imputers,
scalers, feature selection and models on chronological training data only. Use
validation for tuning/calibration and an untouched later holdout for evaluation.
Never include current-game final minutes, starting status or box scores as predictors.

Planned subsequent stages: XGBoostLSS distribution models, CatBoost stat projections, NGBoost,
Optuna tuning and optionally PyMC/PyTorch Forecasting; calibration with Brier score,
log loss, reliability diagrams, probability buckets and coverage versus hit rate;
then configurable alternate-line probabilities and correlation-aware PRA/multi-leg
analysis. Planned thresholds include points 10–35 by 5, rebounds 4/6/8/10/12,
assists 3/5/7/10 and threes 1–5. Next.js/TypeScript/Tailwind, Supabase and Vercel
are future application work. Only CatBoost is installed for Milestone 2; the other
advanced modeling libraries and services remain deferred.

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

## Milestone 2: historical features and minutes models

Implemented: bounded multi-season expansion, latest-snapshot assembly, versioned
pregame features, three baselines, CatBoost regression, chronological evaluation,
validation-residual intervals, native model serialization and local pregame inference.
No player-stat distribution model or frontend is included.

Install the optional minutes dependencies (CatBoost runs on CPU with two threads):

```bash
source .venv/bin/activate
python -m pip install '.[dev,plots,minutes]'
# During source development, ensure commands use this checkout:
export PYTHONPATH="$PWD/src"
```

A standard wheel install was used because editable `.pth` discovery failed in the
verification environment. Reinstall after source changes, or use the explicit
PYTHONPATH above. No advanced probabilistic libraries are needed yet.

### Historical expansion

```bash
# Limited real request first:
nba-props ingest --start-season 2022 --start-date 2022-10-18 --end-date 2022-10-18
# Four historical seasons; default years are 2022 2023 2024 2025:
python -m nba_player_prop_prediction.milestone2 collect --max-requests 4 --budget-seconds 180
python -m nba_player_prop_prediction.milestone2 assemble
# Optional current season, separate request:
python -m nba_player_prop_prediction.milestone2 collect --years 2026 --max-requests 1
```

Collection uses one regular-season request per year, with existing M1 pacing,
timeouts, retries and immutable raw snapshots. The time budget is checked between
requests, so an in-flight request can run through its configured timeout/retries.
Rerun the same command to resume; completed requests hit the cache. Use `--refresh`
for corrected or current data. `artifacts/expansion.json` records completed, failed
and pending work; incomplete work exits nonzero. M1 monthly-window commands remain
available if a whole-season response is too large or blocked.

Assembly checks SHA-256 manifests, takes the latest snapshot for each window,
and resolves overlaps using the newest retrieval. Every source snapshot remains
on disk. `history.coverage.json` records source hashes, overlap counts, cleaning
results and per-season counts; rejected records remain separate. It assembles
NBA cache manifests, not arbitrary imported files. External datasets can still
use M1 preprocessing and then feed their cleaned Parquet directly to M2 commands.
Never mix the explicitly synthetic test fixtures into production history.

### Feature definitions and availability

```bash
python -m nba_player_prop_prediction.milestone2 features data/processed/history.parquet
```

`features.parquet` contains identities and only pregame predictor columns; labels
remain in cleaned history. `features.json` records version, columns and missingness.
Features include:

- Prior 3/5/10/20-appearance means of minutes and nine box-score statistics;
  minutes medians and sample standard deviations (ddof=1).
- Previous minutes, expanding median/min/max/volatility, span-10 exponentially
  weighted minutes, 3-versus-10-game change and volatility-to-mean consistency.
- Number of prior appearances, home/away, days since last appearance, rest days,
  back-to-back appearance indicator, appearances in the last 7/14 days, and
  days since July 1 of the stated season.
- Prior minutes against the opponent; prior ten complete team-game means of
  points/rebounds/assists scored and allowed, plus win fraction.
- Explicitly estimated pace/offensive/defensive ratings, when attempts, offensive
  rebounds, turnovers and free throws are present. Possessions use the two-team
  average of FGA − OREB + TOV + 0.44×FTA; pace adjusts to 48 minutes using summed
  player minutes. These are box-score proxies, not official NBA advanced ratings.

Player histories continue across trades and season transitions. Team features
follow the game-specific team; rookies retain missing history and a zero history
count. Missing values never become zero except actual counts of prior appearances.
All same-day results are excluded using a strict prior-date lookup, including
multiple same-day observations. Feature order is deterministic after sorting.
Team aggregates require both teams and plausible matching 240+25×OT total minutes;
this is a completeness check, not independent validation against an official schedule.
Partial team inputs produce missing context. Schedule workload represents observed
player appearances, not all scheduled team games; missing source games can undercount it.

No target minutes, box score, result, or postgame starter status is a predictor.
Historical schedule context reconstructed from logs is conditional on appearance.
No historical injury or starter feed is available. `availability_at` accepts
explicit trusted source records with player/game IDs, status and timezone-aware
`available_at`; only records strictly before cutoff qualify. Statuses are returned
as advisory metadata and are not used by the initial model. Teammate/lineup
availability modeling remains deferred until a trustworthy timestamped feed exists.

### Training, evaluation and prediction

```bash
python -m nba_player_prop_prediction.milestone2 train data/processed/history.parquet \
  --config configs/minutes.json --output artifacts/minutes/my-run
python -m nba_player_prop_prediction.milestone2 evaluate artifacts/minutes/my-run
python -m nba_player_prop_prediction.milestone2 predict data/processed/history.parquet \
  --model artifacts/minutes/my-run --context configs/prediction_context.example.json \
  --player-id 2544 --cutoff '2026-11-10T12:00:00-05:00'
```

The prediction context file is an illustrative schedule, not a verified real fixture.
Replace it with a trustworthy scheduled game before operational use. Inference takes
only local history, rejects models selected using labels at/after the cutoff,
requires timezone-aware cutoff before tipoff, and excludes the entire cutoff day.
NBA game dates are interpreted as Eastern calendar dates. It returns model version,
prediction timestamp/cutoff, nonnegative predicted minutes, uncertainty and warnings.
No silent live requests are made.

Default splits train before July 2024, validate from July 2024 through June 2025,
and test from July 2025 onward. Default guards require 5,000 training rows, 1,000
validation and test rows each, 60 training dates, 14 dates per evaluation partition
and 50 players per partition. Insufficient coverage writes a skipped report and
exits 2 without fitting. Counts and boundaries are configurable in `configs/minutes.json`.
Use a new output directory for each run to preserve existing evaluation artifacts.

Baseline forecasts are prior five-game mean, ten-game mean and historical median;
all use the training-only global median for players without prior minutes. CatBoost
uses a fixed seed, CPU regression, missing numeric values, string categories and
chronological input ordering. See the official
[CatBoostRegressor documentation](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor).
Validation MAE selects the stopping iteration; the final test is not used for
fitting, early stopping or interval estimation. No hyperparameter search has been
performed. Compare any future candidates on validation only, never repeatedly tune
the final test. Rolling evaluation uses previously completed held-out games as
history, which reflects sequential daily predictions rather than a season-ahead forecast.

All four approaches use identical held-out records. Reports include exact periods,
players and sample sizes, MAE/RMSE/median error, signed bias and quantiles, actual
minutes buckets, prior-minute role categories, limited history, long rest and
back-to-backs. Native `.cbm`, metadata, feature importance, prediction Parquet,
JSON/Markdown reports and error plots are saved in the run directory.

Intervals use a validation absolute-residual quantile (nominal 90%), with empirical
coverage and width reported on test games. They are a simple residual-interval
investigation, not quantile regression or guaranteed conditional coverage. Validation
is also used for early stopping; serial dependence and selection mean conformal
coverage guarantees do not apply. They do not model inactivity. All performance is
conditional on appearing in the historical player-game dataset.

### Tests and next milestone

```bash
PYTHONPATH=src python -m pytest
ruff check .
ruff format --check .
```

Offline tests cover rolling statistics, rest/workload, trades, season transitions,
missing history, same-day exclusion, target/future mutations, team aggregation,
timestamped availability, data guards, chronological evaluation, serialization and
prediction. Test model fitting uses synthetic data only inside temporary directories;
it is never a production performance report.

Milestone 3 can consume versioned features and serialized minutes predictions.
When fitting player-stat models, training-row minutes predictions must themselves
be generated with chronological out-of-fold models; using in-sample fitted minutes
would leak learned labels. Timestamped availability, verified historical publication
times and more rigorous interval calibration remain future requirements. XGBoostLSS,
NGBoost and points/rebounds/assists/threes distribution models remain unimplemented.

### Milestone 2 verified results (2026-11-04)

Real NBA Stats collection succeeded for all four requested regular seasons:

| Season | Player-games | Players | Games |
|---|---:|---:|---:|
| 2022–23 | 25,894 | 539 | 1,230 |
| 2023–24 | 26,401 | 572 | 1,230 |
| 2024–25 | 26,306 | 569 | 1,230 |
| 2025–26 | 26,651 | 582 | 1,230 |

Total: 105,252 observations, 877 distinct players, 4,920 games, from 2022-10-18
through 2026-04-12. Assembly replaced 89 overlapping sample rows with newer
season snapshots. Cleaning rejected no records; the assembled source columns
had no missing values. Feature missingness remains intentional for cold starts
and insufficient histories. Nine zero-activity observations have unknown
participation and are excluded from appearance-conditional training/evaluation.
No 2026–27 data was requested in this verification.

The run `artifacts/minutes/m2-v1` used:

- Training: 52,291 appearances, 661 players, 2022-10-18 through 2024-04-14.
- Validation: 26,303 appearances, 569 players, 2024-10-22 through 2025-04-13.
- Final test: 26,649 appearances, 582 players, 2025-10-21 through 2026-04-12.

| Model | Validation MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Prior five-game mean | 5.172 | 5.141 | 6.794 |
| Prior ten-game mean | 5.235 | 5.220 | 6.878 |
| Historical median | 6.149 | 6.500 | 8.491 |
| CatBoost | 4.933 | 4.913 | 6.376 |

All errors are minutes and all models use identical evaluation records. CatBoost
improved test MAE by approximately 4.43% versus the five-game baseline in this
single chronological evaluation. This is not a guarantee of future improvement.
CatBoost test median absolute error was 3.962 minutes; signed prediction bias was
−0.321 minutes. The nominal 90% residual interval achieved 89.76% empirical test
coverage with a mean width of 20.18 minutes. Intervals remain broad and are not
validated as reliable for every individual player or subgroup.

The strongest feature importances were exponentially weighted historical minutes,
previous minutes and the prior three-game mean. Full subgroup metrics, feature
importance, predictions and plots are in the local run artifacts. Data completeness
was not independently reconciled against a schedule; pregame publication times,
injuries and lineup feeds remain unavailable. No model hyperparameters were changed
in response to the final test metrics.

Verification completed with 42 offline tests passing (including all 32 Milestone 1
cases), Ruff lint/formatting, `pip check`, package imports and `git diff --check`.
CatBoost 1.2.10 ran on Python 3.14.5. Feature output contains 105,252 rows and
93 columns (89 model inputs plus four identity/date/season metadata columns).
Model save/load and the illustrative prediction CLI worked; re-evaluating saved
predictions exactly reproduced the test report. No generated data or models are
tracked by Git. Nonfatal sandbox PyArrow CPU-query warnings did not prevent exports.

## Milestone 3: probabilistic player statistics

Implemented in `probabilistic/`: chronological out-of-fold minutes, statistical
baselines, count-distribution adapters, dedicated probability calibration,
alternate-line inference, traceable local experiments and offline tests. Targets
are points, total rebounds, assists and made threes. All forecasts are conditional
on appearing; no appearance-risk model or sportsbook integration is implemented.

### Compatible environment and reproducible commands

Use Python 3.12 for this milestone. XGBoostLSS 0.6.1 pins pandas below 2.3 and
other dependencies; an isolated environment avoids changing the M2 runtime.
PyTorch is an XGBoostLSS automatic-differentiation dependency, not a neural-network
or TFT experiment. All training uses CPU with two threads where supported.

```bash
python3.12 -m venv .venv-m3
source .venv-m3/bin/activate
python -m pip install '.[dev,plots,minutes,probabilistic]'
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$PWD/artifacts/matplotlib"

python -m nba_player_prop_prediction.milestone3 oof \
  --output artifacts/probabilistic/oof-new
python -m nba_player_prop_prediction.milestone3 prepare \
  --oof artifacts/probabilistic/oof-new/minutes.parquet \
  --output data/processed/stats-new.parquet

# Small first-target smoke experiment:
python -m nba_player_prop_prediction.milestone3 train \
  --data data/processed/stats-new.parquet --output artifacts/probabilistic/smoke-new \
  --targets points --models xgblss_Poisson --rounds 10 --train-limit 1000

# All four targets and all candidates; bounded CPU experiment:
python -m nba_player_prop_prediction.milestone3 train \
  --data data/processed/stats-new.parquet --output artifacts/probabilistic/comparison-new \
  --rounds 60 --train-limit 6000
python -m nba_player_prop_prediction.milestone3 report \
  artifacts/probabilistic/comparison-new --data data/processed/stats-new.parquet
python -m nba_player_prop_prediction.milestone3 verify \
  artifacts/probabilistic/comparison-new/points/catboost

python -m pytest
ruff check .
ruff format --check .
python -m pip check
```

Use `--train-limit 0` in a **new run directory** to use all eligible training and
validation rows. `--targets` and `--models` restrict work; `--rounds` controls boosting
iterations. The default experiment is a resource-bounded comparison, not exhaustive
tuning. Repeating an identical configuration resumes completed candidates; failed
candidates are retried and failures recorded explicitly. Different configurations
cannot overwrite an existing run. OOF generation requires a new output directory.

### Chronology and minutes provenance

2022–23 is warm-up only. A minutes model fitted on 2022–23 forecasts 2023–24;
an expanded model fitted before July 2024 forecasts 2024–25; the final frozen model
uses only observations before July 2025 to forecast 2025–26. These are expanding
**season-block** forecasts, not daily model refits. Features still update from
strictly earlier game dates. No final-test labels enter minutes training or model
selection. The earlier M2 test results were already inspected; 2025–26 is held out
for M3 fitting/selection, not a claim that nobody has previously analyzed that season.

OOF rows preserve player-game identity, historical prediction-as-of date,
model-freeze date, last training-label date, model hash and actual artifact-generation
time. Historical predictions are retrospective simulations, not predictions recorded
at the original tipoff. Same-day history is excluded. The uncertainty input is prior
ten-game minutes volatility, explicitly a proxy rather than a calibrated predictive
standard deviation. Full integration over a minutes predictive distribution remains
future work. CatBoost direct and minutes-aware variants provide an initial comparison.

Stat-model partitions are fixed before fitting: 2023–24 training; October–December
2024 validation; January–April 2025 calibration; 2025–26 final test. Validation selects
count families by NLL and CatBoost's observation-model dispersion. Calibration does
not use test labels. Minimum coverage is 500 rows and 14 dates in each partition.
Within training/validation, capped experiments choose evenly spaced chronological
rows deterministically. Calibration and test rows are never sampled by that cap.

### Distribution models and baselines

Five baselines cover prior five-/ten-game means, season-to-date mean, historical
stat-per-minute rate times OOF minutes, and a global training mean. They use NB2
observation dispersion estimated only from training residuals (Poisson when the
estimated excess dispersion is zero). A separate training-only empirical threshold
frequency reference is saved in `empirical_base_rates.json`.

[XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/distributions/) fits Poisson,
Negative Binomial, zero-inflated Poisson and zero-inflated Negative Binomial. Models
return distribution parameters. Torch NB parameters are converted to NB2 mean and
dispersion with the correct success-probability convention. Marginal variance and
zero frequency are reported as diagnostics, while validation NLL—not assumptions
about marginal overdispersion—selects candidates. Zero-inflated models may fail to
improve validation and are not automatically preferred.

CatBoost standard regression supplies a nonnegative mean; a validation-selected
NB2 observation model supplies count probabilities. This is explicitly a plug-in
count distribution around a point forecast, not a native CatBoost probabilistic
likelihood. A separate MultiQuantile model predicts 5th/50th/95th percentiles;
sorting prevents crossing and flooring enforces nonnegative quantiles. Its interval
coverage is measured, but quantiles are **not converted into probabilities**.
RMSEWithUncertainty's Gaussian likelihood was not used because it assigns negative
support and does not represent zero-valued counts naturally.

[NGBoost's supported Poisson distribution](https://github.com/stanfordmlgroup/ngboost/blob/master/examples/poisson.py)
provides proper nonnegative count forecasts. No Normal/LogNormal approximation is
forced onto counts. Poisson's equidispersion remains a limitation assessed through
NLL, coverage and probability calibration against NB alternatives.

Training-only median imputation and ordinal categorical encoding are serialized
with each statistical adapter. Missingness is not replaced using future observations.
Stat target columns and realized minutes are excluded through an explicit feature
allowlist. Historical rates, season means and volatility use strictly earlier games.

### Calibration, scoring and selection accounting

Each candidate has identity, constrained-positive-slope sigmoid and isotonic mappings
fitted on the dedicated calibration period. A **single monotone mapping per target
and model**, pooled across configured lines, preserves decreasing probabilities as
thresholds rise and avoids sparse independent calibrators. Sigmoid is the predefined
default; all three methods are reported without selecting a calibrator from test
outcomes. Under probabilities use the complement of calibrated survival probabilities.

Reports include MAE, RMSE, bias, raw-distribution NLL and 90% interval coverage;
Brier score, binary log loss, ten-bin calibration error, reliability tables/plots;
and confidence selections at 60/70/80/85/90%. Every selection report includes eligible
candidates, selected count, wins, losses, observed hit rate, mean forecast, selection
coverage, unique player-games and unique NBA games. Fixed-seed 300-replicate game-cluster
bootstrap intervals preserve within-game dependence, but do not fully account for
cross-game player or temporal dependence. Multiple alternate lines remain correlated
candidates, not independent bets. No losing predictions are removed.

Calibrated threshold probabilities are distinct from raw-distribution moments,
NLL and intervals; a threshold calibrator does not retroactively calibrate the entire
raw distribution. The framework does not claim guaranteed hit rates or force selections.

### Alternate lines and saved inference

For integer counts: `at_least 20` is P(X≥20), `over 19.5` is identical, and
`under 19.5` is P(X≤19). Integer `over 20` means P(X≥21), excluding the push at 20.
The engine supports configurable finite integer or half-point lines. Forecasts
remain in [0,1] and monotone; certain boundary events stay at zero or one.
These are statistical thresholds, not confirmed available sportsbook markets.

```bash
# Replace PLAYER:GAME with a player_game_id from the saved 2025–26 test data:
python -m nba_player_prop_prediction.milestone3 predict \
  artifacts/probabilistic/comparison-v1/points/catboost \
  --data data/processed/stats-v2.parquet --player-game-id 'PLAYER:GAME' \
  --line 19.5 --event over
```

The reusable `probabilistic.prediction.predict_game` interface also accepts local
history, a caller-supplied scheduled game context, timezone-aware cutoff, a frozen
minutes-model directory, line and event. It excludes all history on/after the cutoff
date, validates model/calibrator availability, and makes no live requests. Historical
publication timestamps and reliable injury/starting-lineup inputs remain unavailable.
Probabilities must not be interpreted as unconditional betting win probabilities;
DNP outcomes and market-specific participation/void rules require separate treatment.

Local artifacts include model bundles, schemas, parameters, software versions,
dataset/source hashes, chronological coverage, predictions and metrics. Pickles must
only be loaded from trusted local runs; checksums detect corruption, not hostile
artifacts. `verify` recomputes distribution and probability metrics from saved data.

Milestone 4 can build a cautious high-confidence selection engine on these calibrated
threshold APIs and audit tables. Remaining work includes broader/full-row tuning,
rolling-origin robustness checks, appearance modeling, timestamped availability,
more rigorous uncertainty propagation and calibration monitoring. PRA dependence,
Bayesian/TFT experiments, sportsbook integrations and the dashboard remain deferred.

### Milestone 3 final audit (2026-11-05)

See [the complete audit](docs/milestone3-audit.md) for all candidate metrics,
calibration buckets, chronological coverage and the verified 93.41% result.
Validation NLL selected XGBoostLSS ZINB for points and CatBoost with an NB2
observation model for rebounds, assists and threes. These are criterion-specific
choices; no candidate dominates every held-out metric.

The threes 90%+ result was 1,247 hits / 1,335 selections (88 misses), across
1,335 unique player-games and 799 NBA games. No duplicate or repeated selected
lines inflated that result. The 2,000-replicate game-cluster 95% interval is
92.08%–94.68%. Generated lines are not verified sportsbook offers or bets.

Final verification: 59 offline tests passed, Ruff lint/format checks passed,
both environment dependency checks passed, and all 52 model artifacts loaded.
Saved distribution and probability metrics reproduced without model retraining.
Only 6,000 training and 6,000 validation rows per candidate were fitted in this
bounded experiment; calibration/test used 15,868/26,649 rows.

Reproduce the saved-data audit with:

```bash
PYTHONPATH=src .venv-m3/bin/python scripts/audit_milestone3.py
```

This generates `artifacts/probabilistic/comparison-v1/final_audit.json`.
Older milestone roadmap statements describe their status at completion; M3 now
implements the previously planned XGBoostLSS/NGBoost count models. Full-data
tuning, appearance forecasting and the dashboard remain future work.

## Milestone 4 — selection, backtesting and paper tracking

`selection/` integrates the validation-selected M3 models: XGBoostLSS ZINB for
points and CatBoost with NB2 observation distributions for rebounds, assists and
threes. No ensemble or uncertainty-adjusted probability penalty is claimed to be
validated. Rankings use calibrated probability directly. Predictive variance and
historical minutes volatility are not probability-estimation uncertainty.

Policies default to 85%, at most 100 selections per date, ten per game, one per
player-game and 500 calibration observations. These defaults are engineering
settings, **not optimized or established best policies**. Unknown participation
remains unknown; default output is explicitly conditional on appearing. Optional
policies require confirmed availability/markets, minimum history, reject minutes
restrictions, back-to-backs or large recent minutes decreases. Missing required
risk evidence causes abstention. Policies never replace absent injury information
with an assumption of health. Role/lineup feeds remain unavailable.

The engine supports arbitrary valid probability cutoffs, including 60/70/75/80/85/
90/95%, and configured integer or half-point lines/events. It can return zero
selections. All candidates retain selection/exclusion reasons and data warnings.
Deduplication applies across statistical categories for the same player-game.
Daily limits reset per date; game/player limits apply to their identities.

### Commands

Use the M3 environment and run from the repository root:

```bash
source .venv-m3/bin/activate
export PYTHONPATH="$PWD/src"

# Offline exploratory backtest; choose a NEW output directory:
python -m nba_player_prop_prediction.milestone4 backtest --output artifacts/m4/new-backtest

# Illustrative schedule only: replace contexts with a verified upcoming schedule.
# Dry-run writes candidate reports but does not record forward predictions.
python -m nba_player_prop_prediction.milestone4 generate \
  --contexts configs/m4/contexts.example.json --date 2026-11-10 \
  --cutoff '2026-11-05T12:00:00-06:00' --policy configs/m4/default.json \
  --output artifacts/m4/example.parquet --dry-run

python -m nba_player_prop_prediction.milestone4 rank artifacts/m4/example.parquet \
  --policy configs/m4/conservative.json --output artifacts/m4/conservative.parquet

# Real forward paper workflow: remove --dry-run only before scheduled tipoff.
# All generated forecasts, including exclusions, are stored in the ledger.
python -m nba_player_prop_prediction.milestone4 grade \
  --ledger artifacts/predictions.sqlite --outcomes /path/to/verified-outcomes.json
python -m nba_player_prop_prediction.milestone4 report \
  --ledger artifacts/predictions.sqlite --output artifacts/m4/forward-report.json

python -m pytest
ruff check .
ruff format --check .
```

Optional `--availability`, `--markets` and `--lines` JSON inputs supply timestamped
trusted availability, normalized market snapshots, and target-specific lists of
`{"line":19.5,"event":"over"}` respectively. The example schedule is not a verified
NBA fixture and must not be recorded as one. `generate` checks game dates and cutoff
before tipoff; only strictly earlier historical dates enter features.

### Markets

The optional `selection.markets.MarketProvider` interface and `OddsAPI` adapter
support The Odds API's event endpoint. Official documentation reviewed for this
milestone lists [500 free credits](https://the-odds-api.com/) and
[event-level player props](https://the-odds-api.com/liveapi/guides/v4/).
Historical odds are [not available on the free plan](https://the-odds-api.com/liveapi/guides/v4/api-error-codes.html).
Coverage varies by bookmaker/market and cannot be presumed.

```bash
# Set ODDS_API_KEY locally; never commit it.
python -m nba_player_prop_prediction.milestone4 markets --event-id PROVIDER_EVENT_ID
```

Requests have a 20-second timeout, 1.5-second pacing, a 60-second local cache and
immutable timestamped snapshots. HTTP errors, missing credentials and invalid
responses return unavailable; there is no unbounded retry loop. Provider snapshots
preserve bookmaker, market, odds and line information. Explicit NBA/player ID
reconciliation is required before using normalized market rows; fuzzy player-name
matching is deliberately not automatic. Normalized markets must exactly match
player/game/target/line/event, have source/bookmaker/market identifiers, and a snapshot
no later than cutoff and no more than five minutes old. Odds never influence ranking.
No live API call was verified because no credential was supplied. Offline adapter
and cache behavior are tested. No historical market snapshots exist in this repo,
so historical market-only results contain **zero selections**, not fabricated offers.

### Ledger and grading

SQLite stores immutable forecast JSON with a stable SHA-256 identity based on
player/game/target/event/line/model/cutoff/policy. Duplicate identical inserts are
idempotent; conflicting payloads are rejected. Forecasts must be recorded before
tipoff using the actual system clock. Historical backtests cannot enter this forward
ledger. Grades are separate immutable records; database triggers reject updates and
deletes. Pending forecasts remain visible. This is a local audit trail, not a
cryptographically trusted external timestamp service.

Outcome input is a list of objects with `key` (prediction ID), `actual`,
`participation`, `completed_at` (timezone-aware completed-game time) and `evidence`
(source reference). Hypothetical count events are graded hit/miss; equality at an
integer over/under line is a documented push/void. Missing outcomes or DNP/unknown
participation are unable-to-grade, never silently losses or voids. Market-backed
records require explicit `market_settlement` (`hit`, `miss`, `void`, or
`unable_to_grade`) and evidence for the actual provider's rules. No generic rule
claims to settle every sportsbook market. Corrections require explicit reconciliation
outside this immutable ledger; automatic overwrite is prohibited.

JSON/Markdown/Parquet/CSV outputs retain exclusions, warnings and conditional status.
Backtests include daily/weekly/monthly and target/model/player/opponent/game summaries,
confidence buckets, counts, coverage and game-cluster uncertainty. Forward reports
separate hypothetical and market-backed records and exclude pending/void/ungradable
records from hit-rate denominators while retaining their status counts.

### Exploratory results and limitations

The 2025–26 season was already inspected in M2/M3. Every M4 backtest is therefore
**exploratory reused-test analysis**, not a new untouched evaluation or live result.
Policies were not tuned on these new results. No current-season results were fetched.
The previous threes result reproduces: 1,247 hits / 1,335 candidates, 88 misses,
1,335 unique player-games and 799 games. One-per-player-game leaves it unchanged;
one-per-game produces 757 hits / 799 selections (94.74%). Market-only and confirmed-
availability conservative policies abstain completely because evidence is missing.

Across all four targets, the default 85% policy selects 7,720 candidates: 7,197 hits,
523 misses, 93.23% observed versus 90.69% mean estimated probability, covering 1.45%
of 532,980 generated statistical candidates. Its game-cluster 95% interval is
92.71%–93.79%. This does not establish performance on actual offered lines. The
one-per-game 90% policy yields 1,169/1,193 (97.99%) on this reused dataset; this is
an exploratory finding, not an expected future hit rate or recommendation to adopt
that policy. Full comparisons are in `artifacts/m4/backtest-v1/comparison.json`.

Milestone 5 can consume stable candidate, selection, ledger and report schemas.
Before interpreting dashboard results as validated live performance, collect forward
paper forecasts, timestamped schedules/availability and confirmed market snapshots,
monitor calibration drift, and establish a fresh evaluation period. No frontend,
automatic bet placement, payout optimization or new model training was added.

Milestone 4 verification: 75 offline tests, including all 59 previous tests;
real-data outcome reconciliation for all 532,980 candidates; identical default
selection IDs on rerun; all four selected target models loaded; 20 dry-run
alternate-line probabilities checked for bounds/ordering and distribution agreement.
No forward NBA forecasts were recorded and no live provider request was verified.
See [the policy comparison](docs/milestone4-results.md) for the full exploratory table.
The calibration sample minimum currently checks total target calibration rows,
not a per-player or local probability-neighborhood effective sample size.
