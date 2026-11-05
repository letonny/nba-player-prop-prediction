# Milestone 3 final audit — 2026-11-05

All results are historical and conditional on appearance. No sportsbook availability or profit is established.

## Model comparison

Probability scores below use the predefined pooled sigmoid calibrator. NLL and interval coverage refer to raw distributions. Winners use validation NLL, not test scores.

| Target | Candidate | MAE | RMSE | Brier | Log loss | ECE | NLL | 90% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| points | baseline_5 | 4.815 | 6.326 | 0.0837 | 0.2787 | 0.0219 | 3.461 | 82.69% |
| points | baseline_10 | 4.690 | 6.156 | 0.0813 | 0.2696 | 0.0205 | 3.332 | 82.80% |
| points | baseline_season | 4.741 | 6.256 | 0.0830 | 0.2764 | 0.0253 | 3.542 | 82.25% |
| points | baseline_rate | 4.659 | 6.110 | 0.0801 | 0.2604 | 0.0129 | 3.275 | 83.10% |
| points | baseline_global | 6.926 | 8.650 | 0.1177 | 0.3710 | 0.0062 | 3.411 | 84.80% |
| points | xgblss_Poisson | 5.520 | 7.255 | 0.0954 | 0.3034 | 0.0258 | 4.344 | 56.45% |
| points | xgblss_NegativeBinomial | 4.578 | 6.010 | 0.0789 | 0.2596 | 0.0120 | 3.134 | 96.83% |
| points | xgblss_ZIPoisson | 5.463 | 7.115 | 0.0968 | 0.3081 | 0.0290 | 3.842 | 82.85% |
| points | xgblss_ZINB | 4.563 | 5.999 | 0.0783 | 0.2548 | 0.0096 | 3.065 | 96.90% |
| points | catboost | 4.619 | 6.001 | 0.0788 | 0.2568 | 0.0137 | 3.128 | 92.34% |
| points | catboost_direct | 4.610 | 5.992 | 0.0787 | 0.2566 | 0.0143 | 3.129 | 92.27% |
| points | ngboost | 4.574 | 6.001 | 0.0782 | 0.2535 | 0.0149 | 3.729 | 66.25% |
| rebounds | baseline_5 | 1.995 | 2.641 | 0.1023 | 0.3308 | 0.0166 | 2.331 | 91.38% |
| rebounds | baseline_10 | 1.943 | 2.563 | 0.0995 | 0.3204 | 0.0153 | 2.269 | 92.32% |
| rebounds | baseline_season | 1.935 | 2.571 | 0.0999 | 0.3254 | 0.0153 | 2.371 | 92.00% |
| rebounds | baseline_rate | 1.895 | 2.488 | 0.0965 | 0.3092 | 0.0137 | 2.221 | 93.29% |
| rebounds | baseline_global | 2.566 | 3.317 | 0.1339 | 0.4190 | 0.0081 | 2.447 | 96.61% |
| rebounds | xgblss_Poisson | 1.992 | 2.653 | 0.1021 | 0.3230 | 0.0166 | 2.299 | 88.38% |
| rebounds | xgblss_NegativeBinomial | 1.886 | 2.488 | 0.0967 | 0.3106 | 0.0116 | 2.235 | 97.97% |
| rebounds | xgblss_ZIPoisson | 1.973 | 2.635 | 0.1017 | 0.3225 | 0.0176 | 2.262 | 92.71% |
| rebounds | xgblss_ZINB | 3.365 | 4.517 | 0.1441 | 0.4459 | 0.1040 | 6.741 | 30.24% |
| rebounds | catboost | 1.893 | 2.486 | 0.0965 | 0.3079 | 0.0124 | 2.183 | 94.22% |
| rebounds | catboost_direct | 1.900 | 2.494 | 0.0967 | 0.3085 | 0.0119 | 2.187 | 94.26% |
| rebounds | ngboost | 1.889 | 2.490 | 0.0967 | 0.3088 | 0.0145 | 2.230 | 89.66% |
| assists | baseline_5 | 1.403 | 1.919 | 0.0837 | 0.2751 | 0.0146 | 1.994 | 93.46% |
| assists | baseline_10 | 1.374 | 1.874 | 0.0818 | 0.2668 | 0.0113 | 1.866 | 94.16% |
| assists | baseline_season | 1.374 | 1.885 | 0.0826 | 0.2716 | 0.0141 | 1.968 | 93.78% |
| assists | baseline_rate | 1.365 | 1.873 | 0.0818 | 0.2633 | 0.0057 | 1.825 | 94.81% |
| assists | baseline_global | 1.973 | 2.561 | 0.1191 | 0.3791 | 0.0055 | 2.073 | 96.65% |
| assists | xgblss_Poisson | 1.401 | 1.929 | 0.0830 | 0.2666 | 0.0125 | 1.832 | 92.93% |
| assists | xgblss_NegativeBinomial | 1.354 | 1.831 | 0.0801 | 0.2594 | 0.0088 | 1.837 | 98.53% |
| assists | xgblss_ZIPoisson | 1.383 | 1.925 | 0.0832 | 0.2706 | 0.0155 | 1.831 | 94.41% |
| assists | xgblss_ZINB | 1.352 | 1.839 | 0.0804 | 0.2595 | 0.0077 | 1.860 | 98.55% |
| assists | catboost | 1.352 | 1.830 | 0.0798 | 0.2574 | 0.0082 | 1.777 | 95.56% |
| assists | catboost_direct | 1.350 | 1.830 | 0.0797 | 0.2571 | 0.0078 | 1.776 | 95.65% |
| assists | ngboost | 1.356 | 1.836 | 0.0801 | 0.2588 | 0.0099 | 1.799 | 93.53% |
| threes_made | baseline_5 | 0.927 | 1.315 | 0.1220 | 0.4150 | 0.0366 | 1.637 | 94.42% |
| threes_made | baseline_10 | 0.901 | 1.268 | 0.1158 | 0.3781 | 0.0195 | 1.433 | 95.66% |
| threes_made | baseline_season | 0.898 | 1.264 | 0.1159 | 0.3907 | 0.0164 | 1.491 | 95.46% |
| threes_made | baseline_rate | 0.886 | 1.241 | 0.1133 | 0.3622 | 0.0108 | 1.362 | 96.07% |
| threes_made | baseline_global | 1.160 | 1.487 | 0.1456 | 0.4487 | 0.0087 | 1.529 | 96.11% |
| threes_made | xgblss_Poisson | 0.908 | 1.243 | 0.1134 | 0.3549 | 0.0099 | 1.331 | 95.94% |
| threes_made | xgblss_NegativeBinomial | 0.893 | 1.237 | 0.1124 | 0.3523 | 0.0074 | 1.338 | 98.19% |
| threes_made | xgblss_ZIPoisson | 0.865 | 1.270 | 0.1441 | 0.5829 | 0.1375 | 2.797 | 89.42% |
| threes_made | xgblss_ZINB | 0.905 | 1.351 | 0.1516 | 0.4775 | 0.1208 | 2.981 | 81.49% |
| threes_made | catboost | 0.900 | 1.232 | 0.1122 | 0.3521 | 0.0087 | 1.316 | 97.02% |
| threes_made | catboost_direct | 0.903 | 1.233 | 0.1125 | 0.3529 | 0.0092 | 1.319 | 96.97% |
| threes_made | ngboost | 0.891 | 1.235 | 0.1122 | 0.3513 | 0.0076 | 1.320 | 95.79% |

## Validation winners

{
  "points": "xgblss_ZINB",
  "rebounds": "catboost",
  "assists": "catboost",
  "threes_made": "catboost"
}

## Calibration buckets

Intervals are 95% percentile intervals from 2,000 NBA-game cluster bootstrap draws. They preserve within-game dependence, not cross-game temporal/player dependence.

| Target | Bucket | N | Expected | Observed | 95% interval |
|---|---|---:|---:|---:|---|
| points | 60%–70% | 4845 | 65.0% | 67.5% | 66.2%–68.8% |
| points | 70%–80% | 5011 | 74.7% | 78.1% | 77.0%–79.2% |
| points | 80%–90% | 4277 | 85.3% | 88.8% | 87.8%–89.8% |
| points | 90%–95% | 1996 | 92.4% | 95.4% | 94.5%–96.3% |
| points | 95%–100% | 343 | 95.5% | 98.3% | 96.8%–99.4% |
| rebounds | 60%–70% | 5498 | 64.8% | 66.8% | 65.6%–68.0% |
| rebounds | 70%–80% | 3904 | 74.6% | 77.7% | 76.4%–79.1% |
| rebounds | 80%–90% | 3297 | 84.9% | 88.8% | 87.7%–89.9% |
| rebounds | 90%–95% | 872 | 92.4% | 95.2% | 93.8%–96.6% |
| rebounds | 95%–100% | 609 | 96.1% | 98.0% | 96.9%–99.0% |
| assists | 60%–70% | 2836 | 65.1% | 65.1% | 63.3%–66.9% |
| assists | 70%–80% | 2944 | 75.2% | 76.6% | 75.1%–78.2% |
| assists | 80%–90% | 2501 | 84.7% | 89.1% | 87.8%–90.2% |
| assists | 90%–95% | 748 | 92.2% | 94.5% | 92.8%–96.1% |
| assists | 95%–100% | 338 | 96.5% | 96.7% | 94.7%–98.5% |
| threes_made | 60%–70% | 6673 | 65.0% | 64.4% | 63.2%–65.5% |
| threes_made | 70%–80% | 6560 | 75.2% | 75.2% | 74.1%–76.2% |
| threes_made | 80%–90% | 5130 | 84.9% | 85.9% | 84.9%–86.8% |
| threes_made | 90%–95% | 1335 | 91.8% | 93.4% | 92.1%–94.7% |
| threes_made | 95%–100% | 0 | — | — | — |

## 93.41% audit

```json
{
  "all": {
    "eligible_candidates": 133245,
    "selected": 1335,
    "wins": 1247,
    "losses": 88,
    "hit_rate": 0.9340823970037453,
    "average_probability": 0.9184533985925227,
    "coverage": 0.010019137678712146,
    "unique_player_games": 1335,
    "unique_games": 799,
    "game_cluster_95_interval": [
      0.920761318776557,
      0.9467776083868178
    ]
  },
  "one_per_player_game": {
    "eligible_candidates": 26649,
    "selected": 1335,
    "wins": 1247,
    "losses": 88,
    "hit_rate": 0.9340823970037453,
    "average_probability": 0.9184533985925227,
    "coverage": 0.05009568839356073,
    "unique_player_games": 1335,
    "unique_games": 799,
    "game_cluster_95_interval": [
      0.920761318776557,
      0.9467776083868178
    ]
  },
  "duplicate_candidates": 0,
  "extra_selected_lines": 0,
  "lines": [
    1,
    2,
    3,
    4,
    5
  ]
}
```

No duplicate candidates and no repeated selected alternate lines for the same player-game. All model candidates used identical thresholds, player-games and observed labels. Thresholds and confidence cutoffs were fixed in code before the comparison. Historical as-of dates are simulations; source publication-time availability is not independently proven.

## Periods

```json
{
  "train": {
    "rows": 6000,
    "players": 542,
    "games": 1230,
    "start": "2023-10-24 00:00:00",
    "end": "2024-04-14 00:00:00"
  },
  "validation": {
    "rows": 6000,
    "players": 501,
    "games": 484,
    "start": "2024-10-22 00:00:00",
    "end": "2024-12-31 00:00:00"
  },
  "calibration": {
    "rows": 15868,
    "players": 546,
    "games": 746,
    "start": "2025-01-01 00:00:00",
    "end": "2025-04-13 00:00:00"
  },
  "test": {
    "rows": 26649,
    "players": 582,
    "games": 1230,
    "start": "2025-10-21 00:00:00",
    "end": "2026-04-12 00:00:00"
  }
}
```

## OOF minutes

```json
{
  "rows": 79358,
  "folds": [
    {
      "start": "2023-07-01",
      "end_exclusive": "2024-07-01",
      "n_train": 25892,
      "n_predict": 26401,
      "trained_through": "2023-04-09 00:00:00",
      "model": "2023-07-01.cbm"
    },
    {
      "start": "2024-07-01",
      "end_exclusive": "2025-07-01",
      "n_train": 52291,
      "n_predict": 26306,
      "trained_through": "2024-04-14 00:00:00",
      "model": "2024-07-01.cbm"
    },
    {
      "start": "2025-07-01",
      "end_exclusive": "2026-07-01",
      "n_train": 78594,
      "n_predict": 26651,
      "trained_through": "2025-04-13 00:00:00",
      "model": "2025-07-01.cbm"
    }
  ]
}
```

## Verification and limitations

59 offline tests passed; Ruff lint and formatting passed; both environment dependency checks passed. All 52 saved model artifacts loaded. Raw distribution and calibrated probability metrics reproduced from saved predictions. Statistical models never include target-game minutes or label columns. Training and validation were capped at 6,000 evenly spaced chronological rows; calibration/test were uncapped. All generated datasets/models are ignored by Git.

Probability calibration does not improve every target. Poisson points intervals under-cover; NB intervals can over-cover. Quantile forecasts are reported only as quantiles, never converted to probability distributions. No validated injury, appearance, lineup, or PRA-dependence model exists. Minutes uncertainty remains a volatility proxy. The 2025–26 season was previously examined for M2, though not used for M3 fitting or selection.

## Milestone 4 plan

1. Freeze the present configuration and add rolling-origin backtests with a genuinely new holdout.
2. Build a selection ledger with one selection per player-game, abstention, coverage limits and cluster-aware monitoring.
3. Reassess calibration by target/line/history and monitor drift; preserve an independent calibration window.
4. Add trustworthy timestamped schedule/availability feeds and separately validate appearance probabilities.
5. Propagate minutes uncertainty and model PRA dependence before composite outcomes; build the dashboard after these data contracts stabilize.

## Quantile-only diagnostics

| Target | Median MAE | 90% quantile interval coverage |
|---|---:|---:|
| points | 4.549 | 88.67% |
| rebounds | 1.874 | 88.51% |
| assists | 1.328 | 82.71% |
| threes_made | 0.860 | 54.50% |

## Calibration method comparison for validation-selected models

| Target | Raw Brier | Sigmoid Brier | Isotonic Brier |
|---|---:|---:|---:|
| points | 0.0808 | 0.0783 | 0.0781 |
| rebounds | 0.0964 | 0.0965 | 0.0963 |
| assists | 0.0797 | 0.0798 | 0.0796 |
| threes_made | 0.1121 | 0.1122 | 0.1122 |
