# Milestone 4 exploratory backtest

Reused 2025–26 test data; not forward performance or actual sportsbook bets.

| Policy | Selected | Hits | Misses | Hit rate | Mean forecast | 95% game-cluster interval |
|---|---:|---:|---:|---:|---:|---|
| default85 | 7720 | 7197 | 523 | 93.23% | 90.69% | 92.71%–93.79% |
| unrestricted90 | 6241 | 5949 | 292 | 95.32% | 93.00% | 94.75%–95.90% |
| player_game90 | 4209 | 4014 | 195 | 95.37% | 93.25% | 94.77%–96.01% |
| one_game90 | 1193 | 1169 | 24 | 97.99% | 95.23% | 97.15%–98.74% |
| market_only90 | 0 | 0 | 0 | — | — | — |
| conservative90 | 0 | 0 | 0 | — | — | — |
| threes_unrestricted90 | 1335 | 1247 | 88 | 93.41% | 91.85% | 92.02%–94.70% |
| threes_player_game90 | 1335 | 1247 | 88 | 93.41% | 91.85% | 92.02%–94.70% |
| threes_one_game90 | 799 | 757 | 42 | 94.74% | 92.27% | 93.12%–96.12% |
| threes_market_only90 | 0 | 0 | 0 | — | — | — |
| threes_conservative90 | 0 | 0 | 0 | — | — | — |

Intervals use 1,000 fixed-seed NBA-game cluster bootstrap draws. Cross-game dependence and policy-selection uncertainty are not fully covered. Zero selections reflect absent evidence, not evidence of perfect accuracy. No new model was trained.