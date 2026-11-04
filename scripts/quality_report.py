"""Convenience script: python scripts/quality_report.py games.parquet [--plots]."""

import sys

from nba_player_prop_prediction.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "report")
    main()
