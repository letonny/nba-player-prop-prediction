"""Environment configuration. Paths are relative to the working directory."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    timeout: float = 30
    retries: int = 2
    pacing: float = 1.5
    backoff: float = 2

    def __post_init__(self):
        import math

        if (
            not all(math.isfinite(x) for x in (self.timeout, self.pacing, self.backoff))
            or self.timeout <= 0
            or self.retries < 0
            or self.pacing < 0
            or self.backoff < 0
        ):
            raise ValueError("Invalid network settings")

    @classmethod
    def from_env(cls):
        load_dotenv(override=False)
        return cls(
            Path(os.getenv("NBA_DATA_DIR", "data")),
            Path(os.getenv("NBA_ARTIFACTS_DIR", "artifacts")),
            float(os.getenv("NBA_TIMEOUT", "30")),
            int(os.getenv("NBA_RETRIES", "2")),
            float(os.getenv("NBA_PACING", "1.5")),
            float(os.getenv("NBA_BACKOFF", "2")),
        )
