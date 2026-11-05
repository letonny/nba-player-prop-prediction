"""Discrete distributions and exact integer-event semantics."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, poisson

TARGETS = {
    "points": [10, 15, 20, 25, 30, 35],
    "rebounds": [4, 6, 8, 10, 12],
    "assists": [3, 5, 7, 10],
    "threes_made": [1, 2, 3, 4, 5],
}


@dataclass
class CountDistribution:
    mean: np.ndarray
    dispersion: np.ndarray | float = 0.0
    gate: np.ndarray | float = 0.0

    def __post_init__(self):
        self.mean, self.dispersion, self.gate = np.broadcast_arrays(
            np.atleast_1d(self.mean).astype(float), self.dispersion, self.gate
        )
        if (
            not all(np.isfinite(v).all() for v in [self.mean, self.dispersion, self.gate])
            or (self.mean < 0).any()
            or (self.dispersion < 0).any()
            or ((self.gate < 0) | (self.gate >= 1)).any()
        ):
            raise ValueError("Invalid count-distribution parameters")

    def base(self):
        # alpha=0 is Poisson; NB2 variance = mu + alpha*mu**2.
        n = 1 / np.maximum(self.dispersion, 1e-12)
        return nbinom(n, n / (n + self.mean))

    def cdf(self, k):
        value = np.where(self.dispersion > 0, self.base().cdf(k), poisson.cdf(k, self.mean))
        return np.where(np.asarray(k) < 0, 0.0, self.gate + (1 - self.gate) * value)

    def probability(self, line: float, event: str = "at_least"):
        if not np.isfinite(line):
            raise ValueError("Threshold must be finite")
        if event == "at_least":
            return np.clip(1 - self.cdf(np.ceil(line) - 1), 0, 1)
        if event == "over":
            return np.clip(1 - self.cdf(np.floor(line)), 0, 1)
        if event == "under":
            return np.clip(self.cdf(np.ceil(line) - 1), 0, 1)
        raise ValueError("event must be at_least, over or under")

    def logpmf(self, y):
        logp = np.where(self.dispersion > 0, self.base().logpmf(y), poisson.logpmf(y, self.mean))
        return np.where(
            np.asarray(y) == 0,
            np.logaddexp(np.log(np.maximum(self.gate, 1e-300)), np.log1p(-self.gate) + logp),
            np.log1p(-self.gate) + logp,
        )

    def quantile(self, q):
        adjusted = np.clip((q - self.gate) / (1 - self.gate), 0, 1 - 1e-12)
        answer = np.where(
            self.dispersion > 0, self.base().ppf(adjusted), poisson.ppf(adjusted, self.mean)
        )
        return np.maximum(0, answer)

    @property
    def expected(self):
        return (1 - self.gate) * self.mean

    @property
    def variance(self):
        return (1 - self.gate) * (self.mean + self.dispersion * self.mean**2) + self.gate * (
            1 - self.gate
        ) * self.mean**2
