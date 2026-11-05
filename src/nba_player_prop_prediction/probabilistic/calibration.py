"""Pooled monotone calibrators preserve ordering across thresholds."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self, method="sigmoid"):
        if method not in ["identity", "sigmoid", "isotonic"]:
            raise ValueError("Unknown calibration method")
        self.method = method

    def fit(self, p, y, dates, test_start, min_samples=200):
        import pandas as pd

        if pd.to_datetime(dates).max() >= pd.Timestamp(test_start):
            raise ValueError("Calibration overlaps test")
        p, y = np.asarray(p).ravel(), np.asarray(y).ravel()
        if len(p) < min_samples or len(np.unique(y)) != 2:
            raise ValueError("Insufficient calibration outcomes")
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        elif self.method == "sigmoid":
            x = logit(np.clip(p, 1e-6, 1 - 1e-6))

            def loss(theta):
                z = theta[0] * x + theta[1]
                return np.mean(np.logaddexp(0, z) - y * z)

            result = minimize(loss, [1.0, 0.0], bounds=[(0, 20), (-20, 20)], method="L-BFGS-B")
            if not result.success:
                raise ValueError(f"Calibration failed: {result.message}")
            self.a, self.b = result.x
        return self

    def predict(self, p):
        p = np.asarray(p)
        if self.method == "identity":
            return p.copy()
        if self.method == "isotonic":
            answer = self.model.predict(p.ravel()).reshape(p.shape)
        else:
            answer = expit(self.a * logit(np.clip(p, 1e-15, 1 - 1e-15)) + self.b)
        return np.where(p == 0, 0.0, np.where(p == 1, 1.0, answer))
