"""CPU model adapters. Every probability distribution has integer nonnegative support."""

import importlib

import numpy as np
from catboost import CatBoostRegressor
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from nba_player_prop_prediction.features.pregame import CATEGORICAL
from nba_player_prop_prediction.probabilistic.distributions import CountDistribution


class Design:
    def fit(self, frame, columns):
        self.columns = columns
        self.cats = [c for c in columns if c in CATEGORICAL]
        self.nums = [c for c in columns if c not in self.cats]
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder.fit(frame[self.cats].astype(str))
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer.fit(frame[self.nums].astype(float))
        return self

    def transform(self, frame):
        if not set(self.columns).issubset(frame):
            raise ValueError("Feature schema mismatch")
        return np.column_stack(
            [
                self.imputer.transform(frame[self.nums].astype(float)),
                self.encoder.transform(frame[self.cats].astype(str)),
            ]
        )


class Candidate:
    def __init__(self, name, target, columns, rounds=60):
        self.name, self.target, self.columns, self.rounds = name, target, columns, rounds

    def fit(self, train, validation):
        y, vy = (
            train[f"label_{self.target}"].to_numpy(),
            validation[f"label_{self.target}"].to_numpy(),
        )
        if any(not np.isfinite(v).all() or (v < 0).any() or (v % 1 != 0).any() for v in [y, vy]):
            raise ValueError("Targets must be finite nonnegative integer counts")
        self.fallback = float(y.mean())
        self.alpha = 0.0
        if self.name.startswith("baseline_"):
            dist = self.predict(train)
            self.alpha = max(
                0.0, float(np.sum((y - dist.mean) ** 2 - dist.mean) / np.sum(dist.mean**2))
            )
            return self
        self.design = Design().fit(train, self.columns)
        x, vx = self.design.transform(train), self.design.transform(validation)
        if self.name.startswith("xgblss_"):
            import torch
            import xgboost as xgb
            from xgboostlss.model import XGBoostLSS

            torch.set_num_threads(2)
            family = self.name.removeprefix("xgblss_")
            cls = getattr(importlib.import_module(f"xgboostlss.distributions.{family}"), family)
            kwargs = dict(stabilization="L2", initialize=True)
            if family in ["NegativeBinomial", "ZINB"]:
                kwargs["response_fn_total_count"] = "softplus"
            else:
                kwargs["response_fn"] = "softplus"
            self.model = XGBoostLSS(cls(**kwargs))
            self.model.train(
                {
                    "eta": 0.05,
                    "max_depth": 3,
                    "subsample": 1.0,
                    "seed": 42,
                    "nthread": 2,
                    "tree_method": "hist",
                },
                xgb.DMatrix(x, label=y),
                num_boost_round=self.rounds,
                evals=[
                    (xgb.DMatrix(x, label=y), "train"),
                    (xgb.DMatrix(vx, label=vy), "validation"),
                ],
                verbose_eval=False,
            )
        elif self.name == "ngboost":
            from ngboost import NGBRegressor
            from ngboost.distns import Poisson
            from sklearn.tree import DecisionTreeRegressor

            self.model = NGBRegressor(
                Dist=Poisson,
                Base=DecisionTreeRegressor(max_depth=3),
                n_estimators=self.rounds,
                learning_rate=0.05,
                random_state=42,
                verbose=False,
            )
            self.model.fit(x, y)
        else:
            loss = (
                "MultiQuantile:alpha=0.05,0.5,0.95" if self.name == "catboost_quantile" else "RMSE"
            )
            self.model = CatBoostRegressor(
                iterations=self.rounds,
                depth=5,
                learning_rate=0.06,
                loss_function=loss,
                thread_count=2,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                has_time=True,
            )
            self.model.fit(x, y)
            if self.name != "catboost_quantile":
                mu = np.maximum(1e-6, self.model.predict(vx))
                # Dispersion is a validation-selected NB2 observation model around CatBoost's mean.
                fit = minimize_scalar(
                    lambda a: -CountDistribution(mu, a).logpmf(vy).mean(),
                    bounds=(1e-6, 3),
                    method="bounded",
                )
                self.alpha = float(fit.x)
        return self

    def predict(self, frame):
        if self.name.startswith("baseline_"):
            kind = self.name.removeprefix("baseline_")
            if kind in ["5", "10"]:
                mu = frame[f"prior_{kind}_{self.target}_mean"]
            elif kind == "season":
                mu = frame[f"season_mean_{self.target}"]
            elif kind == "rate":
                mu = frame[f"rate_{self.target}"] * frame.predicted_minutes
            else:
                mu = np.full(len(frame), self.fallback)
            mu = np.nan_to_num(np.asarray(mu, dtype=float), nan=self.fallback)
            return CountDistribution(np.maximum(mu, 1e-6), self.alpha)
        x = self.design.transform(frame)
        if self.name.startswith("xgblss_"):
            import xgboost as xgb

            params = self.model.predict(xgb.DMatrix(x), pred_type="parameters")
            gate = params["gate"].to_numpy() if "gate" in params else 0.0
            if "rate" in params:
                return CountDistribution(params.rate.to_numpy(), 0.0, gate)
            n, p = params.total_count.to_numpy(), params.probs.to_numpy()
            return CountDistribution(n * p / (1 - p), 1 / n, gate)
        if self.name == "ngboost":
            dist = self.model.pred_dist(x)
            return CountDistribution(dist.params["mu"])
        if self.name == "catboost_quantile":
            return np.sort(np.maximum(0, self.model.predict(x)), axis=1)
        return CountDistribution(np.maximum(1e-6, self.model.predict(x)), self.alpha)
