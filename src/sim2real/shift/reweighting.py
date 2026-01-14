"""Importance weighting via density-ratio estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ReweightingReport:
    feature: str
    wasserstein_before: float
    wasserstein_after: float
    improvement: float


class DensityRatioReweighter:
    """Logistic-regression density ratio model."""

    def __init__(
        self,
        *,
        continuous_features: Sequence[str],
        categorical_features: Sequence[str],
        max_iter: int = 1000,
    ) -> None:
        self.continuous_features = list(continuous_features)
        self.categorical_features = list(categorical_features)
        transformers = []
        if self.continuous_features:
            transformers.append(("num", StandardScaler(), self.continuous_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features))
        self.model = Pipeline(
            steps=[
                ("pre", ColumnTransformer(transformers)),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=max_iter,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self._fitted = False
        self._n_sim = 0
        self._n_real = 0

    def fit(self, df: pd.DataFrame) -> "DensityRatioReweighter":
        X = df[self.continuous_features + self.categorical_features]
        y = (df["domain"] == "real").astype(int)
        self.model.fit(X, y)
        self._n_real = int(y.sum())
        self._n_sim = int(len(y) - self._n_real)
        self._fitted = True
        return self

    def density_ratio(self, df: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call fit() before requesting density ratios")
        X = df[self.continuous_features + self.categorical_features]
        probs = self.model.predict_proba(X)[:, 1]
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        odds = probs / (1 - probs)
        weight_scale = (self._n_sim / max(self._n_real, 1))
        weights = odds * weight_scale
        return pd.Series(weights, index=df.index)

    def attach_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = self.density_ratio(df)
        output = df.copy()
        output["importance_weight"] = np.where(df["domain"] == "sim", weights, 1.0)
        return output

    def resample_sim_trips(
        self,
        df: pd.DataFrame,
        *,
        n_trips: int | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Return a trip-level resampled SIM dataframe following importance weights."""

        if not self._fitted:
            raise RuntimeError("Call fit() before resampling")
        sim_df = df[df["domain"] == "sim"].copy()
        if sim_df.empty:
            return sim_df
        weights = self.density_ratio(sim_df)
        sim_df["_weight"] = weights
        trip_weights = sim_df.groupby("trip_id")["_weight"].mean()
        trip_weights = trip_weights[trip_weights > 0]
        if trip_weights.empty:
            return sim_df.drop(columns="_weight")
        total = trip_weights.sum()
        probs = (trip_weights / total).to_numpy()
        n_samples = n_trips or len(trip_weights)
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(trip_weights.index.to_numpy(), size=n_samples, replace=True, p=probs)
        trip_groups = {trip_id: group.drop(columns="_weight") for trip_id, group in sim_df.groupby("trip_id")}
        resampled = pd.concat([trip_groups[trip_id].copy() for trip_id in chosen], ignore_index=True)
        return resampled

    def build_reweighted_frame(
        self,
        df: pd.DataFrame,
        *,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Construct a dataframe with resampled SIM trips alongside REAL trips."""

        resampled_sim = self.resample_sim_trips(df, random_state=random_state)
        real_df = df[df["domain"] == "real"].copy()
        combined = pd.concat([resampled_sim, real_df], ignore_index=True)
        if {"trip_id", "timestamp_s"}.issubset(combined.columns):
            combined = combined.sort_values(["domain", "trip_id", "timestamp_s"]).reset_index(drop=True)
        return combined

    def evaluate(self, df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        reports = []
        weights = self.density_ratio(df)
        sim_mask = df["domain"] == "sim"
        real_mask = ~sim_mask
        sim_weights = weights[sim_mask].clip(lower=1e-6)
        sim_weights = sim_weights / sim_weights.sum()
        for feature in features:
            sim_values = df.loc[sim_mask, feature].to_numpy()
            real_values = df.loc[real_mask, feature].to_numpy()
            before = stats.wasserstein_distance(sim_values, real_values)
            after = stats.wasserstein_distance(sim_values, real_values, u_weights=sim_weights)
            reports.append(
                ReweightingReport(
                    feature=feature,
                    wasserstein_before=float(before),
                    wasserstein_after=float(after),
                    improvement=float(before - after),
                )
            )
        return pd.DataFrame([r.__dict__ for r in reports])
