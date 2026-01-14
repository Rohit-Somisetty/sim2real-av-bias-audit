"""Domain classifier that measures SIM vs REAL separability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DomainClassifierResult:
    auc: float
    feature_importances: pd.DataFrame


def train_domain_classifier(
    df: pd.DataFrame,
    *,
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    penalty: str = "l2",
    max_iter: int = 1000,
) -> DomainClassifierResult:
    """Train a simple classifier and report AUC + feature importances."""

    X = df[list(continuous_features) + list(categorical_features)]
    y = (df["domain"] == "real").astype(int)

    transformers = []
    if continuous_features:
        transformers.append(("num", StandardScaler(), list(continuous_features)))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_features)))
    preprocessor = ColumnTransformer(transformers)

    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    penalty=penalty,
                    max_iter=max_iter,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, probs))

    clf = model.named_steps["clf"]
    feature_names = model.named_steps["pre"].get_feature_names_out()
    coefs = clf.coef_.ravel()
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "weight": coefs,
            "abs_weight": np.abs(coefs),
        }
    ).sort_values("abs_weight", ascending=False)

    return DomainClassifierResult(auc=auc, feature_importances=importances)
