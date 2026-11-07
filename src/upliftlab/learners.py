"""Uplift learners on gradient-boosted trees. Each predicts tau(x) = E[Y | x, treated] - E[Y | x, control].

The propensity is known (the experiment was randomised), so it is a constant, never a fitted model.
Base-learner settings are fixed in advance and shared by every learner, so differences come from the method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold

PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    # Few threads: on small data LightGBM's OpenMP threads spin, and grabbing every core slows other work badly.
    "n_jobs": 2,
}


def _outcome_model(binary: bool, seed: int):
    return LGBMClassifier(**PARAMS, random_state=seed) if binary else LGBMRegressor(**PARAMS, random_state=seed)


def _regressor(seed: int):
    return LGBMRegressor(**PARAMS, random_state=seed)


def _predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


@dataclass
class Learner:
    seed: int = 0
    models: dict = field(default_factory=dict)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Learner:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _binary(y: np.ndarray) -> bool:
    return set(np.unique(y)) <= {0.0, 1.0}


class Response(Learner):
    """Not an uplift model: ranks by P(respond | emailed). The usual way campaigns get targeted."""

    def fit(self, X, t, y):
        self.models["m"] = _outcome_model(_binary(y), self.seed).fit(X[t == 1], y[t == 1])
        return self

    def predict(self, X):
        return _predict(self.models["m"], X)


class SLearner(Learner):
    """One model with the treatment flag as a feature. Trees may barely split on it, which shrinks tau towards 0."""

    def fit(self, X, t, y):
        self.models["m"] = _outcome_model(_binary(y), self.seed).fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X):
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return _predict(self.models["m"], np.column_stack([X, ones])) - _predict(self.models["m"], np.column_stack([X, zeros]))


class TLearner(Learner):
    """Separate outcome models for treated and control."""

    def fit(self, X, t, y):
        binary = _binary(y)
        self.models["m1"] = _outcome_model(binary, self.seed).fit(X[t == 1], y[t == 1])
        self.models["m0"] = _outcome_model(binary, self.seed).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return _predict(self.models["m1"], X) - _predict(self.models["m0"], X)


class XLearner(Learner):
    """Künzel et al. (2019): impute each group's individual effect with the other group's model, then regress it."""

    def fit(self, X, t, y):
        base = TLearner(self.seed).fit(X, t, y)
        m0, m1 = base.models["m0"], base.models["m1"]
        d1 = y[t == 1] - _predict(m0, X[t == 1])
        d0 = _predict(m1, X[t == 0]) - y[t == 0]
        self.models["tau1"] = _regressor(self.seed).fit(X[t == 1], d1)
        self.models["tau0"] = _regressor(self.seed).fit(X[t == 0], d0)
        self.e = float(t.mean())
        return self

    def predict(self, X):
        return self.e * _predict(self.models["tau0"], X) + (1 - self.e) * _predict(self.models["tau1"], X)


class DRLearner(Learner):
    """Kennedy (2023): doubly robust pseudo-outcomes from cross-fitted outcome models, then one regression."""

    folds: int = 5

    def fit(self, X, t, y):
        e = float(t.mean())
        mu0, mu1 = np.zeros(len(y)), np.zeros(len(y))
        for train, held in KFold(self.folds, shuffle=True, random_state=self.seed).split(X):
            base = TLearner(self.seed).fit(X[train], t[train], y[train])
            mu0[held] = _predict(base.models["m0"], X[held])
            mu1[held] = _predict(base.models["m1"], X[held])
        pseudo = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self.models["tau"] = _regressor(self.seed).fit(X, pseudo)
        return self

    def predict(self, X):
        return _predict(self.models["tau"], X)


class TransformedOutcome(Learner):
    """Athey and Imbens (2015): with known propensity e, Y* = Y (T - e) / (e (1 - e)) has E[Y* | x] = tau(x)."""

    def fit(self, X, t, y):
        e = float(t.mean())
        self.models["tau"] = _regressor(self.seed).fit(X, y * (t - e) / (e * (1 - e)))
        return self

    def predict(self, X):
        return _predict(self.models["tau"], X)


LEARNERS: dict[str, type[Learner]] = {
    "response": Response,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "dr_learner": DRLearner,
    "transformed_outcome": TransformedOutcome,
}
