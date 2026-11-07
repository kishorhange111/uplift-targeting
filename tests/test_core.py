"""Simulated randomised experiments with a known effect, so each estimator can be checked against the truth."""

import numpy as np
import pandas as pd
import pytest

from upliftlab.data import ARMS, from_frame, split
from upliftlab.evaluation import by_decile, incremental_curve, ipw_value, qini_area, uplift_in_top
from upliftlab.learners import LEARNERS


def simulate(n=20000, seed=0):
    """Effect on P(visit) is 0.15 for x0 > 0 and 0 otherwise. x1 drives baseline visits but not the effect."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    t = rng.integers(0, 2, n).astype(float)
    tau = np.where(X[:, 0] > 0, 0.15, 0.0)
    base = 0.1 + 0.2 * (X[:, 1] > 0)
    y = (rng.uniform(size=n) < base + t * tau).astype(float)
    return X, t, y, tau


def test_curve_ends_at_the_average_effect_and_random_scores_near_zero():
    X, t, y, tau = simulate()
    curve = incremental_curve(np.zeros(len(y)), t, y)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert curve[-1] == pytest.approx(ate)
    assert abs(qini_area(np.zeros(len(y)), t, y)) < 0.005


def test_oracle_beats_random_and_baseline_driven_score():
    X, t, y, tau = simulate()
    oracle = qini_area(tau, t, y)
    assert oracle > 0.015
    # Ranking by baseline visit propensity (what a response model learns here) carries no uplift signal.
    assert abs(qini_area(X[:, 1], t, y)) < 0.006
    assert uplift_in_top(tau, t, y, 0.5) == pytest.approx(0.15, abs=0.03)


@pytest.mark.parametrize("name", ["t_learner", "x_learner", "dr_learner", "transformed_outcome", "s_learner"])
def test_uplift_learners_recover_the_effect_ranking(name):
    X, t, y, tau = simulate()
    Xte, tte, yte, taute = simulate(seed=1)
    pred = LEARNERS[name](0).fit(X, t, y).predict(Xte)
    assert np.corrcoef(pred, taute)[0, 1] > 0.5
    assert qini_area(pred, tte, yte) > 0.6 * qini_area(taute, tte, yte)


def test_deciles_show_the_effect_concentrated_at_the_top():
    X, t, y, tau = simulate()
    rows = by_decile(tau + np.random.default_rng(0).normal(0, 1e-6, len(tau)), t, y)
    assert np.mean([r["observed"] for r in rows[:5]]) > 0.1
    assert abs(np.mean([r["observed"] for r in rows[5:]])) < 0.03


def test_ipw_value_matches_the_true_policy_value():
    rng = np.random.default_rng(0)
    n = 60000
    arm = rng.integers(0, 3, n)
    x = rng.normal(size=n)
    # Arm 1 helps when x > 0, arm 2 helps when x < 0.
    p = 0.1 + 0.1 * ((arm == 1) & (x > 0)) + 0.1 * ((arm == 2) & (x < 0))
    y = (rng.uniform(size=n) < p).astype(float)
    best = np.where(x > 0, 1, 2)
    prop = {a: float(np.mean(arm == a)) for a in range(3)}
    assert ipw_value(best, arm, y, prop) == pytest.approx(0.2, abs=0.01)
    assert ipw_value(np.zeros(n, int), arm, y, prop) == pytest.approx(0.1, abs=0.01)
    assert ipw_value(np.ones(n, int), arm, y, prop) == pytest.approx(0.15, abs=0.01)


def test_data_encoding_and_stratified_split():
    frame = pd.DataFrame(
        {
            "recency": [1, 2, 3, 4, 5, 6],
            "history_segment": ["a"] * 6,
            "history": [10.0, 20, 30, 40, 50, 60],
            "mens": [1, 0, 1, 0, 1, 0],
            "womens": [0, 1, 0, 1, 0, 1],
            "zip_code": ["Urban", "Rural", "Urban", "Surburban", "Rural", "Urban"],
            "newbie": [0, 1, 0, 1, 0, 1],
            "channel": ["Web", "Phone", "Web", "Multichannel", "Web", "Phone"],
            "segment": [ARMS[i % 3] for i in range(6)],
            "visit": [0, 1, 0, 1, 1, 0],
            "conversion": [0] * 6,
            "spend": [0.0] * 6,
        }
    )
    exp = from_frame(frame)
    assert "history_segment" not in exp.X.columns and "zip_code=Urban" in exp.X.columns
    assert exp.arm.tolist() == [0, 1, 2, 0, 1, 2]
    train, test = split(exp)
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(6))
    assert sorted(exp.arm[test].tolist()) == [0, 1, 2]
