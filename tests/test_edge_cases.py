"""Edge-case, regression, and data-validation tests for upliftlab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from upliftlab.config import CustomerFeatures, PolicyConfig
from upliftlab.data import ARMS, from_frame, split
from upliftlab.evaluation import bootstrap, incremental_curve, ipw_value, qini_area, ranking, uplift_in_top
from upliftlab.run import binary_rows
from upliftlab.serve import UpliftPolicyService


def _toy_frame(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recency": np.arange(1, n + 1, dtype=float),
            "history_segment": ["1) $0 - $100"] * n,
            "history": np.linspace(10.0, 300.0, n),
            "mens": [i % 2 for i in range(n)],
            "womens": [(i + 1) % 2 for i in range(n)],
            "zip_code": [["Urban", "Surburban", "Rural"][i % 3] for i in range(n)],
            "newbie": [i % 2 for i in range(n)],
            "channel": [["Web", "Phone", "Multichannel"][i % 3] for i in range(n)],
            "segment": [ARMS[i % 3] for i in range(n)],
            "visit": [1 if i % 4 == 0 else 0 for i in range(n)],
            "conversion": [0] * n,
            "spend": [0.0] * n,
        }
    )


def test_from_frame_rejects_unknown_segment():
    df = _toy_frame(6)
    df.loc[0, "segment"] = "Unknown Arm"
    with pytest.raises(ValueError, match="unknown segment"):
        from_frame(df)


def test_split_preserves_arm_proportions_and_determinism():
    exp = from_frame(_toy_frame(30))
    tr1, te1 = split(exp, test_fraction=0.5, seed=7)
    tr2, te2 = split(exp, test_fraction=0.5, seed=7)
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
    assert len(tr1) == 15 and len(te1) == 15
    for arm in range(3):
        assert int((exp.arm[tr1] == arm).sum()) == 5
        assert int((exp.arm[te1] == arm).sum()) == 5


def test_ranking_breaks_ties_deterministically_by_seed():
    scores = np.zeros(20)
    r1 = ranking(scores, seed=42)
    r2 = ranking(scores, seed=42)
    r3 = ranking(scores, seed=99)
    assert np.array_equal(r1, r2)
    assert not np.array_equal(r1, r3)


def test_incremental_curve_handles_all_treated_prefix_without_nan():
    score = np.array([10.0, 9.0, 8.0, 1.0, 0.5, 0.1])
    t = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    curve = incremental_curve(score, t, y)
    assert np.isfinite(curve).all()
    assert curve[0] == 0.0


def test_qini_area_is_invariant_to_strictly_increasing_score_transform():
    rng = np.random.default_rng(3)
    n = 1000
    s = rng.normal(size=n)
    t = rng.integers(0, 2, size=n).astype(float)
    y = (rng.uniform(size=n) < 0.2 + 0.1 * t * (s > 0)).astype(float)
    assert qini_area(s, t, y) == pytest.approx(qini_area(np.exp(s) + 5.0, t, y), abs=1e-12)


def test_uplift_in_top_at_full_fraction_equals_ate():
    rng = np.random.default_rng(5)
    n = 500
    s = rng.normal(size=n)
    t = rng.integers(0, 2, size=n).astype(float)
    y = rng.binomial(1, 0.25, size=n).astype(float)
    ate = float(y[t == 1].mean() - y[t == 0].mean())
    assert uplift_in_top(s, t, y, 1.0) == pytest.approx(ate)


def test_bootstrap_interval_is_reproducible_and_ordered():
    rng = np.random.default_rng(11)
    n = 400
    t = rng.integers(0, 2, size=n).astype(float)
    y = (rng.uniform(size=n) < 0.1 + 0.15 * t).astype(float)
    stat = lambda tt, yy: float(yy[tt == 1].mean() - yy[tt == 0].mean())  # noqa: E731
    lo1, hi1 = bootstrap(stat, (t, y), t, n_boot=100, seed=0)
    lo2, hi2 = bootstrap(stat, (t, y), t, n_boot=100, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < stat(t, y) < hi1


def test_ipw_value_zero_when_policy_never_matches_observed_arm():
    policy = np.array([1, 1, 1])
    arm = np.array([0, 0, 0])
    y = np.array([1.0, 1.0, 1.0])
    assert ipw_value(policy, arm, y, {0: 0.5, 1: 0.5}) == 0.0


def test_binary_rows_filters_to_control_and_target_arm():
    arm = np.array([0, 1, 2, 0, 2, 1])
    assert binary_rows(arm, 1).tolist() == [0, 1, 3, 5]
    assert binary_rows(arm, 2).tolist() == [0, 2, 3, 4]


def test_service_fit_and_score_end_to_end():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 11))
    arm = rng.integers(0, 3, size=n)
    y = (rng.uniform(size=n) < 0.1 + 0.2 * (arm == 1)).astype(float)
    svc = UpliftPolicyService(PolicyConfig(learner="s_learner", seed=0)).fit(X, arm, y)
    assert svc.is_ready
    resp = svc.score(
        [
            CustomerFeatures(
                recency=2,
                history=120.0,
                mens=1,
                womens=0,
                newbie=0,
                zip_code="Urban",
                channel="Web",
            )
        ]
    )
    assert len(resp.decisions) == 1
    assert sum(resp.arm_counts.values()) == 1
