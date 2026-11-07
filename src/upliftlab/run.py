"""Run every comparison and write ``summary.json``.

    upliftlab-run --out results

Learners are chosen by 5-fold cross-validated Qini area on the training half. The test half is used once, for the
numbers reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import ARMS, load, split
from .evaluation import bootstrap, by_decile, incremental_curve, ipw_value, qini_area, uplift_in_top
from .learners import LEARNERS

PAIRS = {"mens_vs_none": 1, "womens_vs_none": 2}
OUTCOMES = ("visit", "conversion")
BUDGETS = (0.1, 0.2, 0.3, 0.5, 1.0)
N_BOOT = 500


def binary_rows(arm: np.ndarray, treated_arm: int) -> np.ndarray:
    return np.flatnonzero((arm == 0) | (arm == treated_arm))


def cv_qini(name: str, X: np.ndarray, t: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for g in (0, 1):  # stratify by treatment
        rows = rng.permutation(np.flatnonzero(t == g))
        fold[rows] = np.arange(len(rows)) % folds
    scores = []
    for f in range(folds):
        tr, va = fold != f, fold == f
        model = LEARNERS[name](seed).fit(X[tr], t[tr], y[tr])
        scores.append(qini_area(model.predict(X[va]), t[va], y[va]))
    return float(np.mean(scores))


def paired_qini(a: np.ndarray, b: np.ndarray, t: np.ndarray, y: np.ndarray) -> dict:
    stat = lambda sa, sb, tt, yy: qini_area(sa, tt, yy) - qini_area(sb, tt, yy)  # noqa: E731
    low, high = bootstrap(stat, (a, b, t, y), t, N_BOOT)
    return {"diff": round(stat(a, b, t, y), 5), "low": round(low, 5), "high": round(high, 5)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/hillstrom.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    exp = load(args.data)
    train_rows, test_rows = split(exp)
    train, test = exp.subset(train_rows), exp.subset(test_rows)
    report: dict = {
        "customers": {"train": len(train_rows), "test": len(test_rows)},
        "arm_means_test": {
            ARMS[a]: {o: round(float(test.outcomes[o][test.arm == a].mean()), 5) for o in ("visit", "conversion", "spend")} for a in range(3)
        },
        "binary": {},
    }
    rng = np.random.default_rng(1)
    selected_models: dict = {}

    for pair, treated_arm in PAIRS.items():
        tr, te = binary_rows(train.arm, treated_arm), binary_rows(test.arm, treated_arm)
        Xtr, ttr = train.X.to_numpy()[tr], (train.arm[tr] == treated_arm).astype(float)
        Xte, tte = test.X.to_numpy()[te], (test.arm[te] == treated_arm).astype(float)
        for outcome in OUTCOMES:
            ytr, yte = train.outcomes[outcome][tr], test.outcomes[outcome][te]
            key = f"{pair}/{outcome}"
            entry: dict = {"ate_test": round(float(yte[tte == 1].mean() - yte[tte == 0].mean()), 5), "learners": {}}
            cv = {name: round(cv_qini(name, Xtr, ttr, ytr), 5) for name in LEARNERS}
            uplift_learners = [n for n in LEARNERS if n != "response"]
            best = max(uplift_learners, key=lambda n: cv[n])
            entry["cv_qini_train"] = cv
            entry["selected"] = best
            scores = {"random": rng.uniform(size=len(yte))}
            for name in LEARNERS:
                model = LEARNERS[name](0).fit(Xtr, ttr, ytr)
                scores[name] = model.predict(Xte)
                if name == best:
                    selected_models[key] = model
            for name, s in scores.items():
                low, high = bootstrap(qini_area, (s, tte, yte), tte, N_BOOT)
                entry["learners"][name] = {
                    "qini_area": round(qini_area(s, tte, yte), 5),
                    "low": round(low, 5),
                    "high": round(high, 5),
                    **{f"uplift_top{int(f * 100)}": round(uplift_in_top(s, tte, yte, f), 5) for f in (0.1, 0.3, 0.5)},
                }
            entry["paired"] = {
                f"{best} - random": paired_qini(scores[best], scores["random"], tte, yte),
                f"{best} - response": paired_qini(scores[best], scores["response"], tte, yte),
                "response - random": paired_qini(scores["response"], scores["random"], tte, yte),
            }
            entry["deciles_selected"] = by_decile(scores[best], tte, yte)
            curve = incremental_curve(scores[best], tte, yte, np.array(BUDGETS))
            entry["budget_selected"] = {
                f"{int(b * 100)}%": {
                    "extra_per_1000_customers": round(float(c) * 1000, 2),
                    "random_targeting": round(b * float(curve[-1]) * 1000, 2),
                    "share_of_full_effect": round(float(c / curve[-1]), 3) if curve[-1] else None,
                }
                for b, c in zip(BUDGETS, curve)
            }
            report["binary"][key] = entry
            print(key, "selected", best, {n: v["qini_area"] for n, v in entry["learners"].items()}, flush=True)
            (Path(args.out)).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "summary.json").write_text(json.dumps(report, indent=2))

    # Which email, if any, should each customer get? Value per customer, estimated by inverse propensity weighting.
    propensity = {a: float(np.mean(test.arm == a)) for a in range(3)}
    report["policy"] = {"propensity_test": {ARMS[a]: round(p, 4) for a, p in propensity.items()}}
    Xall_train, Xall_test = train.X.to_numpy(), test.X.to_numpy()
    for outcome in ("visit", "conversion", "spend"):
        y = test.outcomes[outcome]
        taus = []
        learner_names = []
        for pair, treated_arm in PAIRS.items():
            if outcome in OUTCOMES:
                name = report["binary"][f"{pair}/{outcome}"]["selected"]
            else:
                name = "dr_learner"  # spend wasn't part of learner selection; DR was fixed in advance for it
            tr = binary_rows(train.arm, treated_arm)
            t = (train.arm[tr] == treated_arm).astype(float)
            model = LEARNERS[name](0).fit(Xall_train[tr], t, train.outcomes[outcome][tr])
            taus.append(model.predict(Xall_test))
            learner_names.append(name)
        tau = np.column_stack([np.zeros(len(y)), *taus])
        policies = {
            "no_email": np.zeros(len(y), dtype=int),
            "everyone_mens": np.ones(len(y), dtype=int),
            "everyone_womens": np.full(len(y), 2),
            "model_best_arm": tau.argmax(axis=1),
        }
        rows = {}
        for name, pol in policies.items():
            low, high = bootstrap(lambda p, a, yy: ipw_value(p, a, yy, propensity), (pol, test.arm, y), test.arm, N_BOOT)
            rows[name] = {
                "value_per_customer": round(ipw_value(pol, test.arm, y, propensity), 5),
                "low": round(low, 5),
                "high": round(high, 5),
                "share_mens": round(float(np.mean(pol == 1)), 3),
                "share_womens": round(float(np.mean(pol == 2)), 3),
            }
        diff = lambda p, q, a, yy: ipw_value(p, a, yy, propensity) - ipw_value(q, a, yy, propensity)  # noqa: E731
        low, high = bootstrap(diff, (policies["model_best_arm"], policies["everyone_mens"], test.arm, y), test.arm, N_BOOT)
        report["policy"][outcome] = {
            "learners": learner_names,
            "policies": rows,
            "model_minus_everyone_mens": {
                "diff": round(diff(policies["model_best_arm"], policies["everyone_mens"], test.arm, y), 5),
                "low": round(low, 5),
                "high": round(high, 5),
            },
        }
        print(outcome, json.dumps(report["policy"][outcome]), flush=True)
        (Path(args.out) / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
