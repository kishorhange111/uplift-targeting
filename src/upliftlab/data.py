"""The MineThatData E-Mail Analytics challenge (Hillstrom, 2008).

64,000 past customers were randomly split three ways: a men's merchandise email, a women's merchandise email, or no
email. Outcomes were tracked for two weeks: whether they visited the site, whether they bought, and how much they spent.
Because assignment was random with known shares, uplift and policy value can be estimated without bias.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
ARMS = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
OUTCOMES = ("visit", "conversion", "spend")


@dataclass
class Experiment:
    X: pd.DataFrame
    arm: np.ndarray  # 0 = no email, 1 = men's, 2 = women's
    outcomes: dict[str, np.ndarray]

    def subset(self, rows: np.ndarray) -> Experiment:
        return Experiment(self.X.iloc[rows].reset_index(drop=True), self.arm[rows], {k: v[rows] for k, v in self.outcomes.items()})


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment customer attributes only. ``history_segment`` is a binned copy of ``history`` and is dropped."""
    X = frame[["recency", "history", "mens", "womens", "newbie"]].astype(float).copy()
    for column in ("zip_code", "channel"):
        for value in sorted(frame[column].unique()):
            X[f"{column}={value}"] = (frame[column] == value).astype(float)
    return X


def from_frame(frame: pd.DataFrame) -> Experiment:
    arm = frame["segment"].map({name: i for i, name in enumerate(ARMS)}).to_numpy()
    if np.isnan(arm.astype(float)).any():
        raise ValueError("unknown segment")
    return Experiment(features(frame), arm.astype(int), {o: frame[o].to_numpy(dtype=float) for o in OUTCOMES})


def load(path: str | Path = "data/hillstrom.csv") -> Experiment:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path} has sha256 {digest}, expected {SHA256}")
    return from_frame(pd.read_csv(path))


def split(exp: Experiment, test_fraction: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train and test rows, stratified by arm so each half keeps the experiment's balance."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for a in np.unique(exp.arm):
        rows = rng.permutation(np.flatnonzero(exp.arm == a))
        cut = int(round(len(rows) * test_fraction))
        test.append(rows[:cut])
        train.append(rows[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))
