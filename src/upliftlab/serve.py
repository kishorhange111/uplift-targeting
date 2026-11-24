"""Production uplift policy scoring service, FastAPI wrapper, and CLI entrypoint."""

from __future__ import annotations

import argparse

import numpy as np

from .config import (
    CustomerDecision,
    CustomerFeatures,
    PolicyConfig,
    PolicyEvalRequest,
    PolicyEvalResponse,
    ScoreRequest,
    ScoreResponse,
    ServeConfig,
)
from .data import ARMS
from .evaluation import ipw_value
from .learners import LEARNERS, Learner


def encode_customers(customers: list[CustomerFeatures]) -> np.ndarray:
    """Convert validated customer records into the canonical 11-column Hillstrom feature matrix."""
    rows = []
    for c in customers:
        rows.append(
            [
                float(c.recency),
                float(c.history),
                float(c.mens),
                float(c.womens),
                float(c.newbie),
                1.0 if c.zip_code == "Rural" else 0.0,
                1.0 if c.zip_code == "Surburban" else 0.0,
                1.0 if c.zip_code == "Urban" else 0.0,
                1.0 if c.channel == "Multichannel" else 0.0,
                1.0 if c.channel == "Phone" else 0.0,
                1.0 if c.channel == "Web" else 0.0,
            ]
        )
    return np.asarray(rows, dtype=float)


class UpliftPolicyService:
    """Two-arm uplift policy wrapper recommending No E-Mail, Mens E-Mail, or Womens E-Mail."""

    def __init__(
        self,
        config: PolicyConfig | None = None,
        mens_model: Learner | None = None,
        womens_model: Learner | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.mens_model = mens_model
        self.womens_model = womens_model

    @property
    def is_ready(self) -> bool:
        return self.mens_model is not None and self.womens_model is not None

    def fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> UpliftPolicyService:
        cls = LEARNERS[self.config.learner]
        m_mask = (arm == 0) | (arm == 1)
        w_mask = (arm == 0) | (arm == 2)
        self.mens_model = cls(self.config.seed).fit(X[m_mask], (arm[m_mask] == 1).astype(float), y[m_mask])
        self.womens_model = cls(self.config.seed).fit(X[w_mask], (arm[w_mask] == 2).astype(float), y[w_mask])
        return self

    def score(self, customers: list[CustomerFeatures], min_uplift_threshold: float | None = None) -> ScoreResponse:
        if not self.is_ready or self.mens_model is None or self.womens_model is None:
            raise RuntimeError("policy models are not fitted")
        X = encode_customers(customers)
        tau_m = self.mens_model.predict(X)
        tau_w = self.womens_model.predict(X)
        threshold = self.config.min_uplift_threshold if min_uplift_threshold is None else min_uplift_threshold

        decisions = []
        counts = dict.fromkeys(ARMS, 0)
        for i, (tm, tw) in enumerate(zip(tau_m, tau_w)):
            tm_f, tw_f = float(tm), float(tw)
            if max(tm_f, tw_f) <= threshold:
                chosen = ARMS[0]
            elif tm_f >= tw_f:
                chosen = ARMS[1]
            else:
                chosen = ARMS[2]
            counts[chosen] += 1
            decisions.append(
                CustomerDecision(
                    customer_index=i,
                    tau_mens=round(tm_f, 6),
                    tau_womens=round(tw_f, 6),
                    recommended_arm=chosen,
                )
            )
        return ScoreResponse(learner=self.config.learner, decisions=decisions, arm_counts=counts)


def create_app(service: UpliftPolicyService | None = None):
    """Create a FastAPI app for uplift scoring and IPW policy evaluation."""
    from fastapi import FastAPI, HTTPException

    svc = service or UpliftPolicyService()
    app = FastAPI(title="upliftlab Policy API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "ready": svc.is_ready, "learner": svc.config.learner}

    @app.get("/learners")
    def learners() -> dict:
        return {"learners": list(LEARNERS), "arms": list(ARMS)}

    @app.post("/score", response_model=ScoreResponse)
    def score_customers(req: ScoreRequest) -> ScoreResponse:
        try:
            return svc.score(req.customers, min_uplift_threshold=req.min_uplift_threshold)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/policy/evaluate", response_model=PolicyEvalResponse)
    def evaluate_policy(req: PolicyEvalRequest) -> PolicyEvalResponse:
        if not (len(req.policy) == len(req.arm) == len(req.outcome)):
            raise HTTPException(status_code=422, detail="policy, arm, and outcome must have equal length")
        if any(p <= 0 for p in req.propensity.values()):
            raise HTTPException(status_code=422, detail="propensities must be strictly positive")
        try:
            val = ipw_value(
                np.asarray(req.policy, dtype=int),
                np.asarray(req.arm, dtype=int),
                np.asarray(req.outcome, dtype=float),
                req.propensity,
            )
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"missing propensity for arm {exc}") from exc
        return PolicyEvalResponse(ipw_value=round(val, 6), samples=len(req.policy))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve upliftlab targeting policy API via uvicorn.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--learner", default="s_learner", choices=list(LEARNERS))
    args = parser.parse_args()

    cfg = ServeConfig(host=args.host, port=args.port, learner=args.learner)
    import uvicorn

    uvicorn.run(create_app(UpliftPolicyService(PolicyConfig(learner=cfg.learner))), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
