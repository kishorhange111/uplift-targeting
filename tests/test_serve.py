import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from upliftlab.config import CustomerFeatures, PolicyConfig, ServeConfig
from upliftlab.learners import Learner
from upliftlab.serve import UpliftPolicyService, create_app, encode_customers


class _ConstantLearner(Learner):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def _sample_customer(mens: int = 1, womens: int = 0) -> dict:
    return {
        "recency": 3,
        "history": 150.0,
        "mens": mens,
        "womens": womens,
        "newbie": 0,
        "zip_code": "Urban",
        "channel": "Web",
    }


def test_config_validation_rejects_invalid_inputs():
    with pytest.raises(ValidationError):
        PolicyConfig(learner="invalid")
    with pytest.raises(ValidationError):
        ServeConfig(port=99999)
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**_sample_customer(), "zip_code": "Mars"})
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**_sample_customer(), "recency": -1})


def test_encode_customers_produces_canonical_11_features():
    X = encode_customers([CustomerFeatures(**_sample_customer())])
    assert X.shape == (1, 11)
    assert X[0, 0] == 3.0 and X[0, 7] == 1.0 and X[0, 10] == 1.0


def test_service_and_fastapi_endpoints():
    unfitted_client = TestClient(create_app())
    assert unfitted_client.get("/health").json()["ready"] is False
    assert unfitted_client.post("/score", json={"customers": [_sample_customer()]}).status_code == 503

    svc = UpliftPolicyService(
        config=PolicyConfig(learner="s_learner", min_uplift_threshold=0.01),
        mens_model=_ConstantLearner(0.08),
        womens_model=_ConstantLearner(0.03),
    )
    client = TestClient(create_app(svc))
    assert client.get("/health").json() == {"status": "ok", "ready": True, "learner": "s_learner"}
    assert "s_learner" in client.get("/learners").json()["learners"]

    res = client.post("/score", json={"customers": [_sample_customer(), _sample_customer(0, 1)]})
    assert res.status_code == 200
    body = res.json()
    assert body["arm_counts"]["Mens E-Mail"] == 2
    assert body["decisions"][0]["recommended_arm"] == "Mens E-Mail"

    high_thresh = client.post("/score", json={"customers": [_sample_customer()], "min_uplift_threshold": 0.2})
    assert high_thresh.json()["decisions"][0]["recommended_arm"] == "No E-Mail"

    eval_res = client.post(
        "/policy/evaluate",
        json={"policy": [1, 1, 0], "arm": [1, 0, 0], "outcome": [1.0, 0.0, 1.0], "propensity": {0: 0.5, 1: 0.5}},
    )
    assert eval_res.status_code == 200
    assert eval_res.json()["samples"] == 3
    assert eval_res.json()["ipw_value"] == pytest.approx(4.0 / 3.0, abs=1e-5)

    bad_eval = client.post(
        "/policy/evaluate",
        json={"policy": [1], "arm": [1, 0], "outcome": [1.0], "propensity": {0: 0.5, 1: 0.5}},
    )
    assert bad_eval.status_code == 422
