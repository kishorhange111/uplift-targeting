"""Typed configuration and API schemas for uplift policy serving."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LearnerName = Literal["response", "s_learner", "t_learner", "x_learner", "dr_learner", "transformed_outcome"]
ZipCode = Literal["Urban", "Surburban", "Rural"]
Channel = Literal["Web", "Phone", "Multichannel"]

CANONICAL_COLUMNS = (
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "zip_code=Rural",
    "zip_code=Surburban",
    "zip_code=Urban",
    "channel=Multichannel",
    "channel=Phone",
    "channel=Web",
)


class CustomerFeatures(BaseModel):
    """Pre-treatment customer features matching the Hillstrom schema."""

    model_config = ConfigDict(extra="forbid")

    recency: float = Field(ge=0)
    history: float = Field(ge=0)
    mens: int = Field(ge=0, le=1)
    womens: int = Field(ge=0, le=1)
    newbie: int = Field(ge=0, le=1)
    zip_code: ZipCode
    channel: Channel


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learner: LearnerName = "s_learner"
    seed: int = 0
    min_uplift_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)
    min_uplift_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class CustomerDecision(BaseModel):
    customer_index: int
    tau_mens: float
    tau_womens: float
    recommended_arm: str


class ScoreResponse(BaseModel):
    learner: LearnerName
    decisions: list[CustomerDecision]
    arm_counts: dict[str, int]


class PolicyEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: list[int] = Field(min_length=1)
    arm: list[int] = Field(min_length=1)
    outcome: list[float] = Field(min_length=1)
    propensity: dict[int, float]


class PolicyEvalResponse(BaseModel):
    ipw_value: float
    samples: int


class ServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    learner: LearnerName = "s_learner"
