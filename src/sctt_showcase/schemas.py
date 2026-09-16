"""Bounded, finite JSON inputs for the small synthetic showcase."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Number = Annotated[FiniteFloat, Field(ge=-1000, le=1000)]
Vec2 = Annotated[list[Number], Field(min_length=2, max_length=2)]
Vec3 = Annotated[list[Number], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[Number], Field(min_length=4, max_length=4)]
Descriptor = Annotated[list[Number], Field(min_length=16, max_length=16)]
Score = Annotated[FiniteFloat, Field(ge=0, le=1)]


class FeatureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    desc: Annotated[list[Descriptor], Field(min_length=1, max_length=128)]
    uv: Annotated[list[Vec2], Field(min_length=1, max_length=128)]
    ray_dir: Annotated[list[Vec3], Field(min_length=1, max_length=128)]
    score: Annotated[list[Score], Field(min_length=1, max_length=128)]
    mask: Annotated[list[bool], Field(min_length=1, max_length=128)] | None = None

    @model_validator(mode="after")
    def matching_rows(self):
        n = len(self.desc)
        if any(len(x) != n for x in (self.uv, self.ray_dir, self.score)):
            raise ValueError("desc, uv, ray_dir and score must have equal row counts")
        if self.mask is None:
            self.mask = [True] * n
        if len(self.mask) != n or not any(self.mask):
            raise ValueError("mask must match features and retain at least one ray")
        if any(valid and sum(v * v for v in ray) < 1e-12 for valid, ray in zip(self.mask, self.ray_dir)):
            raise ValueError("valid rays must have a nonzero direction")
        return self


class CandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, Field(min_length=1, max_length=64)]
    position: Vec3
    descriptor: Descriptor
    log_var: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion: Quaternion = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])

    @model_validator(mode="after")
    def normalized_quaternion(self):
        if abs(sum(v * v for v in self.quaternion) - 1.0) > 0.01:
            raise ValueError("quaternion must have unit length in x,y,z,w order")
        return self


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: Annotated[str, Field(min_length=1, max_length=64)] = "loop-01"
    top_k: Annotated[int, Field(ge=1, le=8)] = 4
    candidate_offset_m: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    features: FeatureInput | None = None
    candidates: Annotated[list[CandidateInput], Field(min_length=1, max_length=8)] | None = None

    @model_validator(mode="after")
    def complete_custom_input(self):
        if (self.features is None) != (self.candidates is None):
            raise ValueError("custom inference requires both features and candidates")
        if self.candidates is not None:
            ids = [c.id for c in self.candidates]
            if len(set(ids)) != len(ids):
                raise ValueError("candidate IDs must be unique")
        return self
