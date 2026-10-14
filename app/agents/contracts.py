from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContentBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=500)
    platform: Literal["linkedin"]
    audience: str | None = Field(default=None, max_length=300)
    goal: str | None = Field(default=None, max_length=300)
    brand_id: str | None = Field(default=None, min_length=1, max_length=64)


class SafetyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason: str = Field(min_length=1, max_length=1000)
    risk_categories: list[str] = Field(default_factory=list, max_length=20)


class StrategyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    angle: str = Field(min_length=1, max_length=500)
    audience: str = Field(min_length=1, max_length=300)
    hook: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1500)


class DraftContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=3000)

    @field_validator("text")
    @classmethod
    def reject_whitespace_only_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("draft text must not be empty")
        return text


class EditorCritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=10)
    approved: bool
    critique: str = Field(min_length=1, max_length=2000)
    revision_instructions: list[str] = Field(default_factory=list, max_length=10)


class BrandVoiceSampleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: str = Field(min_length=1, max_length=10000)
    source_label: str | None = Field(default=None, max_length=200)

    @field_validator("body")
    @classmethod
    def normalize_sample_body(cls, body: str) -> str:
        normalized = body.strip()
        if not normalized:
            raise ValueError("sample body must not be empty")
        return normalized


class BrandVoiceSamplesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[BrandVoiceSampleInput] = Field(min_length=3, max_length=50)

    @model_validator(mode="after")
    def require_distinct_sample_bodies(self):
        normalized_bodies = [" ".join(sample.body.split()).casefold() for sample in self.samples]
        if len(set(normalized_bodies)) != len(normalized_bodies):
            raise ValueError("brand voice samples must be distinct")
        return self


class BrandVoiceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: str = Field(min_length=1, max_length=500)
    tone_attributes: list[str] = Field(min_length=1, max_length=12)
    preferred_terms: list[str] = Field(default_factory=list, max_length=30)
    avoid_terms: list[str] = Field(default_factory=list, max_length=30)
    structure_guidelines: list[str] = Field(min_length=1, max_length=20)


class PlaybookRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: str = Field(min_length=1, max_length=500)
    platform: str | None = Field(default=None, max_length=50)
    category: str = Field(default="general", min_length=1, max_length=100)


class BrandMemoryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: BrandVoiceProfile | None = None
    profile_version: int | None = Field(default=None, ge=1)
    playbook_rules: list[PlaybookRule] = Field(default_factory=list)
    playbook_version: int | None = Field(default=None, ge=1)
