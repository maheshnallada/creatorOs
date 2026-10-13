from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContentBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=500)
    platform: Literal["linkedin"]
    audience: str | None = Field(default=None, max_length=300)
    goal: str | None = Field(default=None, max_length=300)


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
