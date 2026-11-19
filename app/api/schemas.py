"""Request/response schemas (validated at the edge)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    previous_source_ids: list[str] = Field(default_factory=list, max_length=20)
    stream: bool = True

    @field_validator("message")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str | None = None
    intent: str | None = None
    filters: dict = Field(default_factory=dict)
    total_matches: int | None = None
    relaxed: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    latency_ms: int | None = None
