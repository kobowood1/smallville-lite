"""Pydantic output models, one per LLM task. Phase 3 adds the cognitive task schemas.

Structured outputs do not enforce numeric bounds (minimum/maximum); the SDK moves them into the
field description, and ``LLMClient`` enforces them on validation (with a repair retry).
Prefer ``Literal`` enums for small bounded choices.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ping(_Out):
    pong: str
    echo: str
