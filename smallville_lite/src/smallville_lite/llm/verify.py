"""Verify configured model IDs against the live Models API before a run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class ModelCheck:
    model: str
    ok: bool
    display_name: str | None = None
    max_input_tokens: int | None = None
    max_tokens: int | None = None
    error: str | None = None


def verify_models(model_ids: Iterable[str], client: Any | None = None) -> list[ModelCheck]:
    """Retrieve each model ID; an unknown or retired ID comes back with ``ok=False``."""
    import anthropic

    client = client or anthropic.Anthropic()
    results = []
    for mid in sorted(set(model_ids)):
        try:
            m = client.models.retrieve(mid)
            results.append(ModelCheck(
                model=mid, ok=True, display_name=getattr(m, "display_name", None),
                max_input_tokens=getattr(m, "max_input_tokens", None), max_tokens=getattr(m, "max_tokens", None),
            ))
        except anthropic.NotFoundError:
            results.append(ModelCheck(model=mid, ok=False, error="not found (unknown or retired model ID)"))
        except anthropic.APIError as exc:
            results.append(ModelCheck(model=mid, ok=False, error=f"{type(exc).__name__}: {exc}"))
    return results
