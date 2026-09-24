"""Deterministic fake backend for STUB mode (DESIGN §6.6): zero network, zero spend.

* Output: a per-task *responder* if one is registered (Phase 3 registers scenario-aware
  responders), else a generic generator that produces a schema-valid object from the
  Pydantic model's JSON schema. Randomness is seeded from (seed, task, messages, attempt), so
  the same run reproduces exactly.
* Usage: tokens are ``ceil(chars / chars_per_token)`` with each model's configured ratio. Prompt caching is simulated with the model's real
  minimum cacheable length: the first request with a given prefix writes it, later ones read
  it. Costs are then priced with the real price table, so budget logic runs end to end.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .types import BackendRequest, BackendResponse, Usage

_CHARS_PER_TOKEN = 4
_WORDS = (
    "coffee", "garden", "library", "party", "neighbor", "morning", "plan", "research",
    "market", "walk", "friend", "election", "cafe", "notes", "music", "evening",
)


@dataclass
class FakeReply:
    """What a responder may return to control the raw text or the stop reason."""

    text: str
    stop_reason: str = "end_turn"


Responder = Callable[[BackendRequest, random.Random], "dict[str, Any] | FakeReply"]


class FakeBackend:
    def __init__(
        self,
        seed: int = 0,
        *,
        min_cache_tokens: Mapping[str, int] | None = None,
        chars_per_token: Mapping[str, float] | None = None,
        responders: Mapping[str, Responder] | None = None,
    ) -> None:
        self.seed = seed
        self.min_cache_tokens = dict(min_cache_tokens or {})
        self.chars_per_token = dict(chars_per_token or {})
        self.responders: dict[str, Responder] = dict(responders or {})
        self._seen_prefixes: set[str] = set()
        self.requests: list[BackendRequest] = []

    def register(self, task: str, responder: Responder) -> None:
        self.responders[task] = responder

    def complete(self, req: BackendRequest) -> BackendResponse:
        self.requests.append(req)
        rng = random.Random(self._rng_seed(req))
        responder = self.responders.get(req.task)
        produced = responder(req, rng) if responder else generate_from_schema(req.output_model.model_json_schema(), rng)
        if isinstance(produced, FakeReply):
            text, stop = produced.text, produced.stop_reason
        else:
            text, stop = json.dumps(produced, ensure_ascii=False), "end_turn"
        return BackendResponse(
            text=text,
            stop_reason=stop,
            usage=self._usage(req, text),
            request_id=f"stub-{len(self.requests)}",
            latency_ms=0.0,
        )

    # -- internals ---------------------------------------------------------
    def _rng_seed(self, req: BackendRequest) -> int:
        key = json.dumps([self.seed, req.task, req.messages, req.attempt], sort_keys=True, default=str)
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")

    def _usage(self, req: BackendRequest, text: str) -> Usage:
        prefix_chars, ttl, key = 0, "5m", None
        for i, block in enumerate(req.system):
            if "cache_control" in block:
                prefix_chars = sum(len(b.get("text", "")) for b in req.system[: i + 1])
                ttl = block["cache_control"].get("ttl", "5m")
                key = req.model + "|" + json.dumps(req.system[: i + 1], sort_keys=True)
        total_chars = sum(len(b.get("text", "")) for b in req.system)
        for m in req.messages:
            c = m["content"]
            total_chars += len(c) if isinstance(c, str) else sum(len(b.get("text", "")) for b in c)
        total_chars += len(json.dumps(req.output_schema))
        cpt = self.chars_per_token.get(req.model, _CHARS_PER_TOKEN)
        total = _tokens(total_chars, cpt)
        prefix = _tokens(prefix_chars, cpt)
        read = write5 = write1 = 0
        if key is not None and prefix >= self.min_cache_tokens.get(req.model, 1024):
            if key in self._seen_prefixes:
                read = prefix
            else:
                self._seen_prefixes.add(key)
                if ttl == "1h":
                    write1 = prefix
                else:
                    write5 = prefix
        return Usage(
            input_tokens=total - read - write5 - write1,
            output_tokens=_tokens(len(text), cpt),
            cache_read_tokens=read,
            cache_write_5m_tokens=write5,
            cache_write_1h_tokens=write1,
        )


def _tokens(chars: int, chars_per_token: float = _CHARS_PER_TOKEN) -> int:
    return math.ceil(chars / chars_per_token)


# ---------------------------------------------------------------------------
# Generic schema-valid generator (works on Pydantic's raw JSON schema)


def generate_from_schema(schema: dict[str, Any], rng: random.Random) -> Any:
    return _gen(schema, schema, rng, "value")


def _gen(node: dict[str, Any], root: dict[str, Any], rng: random.Random, path: str) -> Any:
    if "$ref" in node:
        name = node["$ref"].split("/")[-1]
        return _gen(root["$defs"][name], root, rng, path)
    if "const" in node:
        return node["const"]
    if "enum" in node:
        return rng.choice(node["enum"])
    for key in ("anyOf", "oneOf"):
        if key in node:
            variants = [v for v in node[key] if v.get("type") != "null"] or node[key]
            return _gen(variants[0], root, rng, path)
    if "allOf" in node:
        return _gen(node["allOf"][0], root, rng, path)

    t = node.get("type")
    if t == "object":
        props = node.get("properties", {})
        return {k: _gen(v, root, rng, k) for k, v in props.items()}
    if t == "array":
        lo = node.get("minItems", 1)
        hi = node.get("maxItems", max(lo, 3))
        return [_gen(node.get("items", {"type": "string"}), root, rng, path) for _ in range(rng.randint(lo, hi))]
    if t == "string":
        fmt = node.get("format")
        if fmt == "date-time":
            return f"2023-02-13T{rng.randint(6, 22):02d}:{rng.choice([0, 15, 30, 45]):02d}:00"
        if fmt == "date":
            return "2023-02-13"
        text = f"stub {path} {rng.choice(_WORDS)} {rng.choice(_WORDS)}"
        lo, hi = node.get("minLength", 0), node.get("maxLength")
        text = text.ljust(lo, "x")
        return text[:hi] if hi is not None else text
    if t == "integer":
        lo = node.get("minimum", node.get("exclusiveMinimum", 0) + 1 if "exclusiveMinimum" in node else 1)
        hi = node.get("maximum", node.get("exclusiveMaximum", lo + 10) - 1 if "exclusiveMaximum" in node else max(lo, 10))
        return rng.randint(int(lo), int(hi))
    if t == "number":
        lo = node.get("minimum", 0.0)
        hi = node.get("maximum", lo + 1.0)
        return round(rng.uniform(lo, hi), 3)
    if t == "boolean":
        return rng.random() < 0.5
    if t == "null":
        return None
    return f"stub {path}"
