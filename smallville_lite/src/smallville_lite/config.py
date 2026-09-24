"""Load and validate configuration from TOML.

Two files live in the project's ``config/`` directory: ``default.toml`` (simulation knobs)
and ``models.toml`` (task routing, per-task limits, prices). Both are merged into one raw
dict, optional overrides are deep-merged on top, and the result is validated into frozen
dataclasses. Invalid combinations the API would reject (e.g. ``temperature`` on Sonnet 5,
``effort`` on Haiku 4.5) fail here, before any call is made.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

ThinkingMode = Literal["adaptive", "disabled", "none"]
EmbedProvider = Literal["local", "voyage", "stub"]

CONFIG_FILES = ("default.toml", "models.toml")
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Values that look like credentials must never appear in config files.
_SECRET_PATTERN = re.compile(r"(sk-ant-[A-Za-z0-9_-]{8,}|\bpa-[A-Za-z0-9_-]{20,})")
_SECRET_KEYS = {"api_key", "apikey", "anthropic_api_key", "voyage_api_key", "token", "secret"}


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    min_cache_tokens: int
    supports_temperature: bool
    supports_effort: bool
    thinking_modes: tuple[ThinkingMode, ...]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    model: str
    max_tokens: int
    max_tokens_cap: int
    thinking: ThinkingMode
    effort: str | None
    temperature: float | None


@dataclass(frozen=True)
class EmbeddingModelSpec:
    id: str
    provider: EmbedProvider
    dim: int
    price: float


@dataclass(frozen=True)
class LLMSettings:
    max_retries: int
    repair_attempts: int
    chars_per_token: float
    cache_ttl: Literal["5m", "1h"]
    request_timeout_s: float
    log_prompts: bool


@dataclass(frozen=True)
class BudgetSettings:
    max_usd: float
    warn_fraction: float


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: EmbedProvider
    local_model: str
    voyage_model: str
    stub_model: str
    cache_path: str
    batch_size: int

    @property
    def model(self) -> str:
        return {"local": self.local_model, "voyage": self.voyage_model, "stub": self.stub_model}[
            self.provider
        ]


@dataclass(frozen=True)
class Config:
    seed: int
    stub: bool
    llm: LLMSettings
    budget: BudgetSettings
    embedding: EmbeddingSettings
    summary_refresh_on: frozenset[str]
    models: Mapping[str, ModelSpec]
    tasks: Mapping[str, TaskSpec]
    embedding_models: Mapping[str, EmbeddingModelSpec]
    raw: Mapping[str, Any] = field(repr=False)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def task(self, name: str) -> TaskSpec:
        try:
            return self.tasks[name]
        except KeyError:
            raise ConfigError(f"unknown LLM task {name!r}; add it under [tasks.{name}] in models.toml") from None

    def model_for(self, task: str) -> ModelSpec:
        return self.models[self.task(task).model]

    def embedding_model(self) -> EmbeddingModelSpec:
        return self.embedding_models[self.embedding.model]


def default_config_dir() -> Path:
    """Explicit env override, else the repository's ``config/`` next to ``src/``."""
    env = os.environ.get("SMALLVILLE_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    directory = Path(config_dir) if config_dir is not None else default_config_dir()
    raw: dict[str, Any] = {}
    for name in CONFIG_FILES:
        path = directory / name
        if not path.exists():
            raise ConfigError(f"missing config file: {path}")
        with path.open("rb") as fh:
            raw = deep_merge(raw, tomllib.load(fh))
    if overrides:
        raw = deep_merge(raw, overrides)
    return parse_config(raw)


def parse_config(raw: Mapping[str, Any]) -> Config:
    _reject_secrets(raw)
    errors: list[str] = []

    models = {mid: _parse_model(mid, spec, errors) for mid, spec in _table(raw, "models", errors).items()}
    tasks = {
        name: _parse_task(name, spec, models, errors) for name, spec in _table(raw, "tasks", errors).items()
    }
    emb_models = {
        mid: _parse_embedding_model(mid, spec, errors)
        for mid, spec in _table(raw, "embedding_models", errors).items()
    }

    llm_raw = raw.get("llm", {})
    cache_ttl = llm_raw.get("cache_ttl", "5m")
    if cache_ttl not in ("5m", "1h"):
        errors.append(f"llm.cache_ttl must be '5m' or '1h', got {cache_ttl!r}")
    llm = LLMSettings(
        max_retries=int(llm_raw.get("max_retries", 2)),
        repair_attempts=int(llm_raw.get("repair_attempts", 1)),
        chars_per_token=float(llm_raw.get("chars_per_token", 3.0)),
        cache_ttl=cache_ttl,
        request_timeout_s=float(llm_raw.get("request_timeout_s", 120)),
        log_prompts=bool(llm_raw.get("log_prompts", False)),
    )
    if llm.chars_per_token <= 0:
        errors.append("llm.chars_per_token must be > 0")

    b = raw.get("budget", {})
    budget = BudgetSettings(max_usd=float(b.get("max_usd", 5.0)), warn_fraction=float(b.get("warn_fraction", 0.8)))
    if budget.max_usd < 0:
        errors.append("budget.max_usd must be >= 0")
    if not 0 < budget.warn_fraction <= 1:
        errors.append("budget.warn_fraction must be in (0, 1]")

    run = raw.get("run", {})
    stub = bool(run.get("stub", False))

    e = raw.get("embedding", {})
    embedding = EmbeddingSettings(
        provider=e.get("provider", "local"),
        local_model=e.get("local_model", "all-MiniLM-L6-v2"),
        voyage_model=e.get("voyage_model", "voyage-4-lite"),
        stub_model=e.get("stub_model", "stub-hash-256"),
        cache_path=e.get("cache_path", ".cache/embeddings.sqlite"),
        batch_size=int(e.get("batch_size", 64)),
    )
    if stub and embedding.provider != "stub":
        # Stub mode means zero network: force the stub embedder.
        embedding = EmbeddingSettings(**{**embedding.__dict__, "provider": "stub"})
    if embedding.provider not in ("local", "voyage", "stub"):
        errors.append(f"embedding.provider must be local|voyage|stub, got {embedding.provider!r}")
    elif embedding.model not in emb_models:
        errors.append(f"embedding model {embedding.model!r} has no [embedding_models] entry")
    elif emb_models[embedding.model].provider != embedding.provider:
        errors.append(
            f"embedding model {embedding.model!r} belongs to provider "
            f"{emb_models[embedding.model].provider!r}, not {embedding.provider!r}"
        )

    refresh_on = frozenset(raw.get("summary", {}).get("refresh_on", ["reflection", "new_day"]))

    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))
    return Config(
        seed=int(run.get("seed", 7)),
        stub=stub,
        llm=llm,
        budget=budget,
        embedding=embedding,
        summary_refresh_on=refresh_on,
        models=models,
        tasks=tasks,
        embedding_models=emb_models,
        raw=copy.deepcopy(dict(raw)),
    )


def _table(raw: Mapping[str, Any], key: str, errors: list[str]) -> dict[str, Mapping[str, Any]]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        errors.append(f"[{key}] must be a table")
        return {}
    return dict(value)


def _parse_model(mid: str, spec: Mapping[str, Any], errors: list[str]) -> ModelSpec:
    modes = tuple(spec.get("thinking_modes", ["none"]))
    for m in modes:
        if m not in ("adaptive", "disabled", "none"):
            errors.append(f"models.{mid}.thinking_modes: unknown mode {m!r}")
    try:
        return ModelSpec(
            id=mid,
            input=float(spec["input"]),
            output=float(spec["output"]),
            cache_write_5m=float(spec["cache_write_5m"]),
            cache_write_1h=float(spec["cache_write_1h"]),
            cache_read=float(spec["cache_read"]),
            min_cache_tokens=int(spec.get("min_cache_tokens", 1024)),
            supports_temperature=bool(spec.get("supports_temperature", True)),
            supports_effort=bool(spec.get("supports_effort", False)),
            thinking_modes=modes,  # type: ignore[arg-type]
        )
    except KeyError as exc:
        errors.append(f"models.{mid}: missing price field {exc.args[0]!r}")
        return ModelSpec(mid, 0, 0, 0, 0, 0, 0, False, False, ("none",))


def _parse_task(
    name: str, spec: Mapping[str, Any], models: Mapping[str, ModelSpec], errors: list[str]
) -> TaskSpec:
    model_id = spec.get("model", "")
    model = models.get(model_id)
    if model is None:
        errors.append(f"tasks.{name}: model {model_id!r} is not defined under [models]")
    max_tokens = int(spec.get("max_tokens", 1024))
    thinking = spec.get("thinking", "none")
    effort = spec.get("effort")
    temperature = spec.get("temperature")
    if max_tokens <= 0:
        errors.append(f"tasks.{name}: max_tokens must be > 0")
    if model is not None:
        if thinking not in model.thinking_modes:
            errors.append(
                f"tasks.{name}: thinking={thinking!r} not supported by {model_id} "
                f"(allowed: {', '.join(model.thinking_modes)})"
            )
        if effort is not None and not model.supports_effort:
            errors.append(f"tasks.{name}: effort is not supported by {model_id}")
        if temperature is not None and not model.supports_temperature:
            errors.append(f"tasks.{name}: temperature is not supported by {model_id}")
    if effort is not None and effort not in _EFFORT_LEVELS:
        errors.append(f"tasks.{name}: effort must be one of {_EFFORT_LEVELS}")
    return TaskSpec(
        name=name,
        model=model_id,
        max_tokens=max_tokens,
        max_tokens_cap=int(spec.get("max_tokens_cap", max_tokens * 2)),
        thinking=thinking,
        effort=effort,
        temperature=None if temperature is None else float(temperature),
    )


def _parse_embedding_model(mid: str, spec: Mapping[str, Any], errors: list[str]) -> EmbeddingModelSpec:
    provider = spec.get("provider", "")
    if provider not in ("local", "voyage", "stub"):
        errors.append(f"embedding_models.{mid}: provider must be local|voyage|stub")
    return EmbeddingModelSpec(id=mid, provider=provider, dim=int(spec.get("dim", 0)), price=float(spec.get("price", 0.0)))


def _reject_secrets(node: Any, path: str = "") -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            sub = f"{path}.{key}" if path else str(key)
            if str(key).lower() in _SECRET_KEYS:
                raise ConfigError(f"{sub}: API keys must come from environment variables, not config files")
            _reject_secrets(value, sub)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _reject_secrets(item, f"{path}[{i}]")
    elif isinstance(node, str) and _SECRET_PATTERN.search(node):
        raise ConfigError(f"{path}: value looks like an API key; use environment variables instead")
