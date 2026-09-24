from __future__ import annotations

import pytest

from smallville_lite.config import ConfigError, load_config

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"


def test_routing_matches_design(cfg):
    for task in ("importance", "label_actions", "react", "relationship", "summarize_dialogue", "judge"):
        assert cfg.task(task).model == HAIKU, task
    for task in ("plan_day", "plan_hours", "plan_detail", "replan", "recap_day", "reflect_questions",
                 "reflect_insights", "summary", "utterance", "conversation_note", "interview"):
        assert cfg.task(task).model == SONNET, task


def test_sonnet_tasks_set_thinking_explicitly(cfg):
    # Sonnet 5 thinks adaptively when the param is omitted; every Sonnet task must choose.
    for spec in cfg.tasks.values():
        if spec.model == SONNET:
            assert spec.thinking in ("adaptive", "disabled"), spec.name
            assert spec.temperature is None, spec.name


def test_stub_forces_stub_embedder(cfg):
    assert cfg.stub is True
    assert cfg.embedding.provider == "stub"
    assert cfg.embedding_model().id == "stub-hash-256"


def test_unknown_task_is_a_clear_error(cfg):
    with pytest.raises(ConfigError, match="unknown LLM task"):
        cfg.task("nope")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"tasks": {"utterance": {"temperature": 0.5}}}, "temperature is not supported"),
        ({"tasks": {"importance": {"effort": "low"}}}, "effort is not supported"),
        ({"tasks": {"importance": {"thinking": "adaptive"}}}, "thinking='adaptive' not supported"),
        ({"tasks": {"utterance": {"model": "claude-made-up"}}}, "is not defined under"),
        ({"tasks": {"plan_day": {"effort": "extreme"}}}, "effort must be one of"),
        ({"llm": {"cache_ttl": "10m"}}, "cache_ttl"),
        ({"budget": {"warn_fraction": 1.5}}, "warn_fraction"),
    ],
)
def test_invalid_combinations_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


def test_secrets_are_rejected():
    with pytest.raises(ConfigError, match="environment variables"):
        load_config(overrides={"llm": {"api_key": "x"}})
    with pytest.raises(ConfigError, match="looks like an API key"):
        load_config(overrides={"run": {"note": "sk-ant-api03-abcdefghijklmnop"}})


def test_config_hash_changes_with_content(cfg):
    other = load_config(overrides={"run": {"stub": True}, "budget": {"max_usd": 1.0}})
    assert cfg.config_hash != other.config_hash
