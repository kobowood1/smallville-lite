"""Prompt templates: one versioned file per task in ``llm/prompts/`` (DESIGN §9).

File format::

    ---
    id: plan_day
    version: 1
    ---
    Body with $placeholders (string.Template syntax; $$ for a literal dollar sign).

Rendering is strict: a missing placeholder value raises, so a prompt never ships half-filled.
``prompt_id@version`` is logged on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: int
    body: str

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def render(self, **values: object) -> str:
        return Template(self.body).substitute({k: str(v) for k, v in values.items()})


@lru_cache(maxsize=None)
def load_prompt(name: str, directory: Path = PROMPT_DIR) -> PromptTemplate:
    text = (directory / f"{name}.md").read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"prompt {name}: missing front matter")
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    if meta.get("id") != name:
        raise ValueError(f"prompt {name}: front-matter id {meta.get('id')!r} does not match file name")
    return PromptTemplate(id=name, version=int(meta.get("version", 1)), body=body.strip())
