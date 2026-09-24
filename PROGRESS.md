# Progress: smallville_lite

A lightweight, text-first reimplementation of *Generative Agents: Interactive Simulacra of Human Behavior*
(Park et al., UIST 2023) on the Claude API, with a web log and a visual town.

_Last updated: 2026-09-24_

## Status at a glance

| Phase | What | Commit | State |
|---|---|---|---|
| 0 | Map the original repo → `ARCHITECTURE.md` | `32580f2` | done |
| 1 | Design → `DESIGN.md` | `0cad3ac` | done |
| 2 | Model layer + cost controls | `96d5ecd` | done |
| 3 | Cognitive core (+ tick engine, party scenario) | `ac4ce91` | done |
| 4 | Runner, checkpoint/resume, report, web log | `c3c22c2` | done |
| 5 | Visual town (replay + live), three.js 3D + SVG 2D | `4775241` | done |

- **Tests:** 110 offline tests pass and 4 are skipped. The skips are 3 live-API tests and one embedder test that's
  redundant once the embedder is installed. Run them with `cd smallville_lite && .venv/Scripts/python -m pytest`.
- **Everything so far has been built and tested in STUB mode** (fake LLM and fake embeddings), so no real Claude API
  call has been made yet.

## What exists

- **Model layer** (`llm/`):
  - one `LLMClient` for every call;
  - task→model routing in `config/models.toml` (Haiku 4.5 `claude-haiku-4-5-20251001` for importance, labels, react,
    and judge; Sonnet 5 `claude-sonnet-5` for planning, reflection, dialogue, and interviews);
  - structured JSON output validated by Pydantic, with repair and truncation retries;
  - prompt caching on the per-agent prefix (rules + world + App. A summary + seed bio, about 1.3k tokens);
  - a hard budget checked before every call, and per-task/model usage reporting;
  - embeddings: local sentence-transformers (the default), Voyage, or stub, cached in sqlite.
- **Cognition** (`cognition/`, `memory/`):
  - memory stream and retrieval (recency 0.995 per game-hour, importance, cosine relevance, min-max normalized);
  - reflection (threshold trigger, 3 questions, cited insights, reflection trees);
  - recursive planning: day → hour → 5–15 min, decomposed just in time, with replanning;
  - react decision (people and objects);
  - turn-by-turn dialogue with commitments;
  - interview mode with the App. B presets.
- **Simulation** (`sim/`):
  - a two-phase tick engine with travel, early departure, conversation claims, and night fast-forward;
  - the party scenario (4 agents, 6 places, party + mayor facts);
  - a JSONL event log with validated schemas;
  - checkpoints after every tick, with resume that's proven equivalent to an uninterrupted run.
- **Report** (`report/`, paper §7.1): fact diffusion with a hallucination check, relationship density at start and end,
  and party attendance.
- **Viewer** (`web/`): FastAPI + one static page.
  - Tabs: Town (3D/2D animated map, playback 1×–60×, live mode over SSE), Timeline, Memory, Reflections, Dialogues,
    Cost, Interview, Report.

## How to run (from `smallville_lite/`)

```bash
.venv/Scripts/smallville run --scenario party --days 2 --budget 5.00 --stub     # zero-cost run
.venv/Scripts/smallville run --stub --days 2 --budget 5 --pace 0.4              # watchable live
.venv/Scripts/smallville serve                                                   # http://127.0.0.1:8765
.venv/Scripts/smallville resume runs/<id> [--budget N] [--days N]
.venv/Scripts/smallville interview runs/<id> --agent "Klaus Mueller" [--tick N] "question"
.venv/Scripts/smallville report runs/<id> [--no-llm]
```

Existing runs in `smallville_lite/runs/` (gitignored): `party-demo` and `party-live`, both 2-day stub runs.

## Next steps

1. **Live API checks** (deferred by choice). First set the key yourself, never in chat: run
   `setx ANTHROPIC_API_KEY "sk-ant-..."`, then restart Claude Code.
   - `smallville check-models`: confirms both model IDs (free).
   - `SMALLVILLE_LIVE_TESTS=1 pytest -m live`: structured output on both models and a prompt-cache read (< $0.01).
   - A calibration run: `smallville run --scenario party --hours 6 --budget 1.00`.
2. **Use the calibration numbers** to set:
   - the reflection threshold (75 in the party scenario gives about 5 per agent per day in stub, versus 2–3 in the paper);
   - the thinking/effort settings per task;
   - the budget for a full 2-day run (stub estimate about $3.3, which leaves out thinking tokens).
3. **Check the real behavior**:
   - Do the Sonnet prompt caches actually hit?
   - Do tasks with different output schemas share one cache entry? This is the open question in DESIGN §6.3.
   - Does the party news spread and do invitees attend, with real model output?
4. **Look at the 3D view on a real GPU.** It was only verified in headless Chromium, which renders WebGL in software.

## Known issues and notes

- **Haiku 4.5 retirement:** the docs say "not sooner than Oct 15, 2026". Routing is config-only (`config/models.toml`),
  so swapping models means editing one file.
- **Haiku prompts don't cache:** its minimum cacheable length is 4096 tokens, so its calls carry the shared prefix
  uncached. That costs about $0.20 per stub day on `react`, and could be trimmed.
- **The 3D view needs the jsdelivr CDN** (three.js 0.186.0). Without it, the view falls back to 2D automatically.
- **Disk use:** checkpoints are about 0.9 MB each (about 30 MB for a 2-day run). Stub event logs are about 12 MB because
  stub mode records full prompts; live runs don't by default.
- **Open design defaults** (DESIGN §11):
  - the paper's character names are reused;
  - the "stay true to your interests" persona instruction is off;
  - all public places are known to every agent;
  - calls run sequentially.

## Environment

- The project venv is `smallville_lite/.venv` (Python 3.13). It has the extras `[dev,local,web]` installed, including
  CPU-only torch.
- The embedding cache is `smallville_lite/.cache/`. The reference clone `./reference/` is kept read-only.
- None of `reference/`, `runs/`, `.cache/`, `.venv/`, or the paper PDF is committed.
