# smallville_lite

A text-only reimplementation of *Generative Agents: Interactive Simulacra of Human Behavior*
(Park et al., UIST 2023) on the Claude API. See `../ARCHITECTURE.md` (map of the original) and
`../DESIGN.md` (this project's design).

**Status:** Phases 2 (model layer + cost controls), 3 (cognitive core) and 4 (runner, checkpoint/resume, report,
web log) are done. Phase 5 adds the visual town.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use .venv/bin/activate elsewhere
pip install -e ".[dev]"           # core + tests (enough for --stub)
pip install -e ".[local]"         # local sentence-transformers embeddings (default for real runs)
pip install -e ".[voyage]"        # optional: Voyage AI embeddings
pip install -e ".[web]"           # the web log viewer
```

API keys come from the environment only. The config loader rejects anything that looks like a key.

```bash
export ANTHROPIC_API_KEY=...      # real LLM calls
export VOYAGE_API_KEY=...         # only if [embedding].provider = "voyage"
```

## Running a simulation

```bash
smallville run --scenario party --days 2 --budget 5.00 --stub      # zero-cost dry run
smallville run --scenario party --hours 6 --budget 1.00            # short live calibration run
smallville resume runs/<id> [--budget 8] [--days 3]                # continue from the latest checkpoint
smallville interview runs/<id> --agent "Klaus Mueller" [--tick 60] "Who is running for mayor?"
smallville report runs/<id> [--no-llm]                             # re-generate the report
smallville serve                                                    # viewer at http://127.0.0.1:8765
```

- **Budget:** `--budget` caps the whole run. The simulation stops at `budget - [report].reserve_usd`, and the report
  interviews use the rest. When the cap is reached, the current tick is abandoned (its cost events are kept), the last full
  tick is already checkpointed, a partial report is written from memory evidence alone, and the command exits 0. Continue
  with `resume --budget <higher>`.
- **Run directory:** `runs/<id>/` holds `events.jsonl`, `config.resolved.json`, `checkpoints/` (the last 5, plus one
  every game hour), `baseline.json` (the start-of-run "Do you know of X?" survey), `report.json` / `report.md`, and
  `interviews.jsonl`.
- **Report (paper §7.1):**
  - who knows about each fact, with every "yes" checked against the memory stream (ungrounded answers are flagged as
    hallucinated), and who heard it from whom;
  - relationship network density at the start and end;
  - who was at the party, and follow-up interviews for invitees who didn't come.
- **Viewer tabs:**
  - timeline, with emoji, label, and full action text, plus a game-time scrubber;
  - memory stream, filterable by type and subtype, with importance and last-accessed times;
  - reflection trees (each insight with the memories it cites);
  - dialogue transcripts, with any commitments made;
  - cost and usage;
  - interviews against any saved checkpoint (the App. B presets are included);
  - the report.

  A running simulation is followed live by polling the log.

## Model-layer utilities (Phase 2)

```bash
smallville check-models                     # verify configured model IDs via the live Models API
smallville llm-demo --stub                  # routing + caching + budget + usage, zero spend
smallville llm-demo --stub --calls 50 --budget 0.02   # watch the budget stop the run cleanly
smallville llm-demo --live --calls 4        # real calls, typically < $0.01
smallville usage runs/<run>/events.jsonl    # usage summary from any event log
```

## Tests

```bash
pytest                                      # offline; no network, no spend
SMALLVILLE_LIVE_TESTS=1 pytest -m live      # real API checks (< $0.01): model IDs, structured output, cache reads
```

## Model layer in one paragraph

Every LLM call goes through `LLMClient.call(task, output=PydanticModel, user=..., prefix=...)`:
- **Routing:** `config/models.toml` maps the task to a model and settings (Haiku 4.5 for importance, labels, and react
  decisions; Sonnet 5 for planning, reflection, and dialogue).
- **Output:** the response is constrained to a JSON schema derived from the Pydantic model and validated. Invalid output gets
  one repair turn, and a truncated response gets one retry with a larger `max_tokens`.
- **Budget:** the run budget is checked with a worst-case estimate *before* each request. When it's exhausted, the client
  raises `BudgetExhausted` without sending anything.
- **Logging:** each attempt is priced from actual `usage`, including cache reads and writes, and logged as an `llm_call` event.
- **Prompt caching:** the system prefix (rules + world, then the agent's cached App. A summary) carries the cache breakpoint.
  `SummaryCache` regenerates the summary only on configured events (reflection, new day).
- **Embeddings:** these go through `CachedEmbedder` (local / Voyage / stub), with an sqlite cache keyed by
  `sha256(provider|model|kind|text)`.
- **STUB mode** (`[run].stub = true` or `--stub`) swaps in a deterministic fake LLM and fake embeddings. It simulates prompt
  caching with each model's real minimum length and prices usage with the real table, so budget and usage code runs for $0.
  `llm/stub_brain.py` makes the fake scenario-aware, so a stub run exercises every mechanism: plans honour commitments,
  agents meet, news spreads, and invitations become commitments.

## Cognitive core (Phase 3)

| Module | Paper | What it does |
|---|---|---|
| `memory/stream.py`, `memory/retrieval.py` | §4.1 | Observation / reflection / plan nodes with created and last-accessed times, importance, embedding, and evidence. Retrieval = recency (0.995 per game-hour since last access) + importance + relevance (cosine), each min-max normalized, with configurable weights; it updates last-accessed |
| `cognition/perceive.py`, `importance.py` | §4.1, §5 | Observations at the agent's place (self, others, objects in unusual states), retention and attention caps, one batched importance call per agent per tick |
| `cognition/reflect.py` | §4.2 | Triggered when accumulated observation importance reaches `[reflection].threshold`: 3 questions from the 100 most recent records, retrieval per question, insights citing numbered evidence, reflection trees |
| `cognition/plan.py` | §4.3, App. A | Day plan (5-8 items), then contiguous hour blocks at known places, then 5/10/15-minute steps decomposed just in time. Replanning from *now* |
| `cognition/react.py` | §4.3.1 | "Should X react, and if so, how?" over salient observations, using the paper's two retrieval queries: talk or change plan. Objects can trigger reactions |
| `cognition/converse.py` | §4.3.2 | Turn-by-turn: each line conditioned on the speaker's summary, relationship view, memories retrieved for the last line, and the transcript. Either side can end. Afterwards: dialogue memories, notes, commitments |
| `cognition/summary.py` | App. A | Cached agent summary from three retrievals; the tail of the prompt-cache prefix |
| `cognition/interview.py` | §6, App. B | Retrieval-grounded answers; the 25 App. B questions in `scenarios/_shared/interview_presets.toml` |
| `sim/engine.py` | §5 | Two-phase tick (perceive everyone from the start-of-tick state, then decide in rotated order), travel ticks, conversation claims, night fast-forward, per-tick event commit / abort on budget |

Prompt templates live in `src/smallville_lite/llm/prompts/*.md` (versioned; `id@version` is logged on every call).
Output schemas are in `llm/schemas.py`. Place and object choices are enums built per call from what the agent knows.
