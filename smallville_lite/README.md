# smallville_lite

A text-only reimplementation of *Generative Agents: Interactive Simulacra of Human Behavior*
(Park et al., UIST 2023) on the Claude API. See `../ARCHITECTURE.md` (map of the original) and
`../DESIGN.md` (this project's design).

**Status:** Phase 2 (model layer + cost controls) is done. The cognitive core is Phase 3, and the runner and web log are Phase 4.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use .venv/bin/activate elsewhere
pip install -e ".[dev]"           # core + tests (enough for --stub)
pip install -e ".[local]"         # local sentence-transformers embeddings (default for real runs)
pip install -e ".[voyage]"        # optional: Voyage AI embeddings
```

API keys come from the environment only. The config loader rejects anything that looks like a key.

```bash
export ANTHROPIC_API_KEY=...      # real LLM calls
export VOYAGE_API_KEY=...         # only if [embedding].provider = "voyage"
```

## Commands (Phase 2)

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
