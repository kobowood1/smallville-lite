# DESIGN — smallville_lite

A text-only reimplementation of the Generative Agents architecture (Park et al., UIST 2023)
on the Claude API. 3–5 agents in a small town, a JSONL event log as the single source of
truth, and a web viewer (Phase 4) plus visual town (Phase 5) that only *read* that log.

This document is Phase 1: design only, no code. It builds on `ARCHITECTURE.md`, and the
references `D1`–`D19` point to its code-vs-paper differences table.

**Guiding rules**
1. **Paper first, code second.** Where the reference code diverges from the paper
   (ARCHITECTURE §11), follow the paper unless there's a stated cost reason. Every intentional
   deviation is listed in §10.
2. **Structured output everywhere.** Every LLM call returns a Pydantic-validated JSON object
   (`client.messages.parse` / `output_config.format`). The code never parses free text.
3. **The log is the product.** Anything the viewer, the report, or a replay needs must be in
   `events.jsonl`. No viewer feature may require re-running the simulation or importing the sim core.
4. **Cheap by construction.** Batch the small calls, cache the stable prefix, merge the call chains the
   original split up, and put a hard budget in front of every call.

---

## 1. Package layout

A fresh project in `./smallville_lite` (a `src/` layout, not a fork of the original):

```
smallville_lite/
├── pyproject.toml              # deps + extras; console script `smallville`
├── README.md
├── config/
│   ├── default.toml            # sim knobs (time, retrieval, reflection, dialogue, budget …)
│   └── models.toml             # task → model routing, per-task limits, price table
├── scenarios/
│   └── party/
│       ├── scenario.toml       # start time, days, world ref, event windows for the report
│       ├── world.toml          # location tree (places → areas → objects, initial states)
│       ├── agents.toml         # 4 agents: identity + semicolon-delimited bios
│       ├── questions.toml      # report questions (diffusion, "do you know of …")
│       └── layout.toml         # Phase 5 map layout (x, y, w, h per place; object icon positions)
├── src/smallville_lite/
│   ├── config.py               # TOML → frozen dataclasses; validation; config hash
│   ├── clock.py                # GameClock: tick ↔ datetime, day boundaries
│   ├── world/
│   │   ├── tree.py             # LocationNode tree, object state, NL rendering (§5.1)
│   │   └── state.py            # WorldState: positions, transit, current actions, object states
│   ├── memory/
│   │   ├── node.py             # MemoryNode dataclass
│   │   ├── stream.py           # MemoryStream: append-only store, importance accumulator
│   │   └── retrieval.py        # recency / importance / relevance scoring (pure functions)
│   ├── cognition/
│   │   ├── perceive.py         # observations at the current location
│   │   ├── importance.py       # batched importance scoring
│   │   ├── summary.py          # App. A cached agent summary
│   │   ├── plan.py             # day → hour → 5–15 min, just-in-time; replanning
│   │   ├── react.py            # continue-or-react decision
│   │   ├── converse.py         # turn-by-turn dialogue session
│   │   ├── reflect.py          # trigger, questions, retrieval, insights with evidence
│   │   └── interview.py        # retrieval-grounded Q&A (App. B presets)
│   ├── agent.py                # Agent = identity + MemoryStream + Scratch (short-term state)
│   ├── llm/
│   │   ├── client.py           # the single LLM entry point (routing, caching, budget, logging)
│   │   ├── budget.py           # BudgetGuard (pre-call estimate, settle, hard stop)
│   │   ├── pricing.py          # usage → dollars
│   │   ├── stub.py             # deterministic FakeLLM (schema-aware)
│   │   ├── schemas.py          # Pydantic output models, one per task
│   │   └── prompts/            # one template per task (*.md, `$placeholders`, versioned)
│   ├── embed/
│   │   ├── base.py             # Embedder protocol
│   │   ├── local.py            # sentence-transformers (zero-cost default)
│   │   ├── voyage.py           # Voyage AI
│   │   ├── stub.py             # deterministic hashed bag-of-words vectors
│   │   └── cache.py            # sqlite cache keyed by sha256(provider|model|text)
│   ├── sim/
│   │   ├── engine.py           # tick loop, two-phase tick, conversation arbitration
│   │   ├── scenario.py         # load scenario, seed agents
│   │   ├── events.py           # event models + EventLog writer (per-tick buffer)
│   │   └── checkpoint.py       # save / resume
│   ├── report/
│   │   └── diffusion.py        # post-run report (§7)
│   ├── web/                    # Phase 4/5
│   │   ├── server.py           # FastAPI: read log, SSE tail, interview endpoint
│   │   └── static/index.html   # single page, no build step
│   └── cli.py                  # run / resume / interview / report / serve
└── tests/
```

**Dependencies (minimal):**
- Core: `anthropic`, `pydantic>=2`, `numpy`. TOML comes from the stdlib `tomllib` and the CLI from `argparse`.
- Extras: `[local]` = `sentence-transformers`, `[voyage]` = `voyageai`, `[web]` = `fastapi`, `uvicorn`, `[dev]` = `pytest`.

The core install has no torch, and **STUB mode needs only the core**. The local embedder
is the default for real runs. If the `[local]` extra isn't installed, the CLI exits with the
install command; it never falls back silently.

**Secrets:** `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` come from the environment only. The config
cannot contain keys, and a key-looking value in TOML fails validation.

---

## 2. World model

### 2.1 Location tree (paper §5.1)
A tree of `LocationNode { id, name, kind: town|place|area|object, children, state?, desc? }`.
Depth is free-form (town → place → optional area → object), but **movement cost exists only
between top-level places**. Moving between areas or objects inside a place is instant. There are
no coordinates and no pathfinding.

Party-scenario town (`world.toml`), with 6 places:

| Place | Areas / objects (initial state) |
|---|---|
| **Hobbs Cafe** | *cafe floor*: counter (idle), espresso machine (off), tables, bulletin board (empty); *Isabella's apartment*: bed, desk, kitchenette |
| **Johnson Park** | garden beds (tidy), bench, pond path |
| **The Willows Market** | shelves (stocked), checkout counter, community notice board |
| **Oak Hill College Library** | reading tables, bookshelves, study carrels |
| **Oak Hill Dorm** | *Maria's room*: bed, desk; *Klaus's room*: bed, desk; common room: sofa, TV |
| **Moore House** | bedroom: bed; kitchen: stove, fridge; living room: armchair, bookshelf |

- Object **state** is a short phrase (`"brewing coffee"`, `"idle"`). It changes when an agent's
  action names that object (the new state comes from the same planning call that produced the action,
  so there's no extra call), and it resets to the default when the action ends.
- **Travel time:** `travel_ticks_default = 1`. An optional `travel_ticks` matrix per place pair
  can override it. While travelling, an agent is `in_transit(from, to, arrive_tick)`: it can't be
  perceived, perceives nothing, and can't be talked to.

### 2.2 Per-agent spatial memory
Each agent holds the subtree it knows (paper §5.1):
- every place marked `public = true` in `world.toml`;
- its own home;
- any extra private places listed in `known_places` in `agents.toml`;
- all children of those places.

Visiting a place adds it. Places heard about in conversation are **not** auto-added in v1 (see §11 Q3).

### 2.3 Rendering to natural language
`render_known_world(agent)` flattens the agent's subtree as §5.1 describes, e.g.:

```
Isabella Rodriguez knows these places:
- Hobbs Cafe: cafe floor (counter, espresso machine, tables, bulletin board); Isabella's apartment (bed, desk, kitchenette)
- Johnson Park: garden beds, bench, pond path
…
Isabella Rodriguez is currently at Hobbs Cafe, cafe floor. There is a counter, an espresso machine that is brewing coffee, …
```

Places and objects are always passed to the LLM as **enumerations** (a JSON-schema `enum` built per
call from the agent's known tree). An invalid location is then impossible, which removes the
original's sector → arena → object call chain and its hard-coded fallbacks (ARCHITECTURE §8, #5–7).

---

## 3. Time model

- **Tick:** `tick_minutes = 10` by default (configurable; 5 gives finer timing). Tick 0 is the
  scenario's `start` (party: `2023-02-13T06:00`). `--days N` runs to `start_day + N` at 00:00.
- **Plan resolution is minutes, simulation resolution is ticks.** Plan items keep exact start
  times and durations (5–15 min chunks). At each tick, every plan item whose start falls inside
  `(t_prev, t]` produces an `action_started` event stamped with its **exact** game time. The
  agent's current action is the last of those items. A 5-minute subtask inside a 10-minute tick is still logged,
  so the viewer can animate at minute resolution.
- **Recency clock:** retrieval recency uses game time (hours since `last_accessed`), not ticks
  (fixes D1 and D18).
- **Night fast-forward:** if every agent is asleep and no conversation or reaction is pending, the engine jumps
  to the earliest wake-up without calling the LLM and logs one `time_skipped` event.

### 3.1 Two-phase tick
The tick is split in two phases so that results don't depend on agent iteration order and
conversations don't double-book agents:

```
tick t (game time T)
 ├─ 0. advance transit: agents with arrive_tick == t are placed; emit move_arrived
 ├─ 1. PERCEIVE (all agents, against the start-of-tick snapshot)
 │     observations → memory (batched importance) → reflection accumulator
 ├─ 2. DECIDE (agents in fixed order: sorted by name, rotated by t for fairness)
 │     a. new day?          → daily planning (§5.4)
 │     b. reflect?          → reflection (§5.6) + summary refresh
 │     c. react?            → react decision over this tick's salient observations (§5.5)
 │          talk            → claim partner (first claim wins) → dialogue session (§5.7)
 │          change_plan     → replan from T
 │     d. action due?       → JIT decomposition if needed (§5.4) → next action
 │     e. place ≠ action place → start travel
 ├─ 3. COMMIT: apply moves/object states; emit events; flush tick buffer; checkpoint
```

- Everything an agent sees in step 1 comes from the **snapshot**. Actions and moves decided in step 2
  commit at step 3, so an agent never reacts to something another agent decided earlier in the same tick.
- **Conversation claims:** when A decides to talk to B, B is marked *engaged* for the conversation's
  duration and skips its own step 2c–e. A conversation needs both agents at the same place, both awake, neither
  in transit or engaged, and the pair's cooldown (`pair_cooldown_minutes = 180`) expired. The cooldown
  is the lite version of the original's `chatting_with_buffer`.
- A dialogue takes `ceil(turns × minutes_per_utterance / tick_minutes)` ticks
  (`minutes_per_utterance = 1`). Both agents' action is "chatting with X about …" for that span, and their
  plans resume afterwards (§5.7).

---

## 4. Agents and the seed scenario

### 4.1 Agent definition (`agents.toml`)
```toml
[[agent]]
name = "Isabella Rodriguez"
age = 34
traits = "friendly, outgoing, hospitable"        # innate traits (App. A)
home = "Hobbs Cafe/Isabella's apartment"
start = "Hobbs Cafe/Isabella's apartment"
known_places = []                                 # extra private places; public places + home are implicit
wake_hint = "wakes up around 6am, goes to bed around 11pm"   # soft; the planner may deviate
bio = """Isabella Rodriguez is the owner of Hobbs Cafe who loves to make people feel welcome; …"""
```

### 4.2 Seeding (paper §3.1 and §5)
At tick 0, **each semicolon-delimited phrase in `bio` becomes one memory node**:
- type `observation`, subtype `seed`, `created = start`;
- importance comes from one batched Haiku call per agent.

There's no LLM rewrite of the phrases (the original's `whisper_inner_thought` step, D15, is dropped). Then:
1. the initial summary (App. A, §5.3);
2. the day 1 plan.

Optional **inner-voice injection** (§3.1.2): `smallville whisper <run> "<agent>" "<text>"` appends
an `observation/inner_voice` node before the next tick. It's useful for experiments, but no v1 feature depends on it.

### 4.3 Party scenario: 4 agents, mirroring the §7.1 diffusion test

| Agent | Role | Seeded secret | Starting relationships |
|---|---|---|---|
| **Isabella Rodriguez** (34) | owner of Hobbs Cafe, lives above it | **Plans a Valentine's Day party at Hobbs Cafe, Feb 14, 5–7 pm, and wants to invite everyone** | knows Maria well (regular); knows Klaus by sight |
| **Sam Moore** (65) | retired navy officer; gardens in Johnson Park, shops at the Willows Market, likes coffee | **Has decided to run for mayor in the local election and wants to tell neighbors** | knows no one well yet (new to the neighborhood) |
| **Maria Lopez** (21) | physics student; studies at Hobbs Cafe; lives in Oak Hill Dorm | secret crush on Klaus (from §3.4.3) | close friends with Klaus; friendly with Isabella |
| **Klaus Mueller** (20) | sociology student researching gentrification; works in the library; lives in Oak Hill Dorm | — | close friends with Maria; knows Isabella as the cafe owner |

The names echo the paper's characters so results can be compared with §7.1. The bios are written from
scratch; nothing is copied from the original repo's assets.

**The setup gives measurable diffusion and relationship formation:**
- At t0 only Isabella knows about the party (1/4) and only Sam knows about the candidacy (1/4).
- The mutual-knowledge graph starts at 3 of 6 pairs (I–M, M–K, K–I), so **density 0.50**. Sam is the only
  isolated agent, so every edge he gains is a relationship formed during the run.
- Paths cross naturally: Sam gets coffee at Hobbs Cafe and walks in Johnson Park; Maria studies at the
  cafe; Klaus passes through the park to the library.

`scenario.toml` records the ground truth for the report:
```toml
[[facts]]  id = "party"   origin = "Isabella Rodriguez"  question = "Did you know there is a Valentine's Day party?"
[[facts]]  id = "mayor"   origin = "Sam Moore"           question = "Do you know who is running for mayor?"
[[events]] id = "party"   place = "Hobbs Cafe"  start = "2023-02-14T17:00"  end = "2023-02-14T19:00"
```

The reflection threshold for this scenario is **75**, not the paper's 150 (see §5.6).

---

## 5. Cognitive core

### 5.1 Memory node
```python
@dataclass
class MemoryNode:
    id: str                 # "{agent_slug}:{n}", unique across the run
    agent: str
    type: Literal["observation", "reflection", "plan"]
    subtype: str            # seed | perception | dialogue | inner_voice | insight | conversation_note | day_plan | hour_plan | commitment
    description: str        # the text that is embedded and shown
    created: datetime       # game time
    last_accessed: datetime # game time; initialized to created
    importance: int         # 1–10
    embedding_key: str      # sha256 of (provider|model|description) → vector in the embed cache
    evidence: list[str]     # node ids (reflections, conversation notes, revised plans)
    depth: int              # 0 for observations/plans; 1 + max(depth of evidence) for reflections
    meta: dict              # place, participants, conv_id, plan_item_id, supersedes, …
```
The stream is append-only in memory and in the log (`memory_added`). The `last_accessed` updates go
to the log as `memory_accessed` events.

### 5.2 Retrieval (paper §4.1)
The scored retrieval is the only retrieval function, and it runs on every path, including the react path (fixes D5):

```
candidates = all nodes of the agent (observations, reflections, plans), optional type filter
recency_i    = decay ** hours_since(last_accessed_i)          # decay = 0.995 / game-hour
importance_i = importance_i
relevance_i  = cosine(embed(query), embed(node_i))
each component min-max normalized to [0,1] over the candidates (all-equal → 0.5)
score_i = w_rec·rec_i + w_imp·imp_i + w_rel·rel_i                # defaults all 1.0, no hidden gw (fixes D3)
top_k by score (default 15, per-call override) → last_accessed ← now for returned nodes
```
- The query embedding is cached by text hash, so a repeated query doesn't call the provider.
- Dialogue memories are included, because conversations are observation nodes (fixes D6).
- Each retrieval emits `memory_accessed` with the per-component scores for the viewer's "why was this
  retrieved" panel (verbosity-gated, on by default).

### 5.3 Agent summary: the App. A cache
- **Content:** three scored retrievals, for "`[name]`'s core characteristics", "`[name]`'s current
  daily occupation", and "`[name]`'s feeling about their recent progress in life". They feed one
  Sonnet call that returns `{core, occupation, recent_progress}`. The cached summary is
  `Name (age) · traits` followed by those three paragraphs (D12 fixed; one call instead of three,
  see §10).
- **Refresh:** at tick 0, at the start of each game day, and after each reflection. Never per call.
  Emits `summary_updated {version}`.
- **Prompt-cache role:** the summary is the tail of each agent's stable prefix (§6.3).

### 5.4 Planning (paper §4.3, App. A)
| Level | When | Call | Output schema |
|---|---|---|---|
| **Day** | start of each game day (and at tick 0) | Sonnet `plan_day` | `{wake_time, items: [{start, description}] (5–8)}` |
| **Hour** | right after the day plan | Sonnet `plan_hours` | `{blocks: [{start, end, activity, place (enum)}]}` covering wake → sleep |
| **5–15 min** | just in time: when the current hour block has no decomposition, looking `decompose_horizon_minutes = 60` ahead | Sonnet `plan_detail` | `{steps: [{minutes: 5\|10\|15, activity, object (enum of place objects), object_state}]}` |

- **Day-plan inputs:** the summary; *yesterday's recap*, a 3–5 sentence Sonnet `recap_day` call on
  that day's top-retrieved memories, run at day rollover (skipped on day 1); and the memories retrieved for
  "`[name]`'s plans and commitments for `<date>`". That last retrieval is how a party invitation
  accepted yesterday gets into today's plan (fixes D10).
- **Validation (semantic):**
  - blocks are contiguous and inside the day;
  - detail step minutes sum to the block length (the schema can't express numeric bounds, so minutes use an enum and the code checks the sum);
  - places and objects come from the enums.
  On failure: one repair retry with the validation error appended, then a deterministic fallback
  (one step for the whole block) with `llm_call.fallback` set.
- **Emoji and short label:** one batched **Haiku** `label_actions` call per `plan_detail` output returns
  `{labels: [{emoji (≤2), label (≤4 words)}]}` for every step. The viewer needs both, and the
  user's routing puts this task on Haiku.
- **Memory:** each day-plan item and each hour block becomes a `plan` node. Their importance comes from one batched Haiku call.
  Revised items get new nodes with `meta.supersedes`. Fine-grained steps stay in scratch only, as in the paper's
  JIT description (they'd flood the stream).
- **Sleep:** the block that starts at bedtime is never decomposed.
- **Replanning** (`replan`, Sonnet): input is the remaining hour blocks from *now*, the reaction or
  commitment that caused the replan, and the known places. Output is new hour blocks from now to end of day.
  Pending detail steps after *now* are discarded (JIT regenerates them). Emits `plan_revised`.

### 5.5 Perceive → continue or react (paper §4.3.1)
**Perceive** (at the agent's place, against the snapshot):
- **Observation sources:** each co-located agent's current action ("Maria Lopez is studying for her physics exam
  at a table"); object states that differ from their defaults; arrivals and departures.
- **New vs. repeated:** an observation is *new* if its `(subject, text)` isn't among the agent's last
  `retention = 10` observations.
- **Cap:** at most `att_bandwidth = 5` per tick, co-located agents first.
- **Storage:** new observations become nodes. Their importance comes from **one batched Haiku `importance`
  call per agent per tick**, and the sum is added to the reflection accumulator.

**Decide** (one **Haiku `react`** call per agent per tick, and only if some new observation is *salient*, i.e.
it involves another agent or a non-default object state):
- **Context, per salient observation:** the retrieval queries the paper names, "What is `[agent]`'s relationship
  with `[observed]`?" and "`[observed]` is `[status]`", with top-5 each. The retrieved memories are passed as a
  list; there's no separate summarization call (§10).
- **Prompt:** the paper's wording: "…Should `[agent]` react to the observation, and if so, what would be an
  appropriate reaction?"
- **Schema:** `{react: bool, observation_index: int|null, kind: "talk"|"change_plan"|null, reaction: str, reason: str}`.
  `talk` starts a dialogue (subject to §3.1 gating). `change_plan` calls `replan` with the reaction as the
  inserted activity. Objects can trigger reactions, so the burning-stove case works (fixes D11).

### 5.6 Reflection (paper §4.2)
- **Trigger:** the accumulator sums the importance of new *observations* since the last reflection. When it reaches
  `reflection.threshold`, a reflection runs. The default is 150 as in the paper; the party scenario uses **75** because it
  has 4 agents with 10-minute ticks, which is far fewer observations than 25 agents on 10-second
  steps. Phase 3 calibrates it in stub mode and the first real run checks it against the paper's "2–3 reflections a day".
- **Questions:** Sonnet `reflect_questions` gets the `recent_records = 100` most recent nodes and the paper's
  wording, "Given only the information above, what are 3 most salient high-level questions we can answer
  about the subjects in the statements?", and returns `{questions: [str ×3]}`.
- **Per question:** retrieve top-`k` (default 15) → Sonnet `reflect_insights` with the numbered statements and
  the paper's wording, "What 5 high-level insights can you infer from the above statements? (example format:
  insight (because of 1, 5, 3))". It returns `{insights: [{insight, evidence: [int]}]}`.
  - Indices are validated against the numbered list, and out-of-range ones are dropped. If none remain, the insight is dropped.
  - Evidence maps to node ids. `depth = 1 + max(evidence depth)`, which builds the reflection tree.
- **Insight importance:** one batched Haiku call. Then: reset the accumulator, refresh the summary, emit `reflection`.
- **Conversation notes** (the original's post-chat thoughts, which help coordination; §5.7): stored as
  `reflection/conversation_note` with evidence = the dialogue node. They don't count toward the trigger.

### 5.7 Dialogue (paper §4.3.2)
A dialogue session runs inside the initiator's step, turn by turn:

1. **Setup:** each participant gets one relationship summary, "What does `[self]` know and feel about `[other]`?"
   (retrieval top-10 → Haiku `relationship`). It's computed once per conversation per participant, not every
   turn (fixes D13's cost).
2. **Each turn, for the speaker:**
   - retrieve top-8 for the query "`[other]` said: `<last utterance>`" (first turn: the reaction text);
   - Sonnet `utterance` with the cached prefix (world + summary), status, observation, relationship summary,
     retrieved memories, and the transcript so far;
   - output `{utterance, end_conversation: bool}`.
   The listener's first turn is its decision to respond: it may reply briefly and end. Either side can end,
   and there's a hard stop at `max_turns = 12`.
3. **Every utterance is logged** as its own `utterance` event with a game timestamp
   (`start + i·minutes_per_utterance`). Phase 5 speech bubbles depend on this.
4. **After the conversation:**
   - Haiku `summarize_dialogue` returns `{topic_label, summary}`. The summary is told to preserve plans,
     invitations, news, dates, times, and places verbatim.
   - For each participant, a `observation/dialogue` node is written, with description "Conversation with X at
     `<place>`: `<summary>`" and the transcript in `meta.transcript`.
   - Sonnet `conversation_note` (per participant) returns `{note, commitments: [{what, when, where}], replan: bool}`.
     Each commitment becomes a `plan/commitment` node. If `replan` is true or a commitment falls today, `replan` runs.
     This is how "I'll come to the party tomorrow at 5" survives until the next day's planning.

### 5.8 Interview (paper §6, App. B)
`interview(agent, question, as_of=checkpoint)` retrieves top-15 for the question and makes a Sonnet
`interview` call with the cached prefix, which returns `{answer, cited: [int]}`.
- **Answers are not written to memory by default**, matching the paper's evaluation setting. A `--remember` flag
  can turn that on.
- **App. B presets** ship in `scenarios/_shared/interview_presets.toml`: all 25 questions in the 5 categories.
  Bracketed names are filled with most-interacted agents, as the paper describes.
- The original's anthropomorphization safety filter is dropped. This is a local research tool.

---

## 6. Model layer (the Phase 2 contract, fixed here)

### 6.1 Routing (`config/models.toml`)
| Task | Model | Why |
|---|---|---|
| `importance`, `label_actions`, `react`, `relationship`, `summarize_dialogue`, `judge` (report) | **Haiku** `claude-haiku-4-5-20251001` | short, high-volume classification or extraction (user routing) |
| `plan_day`, `plan_hours`, `plan_detail`, `replan`, `recap_day`, `reflect_questions`, `reflect_insights`, `summary`, `utterance`, `conversation_note`, `interview` | **Sonnet** `claude-sonnet-5` | planning, reflection, dialogue (user routing) |

- **Per-task settings:** `model`, `max_tokens`, `thinking` (`disabled` or `adaptive`), and `effort`.
  - Sonnet 5 rejects `temperature`/`top_p` and assistant prefill, and runs *adaptive thinking when `thinking` is
    omitted*. So every Sonnet task sets thinking explicitly. Starting defaults: `disabled` for `utterance`,
    `label`-like, and summary tasks; `adaptive` with effort `low` for plans and reflections. Phase 2 tunes these against measured cost.
  - Haiku tasks may set `temperature`.
- **Model IDs:** the IDs above are the user's. The current docs list Haiku 4.5 as `claude-haiku-4-5` (alias) and
  Sonnet 5 as `claude-sonnet-5`. Phase 2 checks both against the Models API before first use and fails fast on an unknown ID.

### 6.2 Structured output
Every task has a Pydantic model in `schemas.py`, sent through `client.messages.parse(output_format=Model)`.
Structured outputs support enums, `anyOf`, `$ref`, and `additionalProperties: false`, but **not numeric bounds**.
So bounded integers use `enum` (importance 1–10, minutes 5/10/15), and cross-field rules (sums, index ranges)
are checked in code with one repair retry.

**Per-call handling:**
- Check `stop_reason` before reading.
- `max_tokens` or `refusal` → one retry, then a logged fallback.
- The SDK's own retries (`max_retries = 2`) cover 429/5xx/connection errors.

### 6.3 Prompt caching
The request layout keeps a byte-stable prefix per agent:

```
system[0]  RULES + WORLD      (static per run: simulation rules, town directory, agent roster one-liners)
system[1]  AGENT SUMMARY      (changes only on summary refresh)   ← cache_control: ephemeral (5-min TTL)
messages   TASK               (volatile: time, observations, retrieved memories, transcript, instructions)
```

**Constraints from the current docs:**
- The minimum cacheable prefix is **1024 tokens on Sonnet 5 and 4096 on Haiku 4.5**. Shorter prefixes silently
  don't cache. The Sonnet prefix is designed to exceed 1024 with useful content: the full town directory and a
  richer roster, not filler. **Haiku calls will generally not cache.** Batching those calls (one call per agent per tick)
  is what keeps them cheap.
- Caches are per model, and any byte change invalidates everything after it. No timestamps, dicts, or unsorted data
  go in `system`. The current time and all volatile data live in `messages`.
- **TTL:** each agent makes Sonnet calls well within 5 minutes of wall time while a run is active, so the default
  5-minute TTL (1.25× write, 0.1× read) is right. A 1-hour TTL isn't needed.
- **Unknown to verify in Phase 2:** whether differing `output_config.format` schemas across tasks share one cached
  prefix. If they don't, each task type keeps its own warm entry. It still pays off, because every task recurs every
  few ticks. The `llm_call` events record `cache_read_input_tokens` and `cache_creation_input_tokens`, and Phase 2
  adds a test that a repeated call shows `cache_read > 0`.

### 6.4 Budget
`BudgetGuard(max_usd)` runs before every LLM and embedding call:
- estimate = `ceil(prompt_chars / 3.5)` × input price + `max_tokens` × output price (worst case).
- If `spent + estimate > max_usd`, it raises `BudgetExhausted` **before sending**. After the call, the
  reservation is settled from actual `usage`.
- `budget_warning` fires once at 80%.

**On `BudgetExhausted`:** the engine abandons the current tick and still flushes its `llm_call`/`embed_call`
events, because that money was spent (they're marked `aborted_tick: true`). It discards the tick's other events,
keeps the checkpoint of the last completed tick, writes `run_ended {reason: "budget"}`, and exits 0.
`smallville resume <run> --budget 8` continues from that tick.

### 6.5 Embeddings
`Embedder` protocol: `name`, `dim`, `embed(texts) -> np.ndarray`.

| Implementation | Details |
|---|---|
| **local** (default) | sentence-transformers, `all-MiniLM-L6-v2` (384-d), CPU, $0 |
| **voyage** | model set in config; the current model name and price are checked in Phase 2 |
| **stub** | hashed bag-of-words → L2-normalized vector. Deterministic, and texts that share words land close together, so retrieval tests still mean something |

The cache is `.cache/embeddings.sqlite`, keyed by `sha256(provider|model|text)` → float32 blob. It's shared
across runs, and every call records cache hits and misses in `embed_call`.

### 6.6 STUB mode (`--stub`)
`FakeLLM` implements the same `parse(task, prompt, schema)` interface and never touches the network.
- **Output:** schema-valid output derived from `hash(task, prompt)` plus scenario-aware templates:
  - day plans built from the agent's bio phrases and known places;
  - utterances that repeat a seeded fact with fixed probability, so diffusion actually happens in tests;
  - reflections that cite valid indices.
- **Usage:** token counts are `len/4`, priced with the real table, so the budget, usage summary, and cost panel all
  run end to end with $0 spent.
- Together with the stub embedder, a full run is deterministic for a given seed.

### 6.7 Usage summary
At run end, and in `report.json`: calls, input/output/cache tokens, cost, and p50/p95 latency, **by task and by
model**, plus embedding calls and hit rates. The summary is printed to the terminal and computed from the
`llm_call` events alone, so the viewer can recompute it.

**Rough cost expectation** (to be replaced by Phase 2 measurements): a 2-day, 4-agent party run is roughly
~400 Sonnet calls (≈2.5k in / 300 out each, partly cached) and ~500 Haiku calls. That's on the order of
**$3–6**, which is why the `--budget 5.00` example is realistic but tight. The biggest levers if it's over:
`tick_minutes`, `decompose_horizon_minutes`, `max_turns`, and thinking settings.

---

## 7. Post-run report (defined now so the log carries what it needs)
Written to `runs/<id>/report.md` and `report.json`:
- **Information diffusion** (§7.1): at the end, interview every agent with each fact's question. Haiku `judge`
  labels each answer yes/no, and a **grounding check** requires a `dialogue` or `seed` memory whose text mentions
  the fact. An answer judged "yes" without grounding is reported as *hallucinated*, as the paper does.
  - The **diffusion path** is the first dialogue in which each agent heard the fact, taken from the transcripts. It's
    presented as a table here and as a graph in Phase 5.
- **Relationships** (§7.1): "Do you know of `<name>`?" for every ordered pair at **t0 and at the end**. An edge
  exists if both directions are "yes". Density is η = 2|E| / (|V|(|V|−1)), reported for start → end.
- **Coordination:** agents present at Hobbs Cafe for ≥ 1 tick during the party window (from the move and position
  events), cross-tabulated with whether they knew about the party. Invited-but-absent agents get a follow-up interview
  ("Why didn't you go to the party?"), mirroring §7.1.2.
- Plus the usage summary.

---

## 8. Event log (JSONL)

`runs/<run_id>/events.jsonl`, one JSON object per line, append-only. It's the only interface to the viewer,
the report, and the Phase 5 renderer.

### 8.1 Envelope
```json
{"seq": 1042, "run_id": "party-20260923-1412", "segment": 0, "tick": 57,
 "game_time": "2023-02-13T15:30:00", "wall_time": "2026-09-23T14:31:07.221Z",
 "type": "utterance", "agent": "Isabella Rodriguez", "data": { … }}
```
- `seq` is strictly increasing within a run.
- `segment` increments on each resume. Readers ignore events from an earlier segment whose `tick` is ≥ the resume
  tick, which is how aborted ticks disappear.
- `game_time` is the event's own time, which may be minute-precise inside the tick.
- Every event type has a Pydantic model in `sim/events.py`. The writer validates against it, and the JS viewer uses a
  generated JSON schema.

### 8.2 Event types (`data` fields)

**Run lifecycle**
| type | data |
|---|---|
| `run_started` | scenario, config (full resolved), config_hash, seed, stub, models, git_commit, budget_usd |
| `run_resumed` | from_tick, segment, checkpoint, budget_usd |
| `run_ended` | reason: `completed\|budget\|error\|interrupted`, last_tick, spent_usd |
| `checkpoint_saved` | tick, path |
| `time_skipped` | from_tick, to_tick, reason: `all_asleep` |

**World and agents**
| type | data |
|---|---|
| `world_init` | tree (full), layout (if present), travel matrix |
| `agent_init` | identity, bio phrases, home, start place, known_places |
| `state_snapshot` | every `snapshot_every_ticks = 6`: per agent {place, area, in_transit, action_id, description, emoji, label}, object states, engaged pairs. Lets the Phase 5 scrubber seek fast |
| `action_started` | action_id, description, emoji, label, place, area, object, object_state, start, duration_min, kind: `planned\|reaction\|chat\|wait\|sleep`, plan_block_id |
| `move_started` | from, to, depart_tick, arrive_tick |
| `move_arrived` | place |
| `object_state_changed` | object_path, state, by (agent or null for reset) |

**Memory and cognition**
| type | data |
|---|---|
| `memory_added` | the full MemoryNode except the vector (includes importance, evidence, depth, embedding_key, meta) |
| `memory_accessed` | query, purpose (`react\|dialogue\|reflect\|plan\|summary\|interview`), results: [{node_id, score, recency, importance, relevance}] |
| `perceived` | observations: [{node_id, text, salient}] |
| `react_decision` | observation_node_id, react, kind, reaction, reason, llm_call_id |
| `plan_created` | level: `day\|hour\|detail`, date, items: [{id, start, end, description, place, object}], llm_call_id |
| `plan_revised` | reason, cause (node or conv id), from_time, removed_ids, added: [...] |
| `reflection` | trigger_sum, questions, insights: [{node_id, text, evidence}], llm_call_ids |
| `summary_updated` | version, text, source_node_ids |

**Dialogue**
| type | data |
|---|---|
| `conversation_started` | conv_id, participants, initiator, place, reason (react text) |
| `utterance` | conv_id, turn, speaker, listener, text, end_conversation, retrieved_node_ids, llm_call_id |
| `conversation_ended` | conv_id, turns, ended_by: `speaker\|max_turns`, topic_label, summary, duration_ticks |

**Costs and tools**
| type | data |
|---|---|
| `llm_call` | call_id, task, model, agent, prompt_id@version, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, latency_ms, stop_reason, attempt, ok, fallback, error, aborted_tick; plus `prompt` and `response` text when `log_prompts = true` (off by default to keep logs small, and always on in stub mode) |
| `embed_call` | provider, model, n_texts, n_cached, tokens, cost_usd, latency_ms |
| `budget_warning` / `budget_exhausted` | spent_usd, limit_usd, next_estimate_usd |
| `interview` | agent, question, answer, cited_node_ids, as_of_tick, source: `cli\|web\|report`, llm_call_id |

**Cross-references:** cognitive events carry `llm_call_id`(s) and memory node ids, so the viewer can go from any
behavior to its prompt, cost, and cited memories.

### 8.3 Replay contract (Phase 4 and 5 depend on this)
- **Reconstructing state:** the state at any tick is the nearest `state_snapshot` ≤ tick, then folding later
  `action_started`, `move_*`, and `object_state_changed` events. The memory stream at tick *T* is every
  `memory_added` with tick ≤ *T*, with `last_accessed` from `memory_accessed`.
- **Shared JS module:** one `reduceState(events, tick)` module feeds the timeline, the memory inspector, and the
  map renderer. The Phase 5 renderer interface is `render(state, prevState, dt)` behind a `Renderer` interface, with SVG
  first and three.js later.
- **Live mode** (Phase 5): the server tails `events.jsonl` and streams new lines over SSE. The writer flushes once per
  committed tick, so the viewer never sees a half tick.

### 8.4 Other run artifacts
```
runs/<run_id>/
  events.jsonl
  config.resolved.toml
  checkpoints/tick_<n>.json   # world state, agent scratch, full memory nodes (vectors by key), RNG state, budget spent
  checkpoints/latest          # pointer
  report.md / report.json     # Phase 4
```
- Checkpoints are written at every tick commit (the state is small), and only the last `keep_checkpoints = 5` are kept.
- **Resume:** load `latest`, append `run_resumed`, continue.
- **The interview "against saved memory state"** loads any checkpoint read-only. It never mutates the run.

---

## 9. Prompts: port or rewrite

"Port" means keeping the paper's (or the original's) wording for the core instruction while switching the output
to a JSON schema. "Rewrite" means a new prompt.

| Task (ours) | Replaces (original, ARCHITECTURE §8) | Decision | Notes |
|---|---|---|---|
| `importance` | #16 poignancy_event, #17 poignancy_chat | **Port** (paper §4.1 wording) + batch | `{scores: [1..10 enum]}` for N memories in one call |
| `label_actions` | #8 pronunciatio | **Port** idea, batch | emoji ≤2 + short label for all steps of a decomposition |
| `plan_day` | #1 wake_up_hour, #2 daily_plan, #28–31 revise_identity | **Rewrite** around the paper's §4.3 Eddy prompt | adds yesterday's recap + retrieved commitments; wake time in the same call |
| `recap_day` | revise_identity notes (#28–29) | **Rewrite** | one call at day rollover |
| `plan_hours` | #3 hourly schedule (1 call per hour) | **Rewrite** | one call per day, places from an enum |
| `plan_detail` | #4 task_decomp, #5–7 sector/arena/object, #10 obj state, #9/#11 triples | **Rewrite** | one call gives steps + object + object state |
| `replan` | #12 new_decomp_schedule | **Rewrite** | JSON blocks from now to end of day |
| `react` | #13 decide_to_talk, #14 decide_to_react | **Rewrite to the paper's §4.3.1 prompt** | one decision per agent-tick over salient observations; talk / change_plan |
| `relationship` | #22 summarize_chat_relationship | **Port** | once per conversation per participant |
| `utterance` | #23 iterative_convo | **Rewrite** from the paper's §4.3.2 prompts | `{utterance, end_conversation}` |
| `summarize_dialogue` | #15 summarize_conversation | **Port** + a "preserve specifics" instruction | `{topic_label, summary}` |
| `conversation_note` | #20 planning_thought, #21 memo | **Port**, merged | adds a structured `commitments` list |
| `reflect_questions` | #18 focal_pt | **Port** (paper §4.2 wording) | 100 most recent records |
| `reflect_insights` | #19 insight_and_evidence | **Port** (paper §4.2 wording) | `evidence` becomes a validated int list |
| `summary` | (not implemented, D12) | **New**, from the App. A prompt | one call, three fields |
| `interview` | #24 summarize_ideas, #25 next_convo_line | **Rewrite** | a single call with cited indices |
| `judge` | — | **New** | report only |
| — | #26 whisper_inner_thought, #27 safety_score, all SPO-triple and keyword prompts | **Dropped** | seeds stored verbatim; keywords unused (we retrieve by embedding) |

**Template conventions:**
- One file per task in `llm/prompts/`, with a front-matter header `id`, `version`, `model_task`.
  The body uses `$placeholders` (`string.Template`).
- `prompt_id@version` is logged on every call.
- Every template states the agent's perspective, the current game time, and the rule "use only the information given;
  do not invent people or places". This counters hallucination and the paper's §7.2 over-politeness only partly;
  §11 Q2 covers the rest.

---

## 10. Intentional deviations from the paper (and why)

| # | Paper | Lite | Reason |
|---|---|---|---|
| L1 | the react context is summarized by an LLM from two retrievals (§4.3.1) | retrieved memories passed directly to the Haiku decision | saves a Sonnet call per observation; switchable via `react.context = "summarized"` |
| L2 | App. A summary built from three parallel summarization prompts | one call with three output fields | a third of the calls; same inputs |
| L3 | recursive tree traversal for the location, one level per prompt (§5.1) | enumerated place/object choices inside the planning calls | the tree is 2–3 levels and small; equivalent choice, 3–4 fewer calls per action |
| L4 | dialogue advances over game time | a whole conversation is generated within one tick; time is charged afterwards | simpler engine; each utterance is still conditioned per speaker (the paper's mechanism) |
| L5 | the relationship context is re-derived per turn | a relationship summary per conversation, plus per-turn retrieval on the last utterance | about half the dialogue calls |
| L6 | agents perceive within a visual radius | agents perceive everything at their current place (capped by `att_bandwidth`) | no coordinates |
| L7 | movement takes path-dependent time | fixed `travel_ticks` | no map |
| L8 | reflection threshold 150 | configurable; party scenario 75 | 4 agents and coarse ticks produce fewer observations |
| L9 | plans in the memory stream at every level | day + hour + commitments in the stream; 5–15 min steps in scratch | avoids flooding retrieval with micro-steps; matches the original's JIT spirit |
| L10 | the post-conversation step isn't described | conversation notes + commitments (from the original code) | needed for reliable coordination; logged separately so an ablation can disable it |

**Kept from the paper, correcting the original code:** D1–D3 (recency per game-hour, direction, weights),
D5 (scored retrieval on the react path), D6 (chats retrievable), D7 (100 records, evidence trees),
D10 (new day plan), D11 (object reactions), D12 (App. A summary).

---

## 11. Open questions (defaults chosen; change any before Phase 2)
1. **Character names:** echo the paper's (Isabella, Sam, Maria, Klaus) for comparability. *Default: yes.*
2. **Over-cooperation (§7.2):** add a persona instruction to "stay true to your own interests; you may decline"?
   *Default: off for fidelity, as a config flag for experiments.*
3. **Learning places from dialogue** (e.g. hearing "party at Hobbs Cafe" when Hobbs Cafe isn't in your known tree):
   *Default: public places (the cafe, park, market, and library) are known to everyone, so this doesn't block the
   diffusion test. Private homes stay unknown unless visited or listed.*
4. **Concurrency:** LLM calls run sequentially in v1 for determinism and simple budget accounting. Within-phase
   parallelism (async and budget reservations) is a later optimization.
