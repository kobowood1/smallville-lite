# ARCHITECTURE — Mapping the original Generative Agents code

A read-only map of the reference implementation of *Generative Agents: Interactive
Simulacra of Human Behavior* (Park et al., UIST 2023), written to guide a
lightweight Claude-based port (`smallville_lite`).

- **Reference revision:** `joonspk-research/generative_agents` @ `fe05a71d3e4ed7d10bf68aa4eda6dd995ec070f4` (2023-08-11), cloned to `./reference` (sparse checkout of `reverie/` plus the frontend `translator/` and `templates/`. The full checkout fails on Windows because of long paths in `environment/frontend_server/storage`, and seed files were read with `git show`).
- **Paths:** unless stated otherwise, every `file:line` is relative to
  `reference/reverie/backend_server/`. `rgp` = `persona/prompt_template/run_gpt_prompt.py`;
  `tpl/` = `persona/prompt_template/`.
- **Paper refs** use its section numbers (§4.1 etc.) and appendices (App. A, App. B).
- Terminology: in the code, *persona* = agent, *associative memory* = memory stream,
  *scratch* = short-term state, *poignancy* = importance, *pronunciatio* = emoji label,
  *thought* = reflection (and also plans), *ISS* = "identity stable set".

---

## 1. Overview and per-step control flow

`reverie.py` holds a `ReverieServer`. Its `start_server` loop (`reverie.py:279-412`) is
driven by the Phaser frontend through the filesystem. On each step it:

1. waits for the frontend to write `storage/<sim>/environment/<step>.json` (agent x,y),
2. updates tile events in the `Maze` (`reverie.py:338-365`),
3. calls `persona.move(...)` for every agent in sequence (`reverie.py:379`),
4. writes `movement/<step>.json` (next tile, emoji, description, chat) for the frontend,
5. advances time by `sec_per_step` (10 game-seconds in the seed `meta.json`).

One agent step, `Persona.move` (`persona/persona.py:185-231`):

```
move(maze, personas, curr_tile, curr_time)
 ├─ new_day? ("First day" | "New day" | False)            persona.py:211-217
 ├─ perceive(maze)            → new event ConceptNodes     cognitive_modules/perceive.py:25
 │    └─ per new event: embedding + importance LLM call; decrement reflection counter
 ├─ retrieve(perceived)       → keyword-matched events/thoughts per perceived event
 │                                                         cognitive_modules/retrieve.py:16
 ├─ plan(maze, personas, new_day, retrieved)               cognitive_modules/plan.py:931
 │    ├─ new_day → _long_term_planning (wake hour, day plan, hourly schedule)  plan.py:461
 │    ├─ action finished → _determine_action (JIT decomposition + location + emoji) plan.py:521
 │    └─ focused event → _should_react → _chat_react | _wait_react  plan.py:970-987
 ├─ reflect()                 → threshold reflection + post-conversation thoughts
 │                                                         cognitive_modules/reflect.py:172
 └─ execute(maze, personas, plan) → path-finding, next tile   cognitive_modules/execute.py:15
```

Paper: §4 (Fig. 5) and §5. The code order perceive → retrieve → plan → reflect → act
matches Figure 5.

---

## 2. Memory stream

**Where:** `persona/memory_structures/associative_memory.py`. Paper: §4.1.

### ConceptNode fields (`associative_memory.py:19-43`)

| Field | Meaning |
|---|---|
| `node_id`, `node_count`, `type_count` | `"node_N"`; global and per-type counters |
| `type` | `"event"` (observation), `"thought"` (reflection *and* plan), `"chat"` |
| `depth` | 0 for events and chats; thoughts are `1 + max(depth of evidence)` (`:207-210`), which gives reflection-tree height |
| `created`, `expiration`, `last_accessed` | datetimes. Thoughts expire after +30 days, but expiration is never enforced anywhere |
| `subject`, `predicate`, `object` | SPO triple, produced by an LLM call for thoughts |
| `description` | natural-language text |
| `embedding_key` | the text that was embedded, used as the key into `embeddings` dict (often ≠ description) |
| `poignancy` | importance 1–10 |
| `keywords` | set, used for keyword indexes |
| `filling` | evidence pointers: list of node_ids for thoughts, transcript `[[speaker, utt], ...]` for chats, `[chat_node_id]` for a chat's event node |

### Containers (`associative_memory.py:50-109`)
- `seq_event`, `seq_thought`, `seq_chat`: lists, **newest first** (`[0:0] = [node]` insertion).
- `id_to_node`: dict.
- `kw_to_event`, `kw_to_thought`, `kw_to_chat`: lowercase-keyword → nodes. `kw_strength_*` counters are maintained but unused by the live code.
- `embeddings`: `{embedding_key_text: vector}`, shared across nodes with the same text. This doubles as the only embedding cache.

### Persistence
`save()` (`:112-150`) writes `nodes.json`, `kw_strength.json`, `embeddings.json` under
`storage/<sim>/personas/<name>/bootstrap_memory/associative_memory/`. Scratch
(`scratch.json`) and spatial memory (`spatial_memory.json`) are saved alongside
(`persona.py:51-78`). There's no append-only log: each save overwrites the full state.

### Short-term state: `Scratch` (`memory_structures/scratch.py`)
- Perception hyperparameters: `vision_r=4, att_bandwidth=3, retention=5` (`:19-23`). The seed
  `scratch.json` overrides these to **8/8/8**.
- Identity: `name, age, innate, learned, currently, lifestyle, living_area, daily_plan_req`.
- Retrieval and reflection: `recency_w/relevance_w/importance_w = 1`, `recency_decay = 0.99`
  (`:57-60`), overridden to **0.995** in seed files. Also `importance_trigger_max = 150`,
  `importance_trigger_curr`, `importance_ele_n` (`:61-63`).
- Plans: `daily_req` (broad strokes), `f_daily_schedule` (partially decomposed
  `[task, minutes]` list summing to 1440), `f_daily_schedule_hourly_org` (undecomposed copy).
- Current action: `act_address, act_start_time, act_duration, act_description,
  act_pronunciatio, act_event`, object-state equivalents, `chatting_with, chat,
  chatting_with_buffer, chatting_end_time`, and path state.
- `get_str_iss()` (`:382-414`) is the "[Agent's Summary Description]" actually used in
  prompts (see §11, difference D12).

### Seeding (paper §3.1 and §5)
There are two channels in the code:
1. **Identity fields** in `scratch.json`. For example, Isabella's party intent is written into her
   `currently` field (`storage/base_the_ville_isabella_maria_klaus/.../scratch.json`), and Sam's
   candidacy into his `currently` field (`base_the_ville_n25`). Seed `nodes.json` is `{}`.
2. **Semicolon-delimited "whispers"**: the CLI command `call -- load history <csv>`
   (`reverie.py:577-591`) reads `static_dirs/assets/the_ville/agent_history_init_n3.csv`,
   splits each row on `;`, and `load_history_via_whisper` (`converse.py:239-254`) turns
   each phrase into a **thought** node. Each phrase costs one LLM rewrite
   (`whisper_inner_thought`), one triple call, one importance call, and one embedding.

---

## 3. Retrieval

There are **two** retrieval functions in `persona/cognitive_modules/retrieve.py`:

### 3a. `retrieve()`: keyword retrieval, used on the perceive → react path (`retrieve.py:16-46`)
For each perceived event, it returns all events and thoughts whose keyword index contains that event's
subject, predicate, or object (`associative_memory.py:305-326`). **No scoring.** This is what feeds
`decide_to_talk` / `decide_to_react`.

Case bug: keys are lowercased on insert (`:178`), but `retrieve_relevant_events` looks up the raw
string (`:322`), so capitalized names never match.

### 3b. `new_retrieve()`: the paper's scored retrieval (`retrieve.py:199-270`)
It's used by reflection, dialogue, `revise_identity`, and the interview. For each focal-point string:
1. Candidates are `seq_event + seq_thought` (not chats), excluding embedding keys containing
   `"idle"`, **sorted ascending by `last_accessed`** (`:224-228`).
2. **Recency** (`extract_recency`, `:132-152`): `recency_decay ** i` for `i = 1..N` over that
   sorted order (`:145`). This is by rank, not by game-hours.
3. **Importance**: raw `poignancy` (`:155-172`).
4. **Relevance**: cosine(node embedding, `get_embedding(focal_pt)`) (`:175-193`). The query is
   embedded on every call (`:189`), with no cache.
5. Each component is min-max normalized to [0, 1] (`normalize_dict_floats`, `:70-104`). If all
   values are equal, it returns 0.5.
6. Score = `recency_w·rec·0.5 + relevance_w·rel·3 + importance_w·imp·2`. The hard-coded
   `gw = [0.5, 3, 2]` (`:244`) is multiplied on top of the α weights of 1.
7. Top `n_count` (default 30; callers use 50/15) (`:262`). Retrieved nodes get
   `last_accessed = curr_time` (`:266-267`). **Retrieval does update last-accessed**, but
   only in `new_retrieve`.

Paper: §4.1 says the score is α-weighted with all α = 1, recency decays exponentially 0.995 per
game-hour since last retrieval, and min-max normalization is applied. See differences D1–D5.

---

## 4. Reflection

**Where:** `persona/cognitive_modules/reflect.py`. Paper: §4.2.

- **Trigger counter:** in `perceive`, each *new perceived event* decrements
  `importance_trigger_curr` by its importance and increments `importance_ele_n`
  (`perceive.py:178-179`). Thoughts and chats do not decrement it.
- **Trigger:** `reflection_trigger` fires when `importance_trigger_curr <= 0` (the sum of new
  event importance ≥ 150) (`reflect.py:135-155`). After reflecting, the counter resets to max and
  `importance_ele_n` to 0 (`:158-169`).
- **Flow** (`run_reflect`, `reflect.py:99-132`):
  1. `generate_focal_points(n=3)` (`:21-35`): takes the last `importance_ele_n` nodes by
     `last_accessed` (events and thoughts, not idle), lists their `embedding_key`s, and asks for the
     3 most salient high-level questions (`focal_pt`).
  2. `new_retrieve(persona, focal_points)`: 30 nodes per question.
  3. For each question: `generate_insights_and_evidence(nodes, n=5)` (`:38-55`) numbers the
     statements `0..k`, asks for "5 high-level insights … (because of 1, 5, 3)", and maps the
     cited indices to node_ids. That gives **up to 15 insights per reflection.**
  4. Each insight becomes a `thought` with an SPO triple (LLM), importance (LLM, event prompt),
     an embedding, `filling = evidence node_ids`, and `expiration = +30 days`.
- **Post-conversation reflection** (not in the paper, `reflect.py:190-244`): when
  `curr_time + 10s == chatting_end_time`, the agent writes two thoughts from the transcript:
  a *planning thought* ("For X's planning: …") and a *memo* ("X …"). Both have evidence
  `[last chat node]`, and each costs a triple call, an importance call, and an embedding.

---

## 5. Planning (day → hour → 5-min, just-in-time)

**Where:** `persona/cognitive_modules/plan.py`. Paper: §4.3 and App. A.

**Long-term planning on a new day** (`_long_term_planning`, `plan.py:461-513`):
1. `wake_up_hour` (LLM): an integer hour.
2. First day: `daily_plan` (LLM) produces `daily_req`, a list of 4–8 broad-stroke items that always starts with
   "wake up and complete the morning routine at H:00 am".
   New day: `revise_identity` (`plan.py:408-458`) makes 4 raw `ChatGPT_single_request` calls
   (a plan note and a feelings note from `new_retrieve` over two focal points, a new `currently`,
   and a new `daily_plan_req`). **`daily_req` itself is not regenerated** (`plan.py:489` is a
   `TODO` no-op).
3. `generate_hourly_schedule` (`plan.py:71-138`): **one LLM call per waking hour**, with pre-wake
   hours hard-coded to `"sleeping"`. The whole pass repeats up to 3 times if fewer than 5
   distinct activities come back. Consecutive identical hours are merged into `[task, 60·n]`.
4. The day plan is stored in memory as a single thought, "This is X's plan for <date>: …",
   with **importance fixed at 5**, keyword `plan`, and an embedding (`plan.py:501-513`).

**Short-term, just-in-time decomposition** (`_determine_action`, `plan.py:521-652`,
which runs whenever `act_check_finished()`):
- If an hourly block ≥ 60 min covers *now* (on the first index) or *now + 60 min*, it is replaced
  in place by `task_decomp` output. That's an LLM list "in 5 min increments" with per-item durations,
  re-padded to sum to the block, and each item is renamed `"<hourly task> (<subtask>)"`.
- Sleep-like tasks are never decomposed (`determine_decomp`, `:533-552`).
- The schedule is padded with a final `sleeping` block to reach 1440 min (`:606-613`).
- The action at the current index is grounded to the world through a chain of LLM calls:
  **sector → arena → object** (`action_sector`, `action_arena`, `action_game_object`),
  then **emoji** (`pronunciatio`), **SPO triple**, **object state description**, emoji for the
  object state, and the object SPO triple (`plan.py:624-652`). That's about 8 LLM calls per new action.

Paper §4.3 describes a day plan in 5–8 chunks → hour-long chunks → 5–15 min chunks, stored
in the memory stream with location, start time, and duration. See differences D8 and D9.

---

## 6. Perceive → react-or-continue

**Where:** `perceive.py:25-181`, `plan.py:655-1007`. Paper: §4.3.1, §5.

**Perceive** (tile-map bound):
- Spatial: all tiles within `vision_r` update the agent's `s_mem` tree (world → sector → arena → objects) (`perceive.py:46-68`).
- Events: collect events on nearby tiles in the **same arena**, sort by distance, and keep the
  closest `att_bandwidth` (`:73-103`). An event is **new** if its (s, p, o) is not among the last
  `retention` events in memory (`:122-124`).
- Each new event gets an embedding (cache lookup by text) and importance (LLM), and is added with `add_event`.
  If the event is the agent's own `chat with`, a `chat` node holding the transcript is also created, and
  the event's `filling` points to it (`:155-172`).

**Choose focus** (`_choose_retrieved`, `plan.py:655-696`): drop self-events. Prefer events whose
subject is another *persona* (no `:` in the subject), then any non-idle event. Pick randomly among those.

**Decide** (`_should_react`, `plan.py:699-803`). Only persona subjects can trigger a reaction:
- `lets_talk` has hard gates: both agents have an action, neither is sleeping, the hour isn't 23,
  the target isn't `<waiting>`, neither is already chatting, and the `chatting_with_buffer` cooldown has expired.
  After that comes the **LLM `decide_to_talk`** (yes/no). A yes returns `"chat with <name>"`.
- Otherwise `lets_react` applies the same gates, plus: the initiator must be on the move
  (`planned_path` non-empty) and both must have the **same `act_address`**. Then the
  **LLM `decide_to_react`** returns Option 1 (wait until the other is done) or Option 2 (continue).
  Only Option 1 produces a reaction (`wait: <time>`). Option 2 and anything else mean no reaction.

**React / replan** (`_create_react`, `plan.py:806-857`): take the current hourly-org block
(and the next one if it's < 2 h), truncate the schedule at *now*, insert the new act, and call
**LLM `new_decomp_schedule`** to rewrite the rest of that window. `_chat_react`
(`:860-904`) does this for **both** agents with the conversation summary as the inserted act, emoji 💬,
and a cooldown buffer of 800 steps. `_wait_react` (`:907-928`) inserts
"waiting to start …" with emoji ⌛.

---

## 7. Dialogue (turn-by-turn)

**Where:** `converse.py:126-179` (`agent_chat_v2`), invoked from `generate_convo`
(`plan.py:277-293`). Paper: §4.3.2.

- The **entire conversation is generated synchronously** inside the initiator's `plan()` call at
  the moment the chat starts, for up to 8 rounds (16 utterances).
- Per utterance, for the speaker:
  1. `new_retrieve(speaker, [other_name], 50)` → **LLM `agent_chat_summarize_relationship`**.
     The relationship summary is recomputed on every turn.
  2. `new_retrieve(speaker, [relationship, "<other> is <other's action>", last ≤4 lines], 15)`.
  3. **LLM `iterative_chat_utt`** (`tpl/v3_ChatGPT/iterative_convo_v1.txt`). The prompt includes the speaker's ISS,
     the retrieved memory descriptions, past-conversation context (if chatted < 8 h ago),
     location, current context, and the transcript so far. It returns JSON
     `{"<name>": utterance, "Did the conversation end with <name>'s utterance?": bool}`.
  4. If `end` is set, the conversation stops. Either agent can end it. The listener always responds;
     there's no "decide to respond".
- Afterwards: **LLM `summarize_conversation`** produces "conversing about …", which becomes both agents' action.
  Duration is `ceil(len(transcript_chars)/8/30)` minutes (`plan.py:290`). Both agents' schedules are
  replanned. The transcript is stored in `scratch.chat`, and the next perceive step turns it into a chat node.

**Interview** ("call -- analysis <name>", `converse.py:257-277`): a safety score
(anthropomorphization, ≥8 means refuse), then `new_retrieve([question], 50)` → `summarize_ideas` →
`generate_next_convo_line` with interlocutor "Interviewer". The answer is **not** written
back to memory. Paper §6 and App. B use this same kind of interview.

---

## 8. LLM call-site inventory

Plumbing is in `tpl/gpt_structure.py`:
- `safe_generate_response` (`:255-273`) wraps the **legacy Completions API**
  (`GPT_request`, `:197-224`, text-davinci-002/003). It retries up to 5× until `func_validate` passes, then
  applies `func_clean_up`. After that it returns the hard-coded `fail_safe`.
- `ChatGPT_safe_generate_response` (`:123-164`) wraps **gpt-3.5-turbo**. It encloses the prompt in `"""`,
  appends "Output the response to the prompt above in json … Example output json:
  {"output": "<example>"}", truncates at the last `}`, runs `json.loads(...)["output"]`,
  and retries up to 3×. **On failure it returns `False`, not the fail-safe.** Every caller does
  `if output != False: return …` and then falls off the end of the function, returning `None`.
  The caller's `[0]` then raises `TypeError`, except where noted below.
- `ChatGPT_safe_generate_response_OLD` (`:167-190`): raw gpt-3.5 text, validate/clean, then the fail-safe.
- `ChatGPT_single_request` (`:19-26`): raw gpt-3.5 with no retry or validation.
- `generate_prompt` (`:227-252`) substitutes `!<INPUT n>!` placeholders and strips the header above
  `<commentblockmarker>###</commentblockmarker>`.

`rgp` defines 34 `run_gpt_*` functions: **27 live**, 3 referenced only by dead code, and 4 never
called. `plan.py` also makes 4 inline raw calls.

**Model key:** D2 = text-davinci-002 completion, D3 = text-davinci-003 completion,
C = gpt-3.5-turbo via the JSON wrapper, C-raw = gpt-3.5-turbo with no JSON wrapper.

| # | Function (`rgp` line) | Called from | Template (`tpl/`) | Model / params | Expected output | Parse | Fallback |
|---|---|---|---|---|---|---|---|
| 1 | `wake_up_hour` :39 | plan.py:38 (daily) | v2/wake_up_hour_v1 | D2, T=0.8, max 5, stop `\n` | e.g. `8am` | `int(split("am")[0])` | `8` |
| 2 | `daily_plan` :87 | plan.py:68 (first day) | v2/daily_planning_v6 | D3, T=1, max 500 | continues `1) …, 2) …` | split on `)`, drop trailing index digit and `,`/`.` | fixed 7-item day; wake-up item always prepended |
| 3 | `generate_hourly_schedule` :161 | plan.py:108 (**once per waking hour**, ×≤3 passes) | v2/generate_hourly_schedule_v2 | D3, T=0.5, max 50, stop `\n` | activity after "X is" | strip, drop trailing `.` | `"asleep"` |
| 4 | `task_decomp` :297 | plan.py:164 (JIT) | v2/task_decomp_v3 | D3, T=0, max 1000 | numbered lines `… (duration in minutes: N, minutes left: M)` | per line: drop first 3 words, split on `(duration in minutes:`, expand to per-minute slots, pad or truncate to the total | validate always truthy, so parse errors **raise**; post-processing trims to the block |
| 5 | `action_sector` :493 | plan.py:182 | v1/action_location_sector_v1 | D2, T=0, max 15 | `<sector>}` | `split("}")[0]`, reject if `,` | `"kitchen"`; if not a known sector, use the living-area sector |
| 6 | `action_arena` :631 | plan.py:200 | v1/action_location_object_vMar11 | D3, T=0, max 15 | `<arena>}` | same as #5 | `"kitchen"` (no tree check) |
| 7 | `action_game_object` :726 | plan.py:223 | v1/action_object_v2 | D3, T=0, max 15 | object name | strip | `"bed"`; if not in arena, `random.choice` |
| 8 | `pronunciatio` :785 | plan.py:243 (×2 per action) | v3_ChatGPT/generate_pronunciatio_v1 | C | emoji | first 3 chars | `None` → caller try/except → 🙂 |
| 9 | `event_triple` :876 | plan.py:264, reflect.py:70, converse.py:223 | v2/generate_event_triple_v1 | D3, T=0, max 30, stop `\n` | `pred, obj)` | `split(")")[0].split(",")`, need 2 | `(name, "is", "idle")` |
| 10 | `act_obj_desc` :965 | plan.py:269 | v3_ChatGPT/generate_obj_event_v1 | C | object-state phrase | strip trailing `.` | `None` → **crash** |
| 11 | `act_obj_event_triple` :1045 | plan.py:274 | v2/generate_event_triple_v1 | D3, T=0, max 30 | `pred, obj)` | as #9 | `(obj, "is", "idle")` |
| 12 | `new_decomp_schedule` :1090 | plan.py:395 (every reaction, both chat parties) | v2/new_decomp_schedule_v1 | D3, T=0, max 1000 | lines `HH:MM ~ HH:MM -- action` | prompt+completion re-split, time deltas; valid only if the sum equals the window | truncated plan spliced with the original remainder |
| 13 | `decide_to_talk` :1244 | plan.py:302 | v2/decide_to_talk_v2 | D3, T=0, max 20 | CoT then yes/no | `split("Answer in yes or no:")`, which **doesn't match the template's quoted `"yes" or "no"`** | `"yes"` |
| 14 | `decide_to_react` :1344 | plan.py:313 | v2/decide_to_react_v1 | D3, T=0, max 20 | CoT then `Answer: Option N` | split, expect 1/2/3 | `"3"` (no reaction) |
| 15 | `summarize_conversation` :1591 | plan.py:297 | v3_ChatGPT/summarize_conversation_v1 | C | phrase | prefixed with `"conversing about "` | `None` → crash |
| 16 | `event_poignancy` :1845 | perceive.py:20, reflect.py:80, converse.py:233 | v3_ChatGPT/poignancy_event_v1 | C | int 1–10 | `int()` (no range check) | `None` → crash |
| 17 | `chat_poignancy` :1989 | perceive.py:22 | v3_ChatGPT/poignancy_chat_v1 | C | int 1–10 | `int()` | `None` → crash |
| 18 | `focal_pt` :2064 | reflect.py:35 | v3_ChatGPT/generate_focal_pt_v1, then v2/generate_focal_pt_v1 | C, then D3 (T=0, max 150) | list of questions | `ast.literal_eval`; fallback parses `1) …` lines | `["Who am I"]*n` |
| 19 | `insight_and_guidance` :2142 | reflect.py:45 | v2/insight_and_evidence_v1 | D3, T=0.5, max 150 | `insight (because of 1, 5, 3)` lines | split on `(because of `, regex digits | fail-safe is a *list*; the caller's `.items()` throws → `{"this is blank": "node_1"}` |
| 20 | `planning_thought_on_convo` :2655 | reflect.py:89 | v2/planning_thought_on_convo_v1 | D3, T=0, max 50 | sentence | `split('"')[0]` | `"..."` |
| 21 | `memo_on_convo` :2692 | reflect.py:94 | v3_ChatGPT/memo_on_convo_v1, then v2/memo_on_convo_v1 | C, then D3 | sentence | strip / `split('"')[0]` | `"..."` |
| 22 | `agent_chat_summarize_relationship` :2265 | converse.py:53 (every utterance) | v3_ChatGPT/summarize_chat_relationship_v2 | C | relationship summary | `split('"')[0]` | `None` → crash |
| 23 | `iterative_chat_utt` :2821 | converse.py:118 (every utterance) | v3_ChatGPT/iterative_convo_v1 | C-raw | JSON `{name: utt, "Did … end …": bool}` | first `{…}`, values taken **by position**; `end = True` unless the value's text contains "f"/"F" | `{"utterance": "...", "end": False}` |
| 24 | `summarize_ideas` :2474 | converse.py:190 (interview) | v3_ChatGPT/summarize_ideas_v1 | C | summary | `split('"')[0]` | `None` → crash |
| 25 | `generate_next_convo_line` :2540 | converse.py:200 (interview) | v2/generate_next_convo_line_v1 | D3, T=1, max 250 | utterance up to closing quote | `split('"')[0]` | `"..."` |
| 26 | `generate_whisper_inner_thought` :2618 | converse.py:208 (seeding, whisper) | v2/whisper_inner_thought_v1 | D3, T=0, max 50 | third-person statement | `split('"')[0]` | `"..."` |
| 27 | `generate_safety_score` :2759 | converse.py:267 (interview) | safety/anthromorphosization_v1 | C-raw | `{"output": int}` | `json.loads` | `None` → `int(None)` crash |
| 28–31 | *inline* `revise_identity` | plan.py:426, 432, 444, 455 (new day) | f-strings in plan.py | `ChatGPT_single_request` | plan note; feelings note; `Status: <new status>`; numbered day plan | **none**: stored verbatim (including the `Status:` prefix); the day plan is flattened to one line | none (exceptions propagate) |

**Dead or unused:** `create_conversation` :1455 (call commented out at plan.py:280);
`agent_chat` :2333 and `agent_chat_summarize_ideas` :2196 (only reached via the unused
`agent_chat_v1`); `extract_keywords` :1665, `keyword_to_thoughts` :1725,
`convo_to_thoughts` :1769, and `thought_poignancy` :1918 are never called. `defunct_run_gpt_prompt.py`
is an older copy of the same file.

**Rough call volume** (from reading the code, not measured):
- Each new action: ~8 calls (#5–11, with #8 twice), plus #4 when a block is decomposed.
- Each newly perceived event: 1 importance call and 1 embedding.
- Each conversation of *k* utterances: 2k LLM calls (#22, #23) and about 3k query embeddings, plus #15.
  Then per agent: #12 (replan), #20, #21, and a triple, importance, and embedding for each of those two thoughts.
- Each reflection: 1 (#18) + 3 (#19) + up to 15 × (triple + importance + embedding).
- Each new day: #1, 4 raw calls, and up to 24–72 calls for #3.

---

## 9. Embedding call sites

All go through `get_embedding(text, model="text-embedding-ada-002")` (`gpt_structure.py:276-281`).
It replaces newlines with spaces, maps empty text to "this is blank", and makes one API call per text.
The only cache is `a_mem.embeddings` (keyed by exact text), and it's consulted only at the two
perceive sites.

| Site | What is embedded | Cached? |
|---|---|---|
| `perceive.py:144` | new event description (the parenthesized sub-action if present) | yes, looked up in `a_mem.embeddings` (`:141`) |
| `perceive.py:161` | own chat's `act_description` ("conversing about …") | yes (`:157`) |
| `plan.py:510` | daily plan thought | no |
| `reflect.py:128` | each reflection insight | no |
| `reflect.py:224`, `:240` | post-chat planning thought, memo | no |
| `converse.py:251`, `:288` | seeded or whispered thoughts | no |
| `retrieve.py:189` | **every `new_retrieve` focal point / query** | no. This is the hottest site (≈3 per utterance, 3 per reflection, 1 per interview turn) |

---

## 10. Tile-map and frontend coupling to drop

| Component | Where | Why it can go |
|---|---|---|
| `Maze`: tile matrix, collision map, `address_tiles`, per-tile event sets | `maze.py` (whole file) | replace with a location tree plus a per-location set of current events |
| Path-finding | `path_finder.py` (whole file), `execute.py:35-147` | lite uses a fixed travel time in ticks |
| `execute` output `(next_tile, emoji, "desc @ address")` | `execute.py:150-159` | keep only emoji and description plus address |
| Tile fields in scratch: `curr_tile`, `planned_path`, `act_path_set`, `vision_r` | `scratch.py` | "in transit" becomes an explicit state |
| Vision radius, distance sort, arena check in perceive | `perceive.py:46-103` | perceive = the events at the agent's current location (keep `att_bandwidth` and `retention`) |
| Address sentinels `<persona> X`, `<waiting> x y`, `<random>` | plan.py:884, 914; plan.py:222; execute.py | use typed action fields |
| Sector/arena name filtering by last name (`'s house`, `'s room`) | `rgp:521-530, 648-655` | express ownership/access in the location tree |
| File-based frontend handshake (`environment/<step>.json` ↔ `movement/<step>.json`), `selenium` import, `server_sleep` polling | `reverie.py:31, 138, 293-412` | the lite runner owns the clock |
| Path-tester mode, fork/copy of storage dirs | `reverie.py:190-276, 43-67` | not needed |
| `compress_sim_storage.py`, the whole `environment/frontend_server/` (Django `translator/views.py`, templates, Phaser `static_dirs`, sqlite) | repo root | per the user constraints: no Django, no Phaser, no assets |
| Hard-coded location fallbacks (`"kitchen"`, `"bed"`, `"Johnson Park:park:park garden"`) | rgp #5–7, `execute.py:92` | tied to the_ville map |

**Keep in spirit:**
- `MemoryTree` / `s_mem` as each agent's known subgraph (paper §5.1).
- The pronunciatio emoji (for the web log).
- Object state descriptions (#10).
- The chat cooldown buffer.
- Per-agent perception limits (`att_bandwidth`, `retention`).

---

## 11. Code vs paper: differences

| # | Topic | Paper | Code |
|---|---|---|---|
| D1 | Recency formula | exponential decay, 0.995 per **game-hour since last retrieval** (§4.1) | `decay ** rank` over memories sorted by `last_accessed`. There's no time term (`retrieve.py:145`). Default 0.99 in `scratch.py:60`, 0.995 in seed files |
| D2 | Recency direction | recent = high | Candidates are sorted **ascending** (oldest first) and rank 1 gets `decay**1`, so the **oldest memory gets the highest recency** (`retrieve.py:224-228, 145`). This reads as an inversion bug |
| D3 | Weights | all α = 1 (§4.1) | Hard-coded `gw = [0.5, 3, 2]` is multiplied onto α (`retrieve.py:244`). Relevance dominates, and recency is nearly ignored |
| D4 | Top-k | "top memories that fit in the context window" | fixed `n_count`: 30 by default, 50/15 in dialogue and interview |
| D5 | Retrieval in the react path | scored retrieval on "[observer]'s relationship with [observed]" and "[observed] is [status]", then summarized (§4.3.1) | exact keyword match with no scoring and no summary (`retrieve.py:16-46`); proper names never match (case bug) |
| D6 | Chats in retrieval | observations include conversations (§4.1) | `new_retrieve` only searches events and thoughts; chat transcripts are reachable only via the "conversing about …" event and post-chat thoughts |
| D7 | Reflection input | the 100 most recent records → 3 questions → 5 insights with citations (§4.2) | uses the last `importance_ele_n` records by `last_accessed` (the count since the last reflection) → 3 questions → **5 insights per question** (≤15); plus undocumented post-conversation planning and memo thoughts |
| D8 | Plan granularity | day (5–8 chunks) → hour-long chunks → 5–15 min chunks (§4.3) | day → hourly via **one LLM call per hour** → "5 min increments" decomposition of the current and next-hour blocks only (JIT, App. A). Sleep is never decomposed |
| D9 | Plans in memory | plan entries with location, start, and duration are stored in the memory stream and retrieved (§4.3) | only the broad `daily_req` is stored, as one thought with **importance fixed at 5**. The decomposed schedule lives in scratch |
| D10 | Next-day planning | the prompt uses the agent summary plus a summary of the previous day (§4.3 example) | `revise_identity` updates `currently` and `daily_plan_req` via 4 raw calls, but `daily_req` is **never regenerated** (`plan.py:486-489` TODO) |
| D11 | React decision | a general "Should X react, and if so, how?" for any observation, then regenerate the plan from now (§4.3.1). §3.2 shows reactions to object state (a burning stove) | reactions only to **other agents**. `_should_react` returns False for object subjects, so the burning-stove demo has no code path. There are two fixed modes: chat (yes/no) or wait-until-done (options). Many hard-coded gates (hour 23, sleeping, same address, 800-step chat cooldown). The decide-to-talk parser mismatch likely makes the "yes" fail-safe common |
| D12 | Agent summary (App. A) | a cached synthesis of 3 scored retrievals ("core characteristics", "current daily occupation", "feeling about recent progress"), refreshed at intervals | `get_str_iss()` concatenates static scratch fields. Only `currently` and `daily_plan_req` change, once per day |
| D13 | Dialogue | each turn: the speaker retrieves and summarizes memory of the other plus the last utterance; the listener *decides* whether to respond; it continues until one ends it (§4.3.2) | matches per-utterance retrieval (the relationship is re-summarized **every turn**), but the whole dialogue (≤16 utterances) is generated **at once** when the chat begins, with no time passing. The listener always responds. Duration comes from character count |
| D14 | Importance prompt | a generic "piece of memory" prompt (§4.1) | prefixed with the agent's ISS, with separate event and chat templates; reflections are scored with the *event* template (`thought_poignancy` is unused) |
| D15 | Seeding | a semicolon-delimited paragraph split into initial memories (§3.1, §5) | identity lives in scratch fields (the party and candidacy are in `currently`); semicolon phrases load via the `call -- load history` CLI and each is LLM-rewritten into a *thought* |
| D16 | Models | gpt-3.5-turbo (§4) | most calls use **text-davinci-002/003 completions** (since retired by OpenAI); some use gpt-3.5-turbo. The code does not run unmodified today |
| D17 | Perception | the server sends all agents and objects within visual range (§5) | same-arena only, nearest `att_bandwidth`, deduplicated against the last `retention` events |
| D18 | Game time | "one second real time is one minute game time" (App. A) | 10 game-seconds per step (`meta.json`). Action completion is tested with exact `HH:MM:SS` equality (`scratch.py:556`) |
| D19 | Retrieval accesses | retrieval updates the access time | true only for `new_retrieve`; keyword `retrieve()` never touches `last_accessed` |

---

## 12. Notes for the lite port (inputs to Phase 1)

**Fragile parsing to replace with structured output (JSON / tool use):**
- `daily_plan` (#2), `task_decomp` (#4), `new_decomp_schedule` (#12), and `insight_and_guidance` (#19)
  use positional free-text formats and prompt+completion re-splitting. These should become typed schemas:
  `[{task, start, minutes, location}]`, `[{insight, evidence_ids}]`.
- `decide_to_talk` / `decide_to_react` (#13, #14): chain-of-thought truncated at 20 tokens, a
  mismatched split string, and fail-safes that silently decide. Replace with a single "react?"
  schema `{react: bool, reaction: "continue"|"chat"|"wait"|"other", description, reason}`,
  closer to the paper's §4.3.1 prompt.
- `iterative_chat_utt` (#23): positional JSON values and a boolean inferred from the letter "f". Use
  `{utterance: str, end_conversation: bool}`.
- The sector → arena → object chain (#5–7) is 3 serial calls with hard-coded fallbacks. With a 4–6 place
  tree, one structured call choosing `{place, object}` from an enumerated list is enough.
- `revise_identity` (#28–31): unvalidated raw text stored verbatim.
- Global `ChatGPT_safe_generate_response` failures return `None` → `TypeError`. Every call needs an explicit,
  logged fallback.

**Semantics to port faithfully (paper-first, and configurable):**
- Recency uses game-hours since last access (fixing D1 and D2).
- α weights are configurable with default 1, and there's no hidden `gw` (D3).
- Scored retrieval on the react path (D5) and chats in the stream (D6).
- The reflection trigger sums new-observation importance, with a configurable threshold.
- Questions come from the last *N* (default 100) records. Each insight gets evidence pointers, depth, and a
  reflection tree (D7).
- Plans go into the stream with time and location (D9), and a new daily plan comes from yesterday's summary (D10).
- The react decision applies to any observation, including object state (D11).
- The App. A cached summary is regenerated on a schedule and is the natural prompt-cache prefix (D12).
- Dialogue advances turn by turn. For a text sim, generating the whole dialogue within one tick is acceptable
  as long as each utterance is conditioned per speaker, which is the paper's mechanism. A cached relationship
  summary per conversation avoids D13's per-turn re-summarization.

**Cost hot spots to design around:**
- Per-utterance relationship re-summarization.
- The per-hour schedule calls (#3).
- About 8 grounding calls per action (collapse to 1–2).
- Uncached query embeddings (`retrieve.py:189`); cache by text hash.
- The emoji call (#8) is a good Haiku task, or can be folded into the action call.

**Logging for the later phases.** The event log must make runs *replayable* without re-running the simulation. The Phase 5 visual town plays back entirely from JSONL, so every event needs:
- tick and game time;
- per-agent location and in-transit state (`from`, `to`, `arrive_tick`);
- the emoji and the full action text;
- dialogue turns as individual timestamped events;
- plan changes, and memory writes carrying node ids, so a clicked agent's memory and current plan can be rebuilt at any tick.
