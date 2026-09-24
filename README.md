# smallville-lite

A small, readable reimplementation of
[*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442)
(Park et al., UIST 2023) on the Claude API. A handful of agents live in a tiny town. They remember what they see,
reflect on it, plan their days, talk to each other, and pass news along. Every step is written to one event log, and a
web viewer replays that log as a 3D town.

**▶ [Watch the replay](https://claude.ai/artifact/7PTcwL9qRDNCm51DCQnAvq)** of a 2-day run in the browser. Nothing to
install.

[![The replay at 17:30 on day 2: all four agents at the Valentine's Day party at Hobbs Cafe, with Klaus and Sam's
conversation in the side panel](docs/replay-town.png)](https://claude.ai/artifact/7PTcwL9qRDNCm51DCQnAvq)

> **About this replay.** It was recorded in *stub mode*, which swaps the real models for an offline stand-in. The
> agents' words are placeholder text ("Hi Klaus! How are you?"), and the dollar figures estimate what the same calls
> would cost on the real models. The memory, retrieval, reflection, planning, movement, and dialogue machinery all ran
> for real; only the text generation was faked. No real-model run has been published yet.

## Contents

- [What it does](#what-it-does)
- [The party scenario](#the-party-scenario)
- [The replay viewer](#the-replay-viewer)
- [Quick start](#quick-start)
- [Models and cost](#models-and-cost)
- [How it differs from the paper](#how-it-differs-from-the-paper)
- [Repository layout](#repository-layout)
- [Project status](#project-status)
- [Sharing a replay](#sharing-a-replay)
- [Credits](#credits)

## What it does

Each agent runs the paper's loop on 10-minute game ticks:

| Mechanism | Paper | In short |
|---|---|---|
| **Memory stream** | §4.1 | Everything an agent observes, says, plans, or concludes becomes a timestamped memory with an importance score (1–10) and an embedding. |
| **Retrieval** | §4.1 | Memories are ranked by recency (decays 0.995 per game hour), importance, and relevance (cosine similarity), each min-max normalized. |
| **Reflection** | §4.2 | When recent importance adds up past a threshold, the agent asks itself three questions and writes insights that cite the memories behind them. Insights can cite earlier insights, which builds reflection trees. |
| **Planning** | §4.3 | A day plan (5–8 items), then hour blocks, then 5–15 minute steps worked out just in time. Plans change when something happens. |
| **Reacting** | §4.3.1 | On each salient observation (a person or an object), the agent decides whether to keep going, change plans, or start a conversation. |
| **Dialogue** | §4.3.2 | Conversations go turn by turn. Each line is conditioned on the speaker's memories of the other person and what was just said. Agreements become commitments that later plans honour. |
| **Interviews** | §6, App. B | You can ask any agent a question at any saved point in the run. The answer is grounded in its memories and shows which ones it used. |

After a run, a report measures the paper's §7.1 outcomes:

- how far each piece of news spread, with every "yes, I know" checked against the agent's memory to catch hallucination;
- how the relationship network grew;
- who turned up to the party.

## The party scenario

The one scenario so far mirrors the paper's information-diffusion test at a smaller scale. There are four agents and
six places in Smallville: Hobbs Cafe, Johnson Park, The Willows Market, Oak Hill College Library, Oak Hill Dorm, and
Moore House.

- **Isabella Rodriguez** runs Hobbs Cafe and is planning a Valentine's Day party there on Feb 14, 5–7 pm.
- **Sam Moore** has decided to run for mayor.
- **Klaus Mueller** and **Maria Lopez** start out knowing neither piece of news.

At the start, the relationship network has a single mutual tie, between Isabella and Klaus.

The run starts at 6:00 am on Feb 13, 2023 and lasts two game days. The question is whether both pieces of news spread
through conversation alone, and whether the people who hear about the party actually go.

In the demo run, both facts reached all four agents, the relationship network went from one tie to all six possible
ties, and all four attended the party. **Treat those numbers as a test that the plumbing works, not as a result.** The
stub model is written to know about the scenario, so it is designed to push the news along. The real test is a run on
the real models, which hasn't been done yet.

## The replay viewer

The viewer reads only the run's event log. It has these tabs:

- **Town:** an animated 3D map (three.js) with a day/night cycle, agents walking between buildings, emoji for what
  each is doing, and speech bubbles during conversations. Click an agent to see its current action, today's plan, and
  its recent memories. Playback runs at 1×–60×, and you can skip nights and slow down for dialogue. If WebGL isn't
  available, it falls back to a 2D map.
- **Timeline:** every action, per agent, per day.
- **Memory:** each agent's full memory stream. Filter it by type and search it, and see importance and when each
  memory was last recalled.
- **Reflections:** insight trees, with the evidence each insight cites.
- **Dialogues:** full transcripts, with any commitments made.
- **Cost:** calls, tokens, cache hits, and dollars per task and per model.
- **Interview:** past interviews. Running locally, you can also ask new questions.
- **Report:** the §7.1 diffusion, relationship, and attendance report.

Use the scrubber at the top to jump to any moment. The other tabs show the state as of that moment.

## Quick start

Requires Python 3.11+. Commands are run from `smallville_lite/`.

```bash
cd smallville_lite
python -m venv .venv
.venv/Scripts/activate              # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,web]"         # core, tests, and the viewer (enough for stub runs)
pip install -e ".[local]"           # local sentence-transformers embeddings, for real runs
```

Run a free simulation with the stand-in model, then open the viewer:

```bash
smallville run --scenario party --days 2 --budget 5.00 --stub
smallville serve                    # http://127.0.0.1:8765
```

To watch a run live as it happens, start it with a pace in one terminal and the viewer in another:

```bash
smallville run --stub --days 2 --budget 5 --pace 0.4
smallville serve
```

To use the real models, set `ANTHROPIC_API_KEY` in your environment (never in a config file; the loader refuses
anything that looks like a key), then drop `--stub`:

```bash
smallville check-models                                      # confirms the model IDs (free)
smallville run --scenario party --hours 6 --budget 1.00      # short calibration run
```

Other commands:

```bash
smallville resume runs/<id> [--budget 8] [--days 3]                              # continue from the last checkpoint
smallville interview runs/<id> --agent "Klaus Mueller" [--tick 60] "Who is running for mayor?"
smallville report runs/<id> [--no-llm]                                           # rebuild the report
pytest                                                                           # offline tests; no network, no spend
```

[`smallville_lite/README.md`](smallville_lite/README.md) has the full command reference, including budget behaviour and
the run directory layout.

## Models and cost

Every call goes through one client that routes each task to a model (`smallville_lite/config/models.toml`):

- **Claude Haiku 4.5** for the frequent, small calls: importance scores, action labels, react decisions, and the report's
  judge.
- **Claude Sonnet 5** for the calls that need judgement: planning, reflection, dialogue, and interviews.

Costs are kept down in four ways:

- **Structured output:** every response is JSON checked against a Pydantic schema. Bad output gets one repair turn and
  truncated output gets one retry, so nothing parses free text.
- **Prompt caching:** each agent's stable prefix (town rules, world, and the agent's summary) is cached. That's about
  1.3k tokens per agent.
- **Batching:** small calls are batched, and call chains the original code split up are merged.
- **A hard budget:** before every request, the client checks a worst-case estimate against the run's `--budget` and
  stops cleanly instead of overspending. The run can be resumed with a higher budget.

The stub estimate for a full 2-day, 4-agent run is about $3.30, not counting thinking tokens. Real numbers will come
from the first calibration run.

## How it differs from the paper

This is a lighter port, and every intentional change is listed with its reason in
[DESIGN.md §10](DESIGN.md#10-intentional-deviations-from-the-paper-and-why). The main ones:

- **Text world instead of a tile map.** Agents perceive everything at their current place, and travel between places
  takes a fixed number of ticks.
- **Fewer, merged calls.** Place choices are made inside the planning calls, the agent summary is one call instead of
  three, and react decisions use retrieved memories directly.
- **Whole conversations inside one tick.** Each line is still generated per speaker, as in the paper, but game time is
  charged afterwards.
- **A lower reflection threshold** (75 instead of 150), because 4 agents on 10-minute ticks see far fewer events than
  25 agents on 10-second steps.

Where the original code differs from the paper, this project follows the paper. The differences are mapped in
[ARCHITECTURE.md §11](ARCHITECTURE.md#11-code-vs-paper-differences).

## Repository layout

```
ARCHITECTURE.md        map of the original Generative Agents code, with file:line references
DESIGN.md              this project's design: world, time, cognition, model layer, event log, deviations
PROGRESS.md            current status, known issues, next steps
docs/                  images for this README
smallville_lite/
  config/              default settings and model routing / prices
  scenarios/party/     the party scenario: agents, world, map layout, facts to track
  scenarios/_shared/   the paper's App. B interview questions
  src/smallville_lite/
    llm/               model client, prompts (versioned .md templates), schemas, budget, stub model
    embed/             embeddings: local, Voyage, or stub, with an sqlite cache
    memory/            memory stream and retrieval
    cognition/         perceive, importance, reflect, plan, react, converse, summary, interview
    sim/               tick engine, scenario loader, event log, checkpoints, runner
    world/             location tree and world state
    report/            post-run diffusion report
    web/               FastAPI server and the static viewer (town, timeline, memory, ...)
  scripts/             export_artifact.py: turn a run into a static, shareable replay
  tests/               110 offline tests (Python and Node)
```

## Project status

| Phase | What | State |
|---|---|---|
| 0 | Map the original code ([ARCHITECTURE.md](ARCHITECTURE.md)) | done |
| 1 | Design ([DESIGN.md](DESIGN.md)) | done |
| 2 | Model layer and cost controls | done |
| 3 | Cognitive core, tick engine, party scenario | done |
| 4 | Runner, checkpoint/resume, report, web log | done |
| 5 | Visual town: replay and live, 3D and 2D | done |
| next | First runs on the real models, then calibration | not started |

Everything so far has been built and tested in stub mode. The next steps are in [PROGRESS.md](PROGRESS.md):

1. check the model IDs and prompt caching against the live API;
2. do a short calibration run and use it to tune the reflection threshold, thinking settings, and budget;
3. check whether the news spreads and invitees attend with real model output.

## Sharing a replay

The replay linked above is a static copy of the viewer made with `scripts/export_artifact.py`. It saves one run's data
through the real server and bundles it with the viewer's page, so the result needs no backend:

```bash
cd smallville_lite
.venv/Scripts/python scripts/export_artifact.py runs/<id> <out_dir>
```

The export can't run live mode or ask new interview questions, because both need the server. The script leaves out
local file paths. The page's note describes a stub run, so edit `NOTE` in the script before exporting a real one.

## Credits

- Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein.
  *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023.
  [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- The original implementation, [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents),
  which [ARCHITECTURE.md](ARCHITECTURE.md) maps in detail. None of its code is included here. Some prompts keep the
  paper's or the original's wording for the core instruction ([DESIGN.md §9](DESIGN.md#9-prompts-port-or-rewrite)).
  The agent names and the party scenario come from the paper.
