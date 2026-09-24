"""Post-run report (paper §7.1; DESIGN §7): information diffusion, relationship formation, coordination.

Pure metric functions (``density``, ``presence_intervals``, ``attendees``, ``evidence_for``) work on
plain data and are unit-tested directly. The interview + judge layer asks the agents, the way the
paper did, and checks every "yes" against the memory stream to flag hallucinations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ..cognition.interview import interview
from ..llm import BudgetExhausted, LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import JudgeOut

if TYPE_CHECKING:
    from ..agent import Agent
    from ..sim.engine import Simulation
    from ..sim.scenario import Scenario


# ---------------------------------------------------------------------------
# pure metrics

def density(agents: Sequence[str], knows: dict[tuple[str, str], bool]) -> tuple[list[tuple[str, str]], float]:
    """Undirected mutual-knowledge graph (paper §7.1): edge if both know of each other.
    Density = 2|E| / (|V|(|V|-1))."""
    edges = [(a, b) for a, b in itertools.combinations(sorted(agents), 2) if knows.get((a, b)) and knows.get((b, a))]
    v = len(agents)
    return edges, (2 * len(edges) / (v * (v - 1)) if v > 1 else 0.0)


def presence_intervals(events: Iterable[dict[str, Any]], place_of, end: datetime) -> dict[str, list[tuple[str, datetime, datetime]]]:
    """Fold agent_init / move_started / move_arrived into (place, from, to) intervals per agent."""
    current: dict[str, tuple[str | None, datetime]] = {}
    out: dict[str, list[tuple[str, datetime, datetime]]] = {}
    start_time: datetime | None = None
    for e in events:
        t = datetime.fromisoformat(e["game_time"]) if e.get("game_time") else None
        if e["type"] == "world_init":
            start_time = datetime.fromisoformat(e["data"]["start"])
        elif e["type"] == "agent_init":
            current[e["agent"]] = (place_of(e["data"]["start"]), t or start_time)
        elif e["type"] in ("move_started", "move_arrived") and e["agent"] in current and t is not None:
            place, since = current[e["agent"]]
            if place is not None and t > since:
                out.setdefault(e["agent"], []).append((place, since, t))
            current[e["agent"]] = (e["data"]["place"] if e["type"] == "move_arrived" else None, t)
    for agent, (place, since) in current.items():
        if place is not None and since is not None and end > since:
            out.setdefault(agent, []).append((place, since, end))
    return out


def attendees(intervals: dict[str, list[tuple[str, datetime, datetime]]], place: str, start: datetime, end: datetime,
              min_minutes: int) -> dict[str, int]:
    """Agents present at ``place`` during [start, end) for at least ``min_minutes``; value = minutes present."""
    out = {}
    for agent, items in intervals.items():
        minutes = sum(max(0.0, (min(e, end) - max(s, start)).total_seconds() / 60) for p, s, e in items if p == place)
        if minutes >= min_minutes:
            out[agent] = int(minutes)
    return out


def evidence_for(agent: "Agent", keywords: Sequence[str]) -> dict[str, Any] | None:
    """The earliest memory that mentions the fact: the grounding for a 'yes' (paper §7.1 hallucination check)."""
    kws = [k.lower() for k in keywords]
    for node in agent.memory.nodes:
        if node.type == "observation" and any(k in node.description.lower() for k in kws):
            return {"node_id": node.id, "subtype": node.subtype, "created": node.created.isoformat(),
                    "heard_from": node.meta.get("with"), "text": node.description}
    return None


# ---------------------------------------------------------------------------
# interviews + judge

@dataclass
class Asked:
    agent: str
    question: str
    answer: str
    knows: bool | None
    cited: list[str]


def ask(sim: "Simulation", agent: "Agent", question: str, keywords: Sequence[str], source: str) -> Asked:
    mind = sim.mind
    now = mind.time()
    result = interview(mind, agent, question, now, remember=False, source=source)
    if not result.ok:
        return Asked(agent.name, question, result.answer, None, [])
    prompt = load_prompt("judge")
    try:
        verdict = mind.llm.call(
            "judge", output=JudgeOut, agent=agent.name, prompt_id=prompt.ref,
            user=prompt.render(name=agent.name, question=question, answer=result.answer),
            hints={"answer": result.answer, "keywords": list(keywords)},
        ).value
        knows: bool | None = verdict.knows
    except LLMFailure as exc:
        mind.fallback("judge", agent, exc.reason, "unknown")
        knows = None
    return Asked(agent.name, question, result.answer, knows, result.cited_node_ids)


def relationship_survey(sim: "Simulation", source: str) -> dict[str, Any]:
    """'Do you know of <name>?' for every ordered pair (paper §7.1)."""
    names = sorted(sim.agents)
    answers = {}
    for a, b in itertools.permutations(names, 2):
        asked = ask(sim, sim.agents[a], f"Do you know of {b}?", [b.split()[0], b], source)
        answers[f"{a}|{b}"] = {"answer": asked.answer, "knows": asked.knows}
    return answers


def _pairs(survey: dict[str, Any]) -> dict[tuple[str, str], bool]:
    return {tuple(k.split("|")): bool(v["knows"]) for k, v in survey.items()}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# the report

def build_report(
    sim: "Simulation",
    scenario: "Scenario",
    events: list[dict[str, Any]],
    *,
    baseline: dict[str, Any] | None,
    use_llm: bool = True,
    run_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = sorted(sim.agents)
    end_time = sim.clock.time_at(sim.tick)
    report: dict[str, Any] = {"run": run_info or {}, "scenario": scenario.name, "ended_at": end_time.isoformat(),
                              "interviews": use_llm, "facts": {}, "relationships": {}, "coordination": {}}
    stopped = None

    def maybe_ask(agent: "Agent", question: str, keywords: Sequence[str], source: str) -> Asked | None:
        nonlocal stopped
        if not use_llm or stopped:
            return None
        try:
            return ask(sim, agent, question, keywords, source)
        except BudgetExhausted as exc:
            stopped = str(exc)
            return None

    # Information diffusion
    knew: dict[str, set[str]] = {}
    for fact in scenario.facts:
        rows = {}
        for name in names:
            agent = sim.agents[name]
            ev = evidence_for(agent, fact["keywords"])
            asked = maybe_ask(agent, fact["question"], fact["keywords"], "report-end")
            claims = asked.knows if asked else None
            knows = claims if claims is not None else ev is not None
            rows[name] = {
                "answer": asked.answer if asked else None, "claims": claims, "grounded": ev is not None,
                "hallucinated": bool(claims) and ev is None, "knows": knows,
                "origin": name == fact["origin"], "evidence": ev,
            }
        knew[fact["id"]] = {n for n, r in rows.items() if r["knows"]}
        k = len(knew[fact["id"]])
        report["facts"][fact["id"]] = {"question": fact["question"], "origin": fact["origin"], "agents": rows,
                                       "knew": k, "knew_pct": round(100 * k / len(names), 1)}

    # Relationship formation
    start_pairs = _pairs(baseline) if baseline else None
    end_survey: dict[str, Any] | None = None
    if use_llm:
        try:
            end_survey = relationship_survey(sim, "report-end") if not stopped else None
        except BudgetExhausted as exc:
            stopped = str(exc)
    for label, pairs in (("start", start_pairs), ("end", _pairs(end_survey) if end_survey else None)):
        if pairs is None:
            report["relationships"][label] = None
            continue
        edges, eta = density(names, pairs)
        report["relationships"][label] = {"edges": [list(e) for e in edges], "density": round(eta, 3),
                                          "directed": {f"{a}|{b}": v for (a, b), v in pairs.items()}}

    # Coordination
    intervals = presence_intervals(events, scenario.world.place_of, end_time)
    for ev in scenario.events:
        start, end = datetime.fromisoformat(ev["start"]), datetime.fromisoformat(ev["end"])
        present = attendees(intervals, ev["place"], start, end, sim.mind.sim.tick_minutes)
        informed = knew.get(ev.get("fact", ""), set())
        entry: dict[str, Any] = {
            "what": ev["what"], "place": ev["place"], "start": ev["start"], "end": ev["end"],
            "happened": end <= end_time, "attendees": present,
            "knew_and_attended": sorted(set(present) & informed),
            "knew_but_absent": sorted(informed - set(present)) if end <= end_time else [],
            "followups": {},
        }
        for name in entry["knew_but_absent"]:
            asked = maybe_ask(sim.agents[name], f"Did you {ev['what']}? Why or why not?", [], "report-followup")
            if asked:
                entry["followups"][name] = asked.answer
        report["coordination"][ev["id"]] = entry

    report["usage"] = sim.mind.llm.usage.as_dict()
    if stopped:
        report["stopped"] = stopped
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Run report: {report['scenario']}", ""]
    run = report.get("run") or {}
    if run:
        lines.append(f"Run `{run.get('run_id', '?')}`, ended {report['ended_at']} ({run.get('reason', '?')}); "
                     f"spent ${run.get('spent_usd', 0):.4f}.")
    if not report["interviews"]:
        lines.append("\n_No interviews: knowledge below is based on memory evidence only._")
    if report.get("stopped"):
        lines.append(f"\n_Report interviews stopped early: {report['stopped']}_")

    lines += ["", "## Information diffusion (paper §7.1)", ""]
    for fid, f in report["facts"].items():
        lines.append(f"### {fid}: \"{f['question']}\" (origin: {f['origin']})")
        lines.append(f"Knew at the end: **{f['knew']} of {len(f['agents'])} ({f['knew_pct']}%)**")
        lines += ["", "| Agent | Knows | Grounded in memory | Hallucinated | Heard from | Answer |", "|---|---|---|---|---|---|"]
        for name, r in f["agents"].items():
            ev = r["evidence"] or {}
            heard = "(origin)" if r["origin"] else (ev.get("heard_from") or ("-" if not ev else ev.get("subtype")))
            ans = (r["answer"] or "").replace("|", "/").replace("\n", " ")[:120]
            lines.append(f"| {name} | {'yes' if r['knows'] else 'no'} | {'yes' if r['grounded'] else 'no'} | "
                         f"{'**yes**' if r['hallucinated'] else 'no'} | {heard} | {ans} |")
        lines.append("")

    lines += ["## Relationships (network density, paper §7.1)", ""]
    rel = report["relationships"]
    for label in ("start", "end"):
        r = rel.get(label)
        if r is None:
            lines.append(f"- {label}: not measured")
        else:
            edges = ", ".join(f"{a.split()[0]}–{b.split()[0]}" for a, b in r["edges"]) or "none"
            lines.append(f"- {label}: density **{r['density']}** (edges: {edges})")
    lines.append("")

    lines += ["## Coordination", ""]
    for eid, c in report["coordination"].items():
        lines.append(f"### {eid}: {c['what']} ({c['start']} to {c['end']})")
        if not c["happened"]:
            lines.append("The run ended before this event finished.")
        present = ", ".join(f"{n} ({m} min)" for n, m in c["attendees"].items()) or "nobody"
        lines.append(f"- Present at {c['place']}: {present}")
        lines.append(f"- Knew and attended: {', '.join(c['knew_and_attended']) or 'none'}")
        lines.append(f"- Knew but absent: {', '.join(c['knew_but_absent']) or 'none'}")
        for name, ans in c["followups"].items():
            lines.append(f"  - {name}: \"{ans}\"")
        lines.append("")

    usage = report.get("usage", {})
    total = usage.get("total", {})
    lines += ["## Usage", "", f"LLM calls: {total.get('calls', 0)}, cost ${usage.get('total_cost_usd', 0):.4f}", ""]
    lines += ["| Task | Calls | Cost $ |", "|---|---|---|"]
    for task, b in sorted(usage.get("by_task", {}).items(), key=lambda kv: -kv[1]["cost_usd"]):
        lines.append(f"| {task} | {b['calls']} | {b['cost_usd']:.4f} |")
    return "\n".join(lines) + "\n"


__all__ = ["attendees", "build_report", "density", "evidence_for", "presence_intervals", "relationship_survey",
           "render_markdown"]
