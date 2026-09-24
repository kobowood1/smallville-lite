"""Recursive planning (paper §4.3, App. A; DESIGN §5.4).

day plan (5-8 broad items)  ->  hour blocks with places  ->  5-15 min steps, decomposed just in time
Replanning regenerates the blocks from *now* to the end of the day.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Sequence

from ..agent import CurrentAction, DayPlan, HourBlock, Step, is_sleep_activity
from ..clock import fmt_date, fmt_time, parse_hhmm
from ..llm import LLMFailure
from ..llm.prompts import load_prompt
from ..llm.schemas import ActionLabels, DayPlanOut, RecapOut, detail_model, hour_blocks_model
from ..memory import NewMemory, numbered
from .importance import score_importance
from .mind import Mind

if TYPE_CHECKING:
    from ..agent import Agent

MIN_BLOCK_MINUTES = 15


# ---------------------------------------------------------------------------
# helpers

def snap5(t: datetime) -> datetime:
    """Round to the nearest 5 minutes (plan steps are 5/10/15 minutes)."""
    minutes = t.hour * 60 + t.minute + (1 if t.second >= 30 else 0)
    snapped = 5 * round(minutes / 5)
    return datetime.combine(t.date(), time()) + timedelta(minutes=snapped)


def _hhmm(t: datetime, day: date) -> str:
    return "24:00" if t.date() > day and t.hour == 0 and t.minute == 0 else fmt_time(t)


def _commitments_for(agent: "Agent", day: date) -> list[str]:
    return [n.description for n in agent.memory.of_type("plan", "commitment") if n.meta.get("date") == day.isoformat()]


def _format_blocks(blocks: Sequence[HourBlock], day: date) -> str:
    return "\n".join(f"{_hhmm(b.start, day)}-{_hhmm(b.end, day)} {b.activity} @ {b.place}" for b in blocks) or "(none)"


def _object_labels(mind: Mind, place: str) -> dict[str, str]:
    """Unique label -> object path. Duplicate names get their area: 'bed (Maria's room)'."""
    objs = mind.world.objects_in(place)
    counts: dict[str, int] = {}
    for o in objs:
        counts[o.name] = counts.get(o.name, 0) + 1
    labels = {}
    for o in objs:
        parent = o.path.split("/")[-2]
        labels[o.name if counts[o.name] == 1 else f"{o.name} ({parent})"] = o.path
    return labels


def _store_plan_nodes(mind: Mind, agent: "Agent", now: datetime, memories: list[NewMemory]) -> None:
    if not memories:
        return
    scores = score_importance(mind, agent, [m.description for m in memories])
    for m, s in zip(memories, scores):
        m.importance = s
    agent.memory.add_many(now, memories)


# ---------------------------------------------------------------------------
# day level

def plan_new_day(mind: Mind, agent: "Agent", day: date) -> DayPlan:
    now = mind.time()
    name, first = agent.name, agent.identity.first_name
    s = mind.sim
    retrieved = mind.retrieve(agent, f"{name}'s plans and commitments for {fmt_date(day)}", k=10, purpose="plan")
    commitments = _commitments_for(agent, day)
    prompt = load_prompt("plan_day")

    def check(v: DayPlanOut) -> None:
        wake, sleep = parse_hhmm(v.wake_time, day), parse_hhmm(v.sleep_time, day)
        if not wake < sleep:
            raise ValueError("wake_time must be earlier than sleep_time")
        if not s.day_plan_items_min <= len(v.items) <= s.day_plan_items_max:
            raise ValueError(f"give between {s.day_plan_items_min} and {s.day_plan_items_max} items")
        for it in v.items:
            parse_hhmm(it.time, day)

    try:
        out = mind.llm.call(
            "plan_day", output=DayPlanOut, agent=name, prefix=mind.prefix(agent), prompt_id=prompt.ref, validate=check,
            user=prompt.render(
                first=first, name=name, date=fmt_date(day),
                recap=agent.scratch.last_recap or "(this is the first day of the simulation)",
                memories=numbered(retrieved), commitments="\n".join(f"- {c}" for c in commitments) or "(none)",
                wake_hint=agent.identity.wake_hint or "(no fixed routine)",
                n_min=s.day_plan_items_min, n_max=s.day_plan_items_max,
            ),
            hints={"agent": name, "date": day.isoformat(), "wake_hint": agent.identity.wake_hint,
                   "bio": agent.identity.bio, "commitments": commitments, "n_min": s.day_plan_items_min},
        ).value
        wake, sleep = snap5(parse_hhmm(out.wake_time, day)), snap5(parse_hhmm(out.sleep_time, day))
        items = sorted((snap5(parse_hhmm(i.time, day)), i.description) for i in out.items)
    except LLMFailure as exc:
        wake, sleep = datetime.combine(day, time(7)), datetime.combine(day, time(22))
        items = [(wake, "wake up and get ready"), (wake + timedelta(hours=2), "go about the usual day"),
                 (datetime.combine(day, time(18)), "have dinner"), (datetime.combine(day, time(21)), "wind down")]
        mind.fallback("plan_day", agent, exc.reason, items)

    items_text = "\n".join(f"{fmt_time(t)} {d}" for t, d in items)
    blocks = make_blocks(mind, agent, day, "plan_hours", wake, sleep, items_text=items_text,
                         commitments=commitments, cause=None, remaining=None)
    plan = DayPlan(day=day, wake=wake, sleep=sleep, items=items, blocks=blocks)
    agent.scratch.day_plan = plan

    memories = [NewMemory("plan", "day_plan", f"{name} planned for {fmt_date(day)}: at {fmt_time(t)}, {d}",
                          importance=0, meta={"date": day.isoformat(), "time": fmt_time(t)}) for t, d in items]
    memories += [NewMemory("plan", "hour_plan",
                           f"{name} plans to be {b.activity} at {b.place} from {fmt_time(b.start)} to {_hhmm(b.end, day)} on {fmt_date(day)}",
                           importance=0, meta={"date": day.isoformat(), "block_id": b.id}) for b in blocks]
    _store_plan_nodes(mind, agent, now, memories)
    mind.sink.emit("plan_created", {
        "level": "day", "date": day.isoformat(), "wake": wake.isoformat(), "sleep": sleep.isoformat(),
        "items": [{"start": t.isoformat(), "description": d} for t, d in items],
    }, agent=name, game_time=now.isoformat())
    _emit_blocks(mind, agent, "hour", day, blocks)
    return plan


def make_blocks(
    mind: Mind,
    agent: "Agent",
    day: date,
    task: str,
    start: datetime,
    end: datetime,
    *,
    items_text: str,
    commitments: list[str],
    cause: str | None,
    remaining: str | None,
) -> list[HourBlock]:
    """plan_hours / replan: contiguous blocks covering [start, end) at known places."""
    first = agent.identity.first_name
    places = tuple(mind.known_places(agent))
    model = hour_blocks_model(places)
    prompt = load_prompt(task)
    start_s, end_s = _hhmm(start, day), _hhmm(end, day)

    def parse(v: Any) -> list[tuple[datetime, datetime, str, str]]:
        rows = [(snap5(parse_hhmm(b.start, day)), snap5(parse_hhmm(b.end, day)), b.activity, b.place) for b in v.blocks]
        if not rows:
            raise ValueError("at least one block is required")
        if rows[0][0] != start:
            raise ValueError(f"the first block must start at {start_s}")
        if rows[-1][1] != end:
            raise ValueError(f"the last block must end at {end_s}")
        for (s1, e1, *_), (s2, *_rest) in zip(rows, rows[1:]):
            if e1 != s2:
                raise ValueError(f"blocks must be contiguous: one ends at {fmt_time(e1)} but the next starts at {fmt_time(s2)}")
        for s1, e1, act, _ in rows:
            if (e1 - s1) < timedelta(minutes=min(MIN_BLOCK_MINUTES, int((end - start).total_seconds() // 60))):
                raise ValueError(f"block '{act}' is shorter than {MIN_BLOCK_MINUTES} minutes")
        return rows

    try:
        out = mind.llm.call(
            task, output=model, agent=agent.name, prefix=mind.prefix(agent), prompt_id=prompt.ref,
            validate=lambda v: parse(v) and None,
            user=prompt.render(
                first=first, date=fmt_date(day), start=start_s, end=end_s, items=items_text,
                commitments="\n".join(f"- {c}" for c in commitments) or "(none)",
                places=mind.world.render_known(places), cause=cause or "", remaining=remaining or "(none)",
                current=agent.scratch.action.place if agent.scratch.action else agent.identity.home,
            ),
            hints={"agent": agent.name, "date": day.isoformat(), "start": start_s, "end": end_s,
                   "places": list(places), "home": agent.identity.home, "items": items_text,
                   "commitments": commitments, "cause": cause},
        ).value
        rows = parse(out)
    except LLMFailure as exc:
        rows = [(start, end, "going about the day", agent.identity.home)]
        mind.fallback(task, agent, exc.reason, rows)
    return [HourBlock(id=agent.scratch.next_id("b"), start=s, end=e, activity=a, place=p,
                      source="plan" if task == "plan_hours" else "replan") for s, e, a, p in rows]


def _emit_blocks(mind: Mind, agent: "Agent", level: str, day: date, blocks: Sequence[HourBlock]) -> None:
    mind.sink.emit("plan_created", {
        "level": level, "date": day.isoformat(),
        "items": [{"id": b.id, "start": b.start.isoformat(), "end": b.end.isoformat(), "description": b.activity,
                   "place": b.place} for b in blocks],
    }, agent=agent.name, game_time=mind.time().isoformat())


# ---------------------------------------------------------------------------
# just-in-time decomposition

def ensure_decomposed(mind: Mind, agent: "Agent", block: HourBlock, now: datetime) -> None:
    """Decompose the part of ``block`` starting at ``now`` up to the horizon, if not done yet."""
    if block.is_sleep:
        block.emoji, block.label = "😴", block.label or "sleeping"
        return
    cursor = block.decomposed_until or block.start
    if cursor >= block.end or now < cursor:
        return
    chunk_end = min(block.end, cursor + timedelta(minutes=mind.sim.decompose_horizon_minutes))
    # Never leave a remainder shorter than 15 minutes at the end of the block.
    if block.end - chunk_end < timedelta(minutes=15):
        chunk_end = block.end
    minutes = int((chunk_end - cursor).total_seconds() // 60)
    steps = decompose(mind, agent, block, cursor, minutes)
    block.steps.extend(steps)
    block.decomposed_until = chunk_end
    mind.sink.emit("plan_created", {
        "level": "detail", "date": block.start.date().isoformat(), "block_id": block.id,
        "items": [{"id": st.id, "start": st.start.isoformat(), "end": st.end.isoformat(), "description": st.activity,
                   "place": block.place, "object": st.object, "emoji": st.emoji, "label": st.label} for st in steps],
    }, agent=agent.name, game_time=mind.time().isoformat())


def decompose(mind: Mind, agent: "Agent", block: HourBlock, start: datetime, minutes: int) -> list[Step]:
    first = agent.identity.first_name
    labels = _object_labels(mind, block.place)
    model = detail_model(tuple(labels))
    prompt = load_prompt("plan_detail")
    plan = agent.scratch.day_plan
    context = _format_blocks([b for b in plan.blocks if abs((b.start - block.start).total_seconds()) <= 3 * 3600], block.start.date()) if plan else ""

    def check(v: Any) -> None:
        total = sum(st.minutes for st in v.steps)
        if total != minutes:
            raise ValueError(f"step minutes add up to {total}; they must add up to exactly {minutes}")

    try:
        out = mind.llm.call(
            "plan_detail", output=model, agent=agent.name, prefix=mind.prefix(agent), prompt_id=prompt.ref,
            validate=check,
            user=prompt.render(
                first=first, date=fmt_date(start), place=block.place, activity=block.activity,
                start=fmt_time(start), end=fmt_time(start + timedelta(minutes=minutes)), minutes=minutes,
                objects=", ".join(labels) or "(none)", context=context,
            ),
            hints={"activity": block.activity, "minutes": minutes, "objects": list(labels)},
        ).value
        raw = [(st.minutes, st.activity, labels.get(st.object), st.object_state or None) for st in out.steps]
    except LLMFailure as exc:
        raw, left = [], minutes
        while left > 0:
            m = 15 if left >= 15 else left
            raw.append((m, block.activity, None, None))
            left -= m
        mind.fallback("plan_detail", agent, exc.reason, raw)

    steps, t = [], start
    for m, act, obj, state in raw:
        steps.append(Step(id=agent.scratch.next_id("s"), start=t, minutes=m, activity=act, object=obj,
                          object_state=state if obj else None))
        t += timedelta(minutes=m)
    label_steps(mind, agent, steps)
    return steps


def label_steps(mind: Mind, agent: "Agent", steps: list[Step]) -> None:
    """Emoji + short label for each step, one batched Haiku call (paper §3.1.1 emoji display)."""
    if not steps:
        return
    prompt = load_prompt("label_actions")
    acts = [s.activity for s in steps]

    def check(v: ActionLabels) -> None:
        if len(v.labels) != len(acts):
            raise ValueError(f"expected exactly {len(acts)} labels, got {len(v.labels)}")

    try:
        out = mind.llm.call(
            "label_actions", output=ActionLabels, agent=agent.name, prompt_id=prompt.ref, validate=check,
            user=prompt.render(n=len(acts), actions="\n".join(f"{i}. {a}" for i, a in enumerate(acts, 1))),
            hints={"activities": acts},
        ).value
        for st, lab in zip(steps, out.labels):
            st.emoji, st.label = lab.emoji.strip()[:8] or "🙂", lab.label.strip()[:40]
    except LLMFailure as exc:
        for st in steps:
            st.emoji, st.label = ("😴" if is_sleep_activity(st.activity) else "🙂"), " ".join(st.activity.split()[:4])
        mind.fallback("label_actions", agent, exc.reason, "default emoji")


# ---------------------------------------------------------------------------
# replanning and recap

def replan(mind: Mind, agent: "Agent", now: datetime, cause: str, cause_ref: str | None = None) -> bool:
    """Regenerate the rest of today's blocks from ``now`` (paper §4.3.1). Returns False if nothing to do."""
    plan = agent.scratch.day_plan
    if plan is None or now >= plan.sleep - timedelta(minutes=MIN_BLOCK_MINUTES):
        return False
    now = snap5(now)
    kept, removed = [], []
    for b in plan.blocks:
        if b.end <= now:
            kept.append(b)
        elif b.start < now:
            b.end = now
            b.steps = [st for st in b.steps if st.start < now]
            if b.steps and b.steps[-1].end > now:
                b.steps[-1].minutes = int((now - b.steps[-1].start).total_seconds() // 60)
            b.decomposed_until = min(b.decomposed_until or now, now)
            kept.append(b)
        else:
            removed.append(b)
    remaining = [b for b in plan.blocks if b.end > now or b in removed]
    new_blocks = make_blocks(
        mind, agent, plan.day, "replan", now, plan.sleep,
        items_text="\n".join(f"{fmt_time(t)} {d}" for t, d in plan.items),
        commitments=_commitments_for(agent, plan.day), cause=cause,
        remaining=_format_blocks([b for b in remaining if b.start >= now] or removed, plan.day),
    )
    plan.blocks = kept + new_blocks
    plan.revision += 1
    _store_plan_nodes(mind, agent, mind.time(), [
        NewMemory("plan", "hour_plan",
                  f"{agent.name} now plans to be {b.activity} at {b.place} from {fmt_time(b.start)} to {_hhmm(b.end, plan.day)} on {fmt_date(plan.day)}",
                  importance=0, meta={"date": plan.day.isoformat(), "block_id": b.id, "revision": plan.revision})
        for b in new_blocks
    ])
    mind.sink.emit("plan_revised", {
        "reason": cause, "cause": cause_ref, "from_time": now.isoformat(),
        "removed_ids": [b.id for b in removed],
        "added": [{"id": b.id, "start": b.start.isoformat(), "end": b.end.isoformat(), "description": b.activity,
                   "place": b.place} for b in new_blocks],
    }, agent=agent.name, game_time=mind.time().isoformat())
    return True


def recap_day(mind: Mind, agent: "Agent", day: date) -> str | None:
    todays = [n for n in agent.memory.nodes if n.created.date() == day and n.type != "plan"]
    if not todays:
        return None
    top = sorted(todays, key=lambda n: (-n.importance, n.created))[:30]
    top.sort(key=lambda n: n.created)
    prompt = load_prompt("recap_day")
    try:
        text = mind.llm.call(
            "recap_day", output=RecapOut, agent=agent.name, prefix=mind.prefix(agent), prompt_id=prompt.ref,
            user=prompt.render(first=agent.identity.first_name, date=fmt_date(day), memories=numbered(top)),
            hints={"memories": [n.description for n in top]},
        ).value.recap
    except LLMFailure as exc:
        text = " ".join(n.description for n in top[:5])
        mind.fallback("recap_day", agent, exc.reason, text)
    agent.scratch.last_recap = f"On {fmt_date(day)}: {text}"
    return text


# ---------------------------------------------------------------------------
# what is the agent doing now?

def resolve_action(mind: Mind, agent: "Agent", now: datetime) -> CurrentAction:
    plan = agent.scratch.day_plan
    home = agent.identity.home
    if plan is None or plan.day != now.date() or now < plan.wake or now >= plan.sleep:
        day = now.date().isoformat()
        return CurrentAction(id=f"sleep-{day}-{'am' if plan and now < plan.wake else 'pm'}", activity="sleeping",
                             place=home, kind="sleep", object=_bed(mind, agent), emoji="😴", label="sleeping")
    block = plan.block_at(now)
    if block is None:
        place = agent.scratch.action.place if agent.scratch.action else home
        return CurrentAction(id=f"idle-{now:%H%M}", activity="taking a moment", place=place, kind="planned")
    if block.is_sleep:
        return CurrentAction(id=block.id, activity=block.activity, place=block.place, kind="sleep",
                             object=_bed(mind, agent) if block.place == home else None, emoji="😴",
                             label=block.label or "sleeping", start=block.start,
                             minutes=int((block.end - block.start).total_seconds() // 60))
    for st in block.steps:
        if st.start <= now < st.end:
            return CurrentAction(id=st.id, activity=st.activity, place=block.place, kind="planned", object=st.object,
                                 object_state=st.object_state, emoji=st.emoji, label=st.label, start=st.start,
                                 minutes=st.minutes)
    return CurrentAction(id=block.id, activity=block.activity, place=block.place, kind="planned",
                         emoji=block.emoji, label=block.label or " ".join(block.activity.split()[:4]),
                         start=block.start, minutes=int((block.end - block.start).total_seconds() // 60))


def _bed(mind: Mind, agent: "Agent") -> str | None:
    area = agent.identity.home_area
    for o in mind.world.objects_in(agent.identity.home):
        if o.name == "bed" and (area is None or o.path.startswith(area + "/")):
            return o.path
    return None
