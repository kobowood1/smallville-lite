"""Scenario-aware responders for STUB mode (DESIGN §6.6).

These fake the model just well enough that a stub run exercises every mechanism end to end:
plans cover the day and honour commitments, agents meet at shared places, speakers repeat the
most important thing they remember (so seeded facts can spread), and hearing about a scenario
event produces a commitment to attend it. Outputs are deterministic given the request.

Responders read ``BackendRequest.hints`` (structured context the cognition layer attaches for the
stub only) rather than parsing prompt text.
"""

from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

from .types import BackendRequest

_EMOJI = [
    ("sleep", "😴"), ("coffee", "☕"), ("espresso", "☕"), ("cafe", "☕"), ("garden", "🌱"), ("walk", "🚶"),
    ("park", "🌳"), ("shop", "🛒"), ("grocer", "🛒"), ("study", "📚"), ("research", "📚"), ("read", "📖"),
    ("write", "✍️"), ("eat", "🍽️"), ("lunch", "🥪"), ("dinner", "🍝"), ("breakfast", "🍳"), ("party", "🎉"),
    ("decorat", "🎈"), ("chat", "💬"), ("talk", "💬"), ("clean", "🧹"), ("cook", "🍳"),
]
_ACTIVITY_AT = {
    "cafe": "having a coffee and chatting at Hobbs Cafe",
    "park": "taking a walk in Johnson Park",
    "market": "shopping for groceries at The Willows Market",
    "library": "working on assignments at the library",
}
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)


def make_responders(facts: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keywords = sorted({k.lower() for f in facts for k in f.get("keywords", [])})

    def mentions_fact(text: str) -> bool:
        low = text.lower()
        return any(k in low for k in keywords)

    # -- Haiku tasks ------------------------------------------------------
    def importance(req: BackendRequest, rng: random.Random) -> dict:
        scores = []
        for t in req.hints["texts"]:
            low = t.lower()
            if mentions_fact(t):
                scores.append(8)
            elif "committed to" in low:
                scores.append(6)
            elif "conversation with" in low:
                scores.append(5)
            elif "sleeping" in low:
                scores.append(1)
            else:
                scores.append(2 + rng.randint(0, 2))
        return {"scores": scores}

    def label_actions(req: BackendRequest, rng: random.Random) -> dict:
        labels = []
        for act in req.hints["activities"]:
            low = act.lower()
            emoji = next((e for k, e in _EMOJI if k in low), "🙂")
            labels.append({"emoji": emoji, "label": " ".join(act.split()[:3])})
        return {"labels": labels}

    def react(req: BackendRequest, rng: random.Random) -> dict:
        obs = req.hints["observations"]
        for i, o in enumerate(obs, 1):
            if o["kind"] == "agent" and rng.random() < 0.6:
                return {"react": True, "observation_index": i, "kind": "talk",
                        "reaction": f"say hello to {o['subject']} and catch up", "reason": "stub: friendly"}
        return {"react": False, "observation_index": 0, "kind": "none", "reaction": "", "reason": "stub: carry on"}

    def relationship(req: BackendRequest, rng: random.Random) -> dict:
        other = req.hints["other"]
        known = [m for m in req.hints["memories"] if other.split()[0] in m]
        text = (f"{req.hints['self']} knows {other}: {known[0]}" if known
                else f"{req.hints['self']} does not know {other} yet.")
        return {"summary": text}

    def summarize_dialogue(req: BackendRequest, rng: random.Random) -> dict:
        lines = [t for _, t in req.hints["transcript"]]
        return {"topic_label": " ".join(lines[0].split()[:5]) if lines else "a chat",
                "summary": " ".join(lines)[:600]}

    # -- planning ------------------------------------------------------------
    def plan_day(req: BackendRequest, rng: random.Random) -> dict:
        hint = req.hints.get("wake_hint") or ""
        times = [_to_minutes(m) for m in _TIME.finditer(hint)]
        wake = times[0] if times else 7 * 60
        sleep = max(times[-1], wake + 12 * 60) if len(times) > 1 else 22 * 60
        sleep = min(sleep, 24 * 60)
        items = [(wake, "wake up and complete the morning routine")]
        for c in req.hints.get("commitments", []):
            m = re.search(r"at (\d{2}):(\d{2})", c)
            if m:
                items.append((int(m.group(1)) * 60 + int(m.group(2)), c.split("committed to: ")[-1]))
        filler = ["go about the usual morning", "have lunch", "spend the afternoon out", "have dinner", "relax at home"]
        slots = [wake + 120, 12 * 60, 14 * 60, 18 * 60, sleep - 60]
        for t, d in zip(slots, filler):
            if len(items) >= max(req.hints.get("n_min", 5), 5):
                break
            if wake < t < sleep and all(abs(t - x) >= 30 for x, _ in items):
                items.append((t, d))
        items.sort()
        return {"wake_time": _hhmm(wake), "sleep_time": _hhmm(sleep),
                "items": [{"time": _hhmm(t), "description": d} for t, d in items]}

    def plan_blocks(req: BackendRequest, rng: random.Random) -> dict:
        h = req.hints
        start, end = _parse(h["start"]), _parse(h["end"])
        places: list[str] = h["places"]
        home = h["home"]
        fixed = []
        for c in h.get("commitments", []):
            m = re.search(r"at (\d{2}):(\d{2})", c)
            place = next((p for p in places if p in c), None)
            if m and place:
                t = int(m.group(1)) * 60 + int(m.group(2))
                t_end = min(end, t + 120)
                if t < start < t_end:          # already in progress: keep going until it ends
                    t = start
                if start <= t < end:
                    fixed.append((t, t_end, c.split("committed to: ")[-1].split(" on ")[0], place))
        if h.get("cause") and not any(f[0] == start for f in fixed):
            fixed.append((start, min(end, start + 30), h["cause"][:80], places[0] if not fixed else fixed[0][3]))
        order = [p for p in places if p != home]
        rng.shuffle(order)
        blocks, t, i = [], start, 0
        for f_start, f_end, act, place in sorted(fixed):
            if f_start < t:
                continue
            t = _fill(blocks, t, f_start, home, order, i)
            i += 1
            blocks.append((t, f_end, act, place))
            t = f_end
        _fill(blocks, t, end, home, order, i)
        return {"blocks": [{"start": _hhmm(s), "end": _hhmm(e), "activity": a, "place": p} for s, e, a, p in blocks]}

    def plan_detail(req: BackendRequest, rng: random.Random) -> dict:
        left, steps = req.hints["minutes"], []
        objs = req.hints["objects"] or ["none"]
        act = req.hints["activity"]
        while left > 0:
            m = 15 if left >= 15 else (10 if left >= 10 else 5)
            obj = objs[len(steps) % len(objs)]
            steps.append({"minutes": m, "activity": f"{act} (part {len(steps) + 1})", "object": obj,
                          "object_state": "in use" if obj != "none" else ""})
            left -= m
        return {"steps": steps}

    def recap_day(req: BackendRequest, rng: random.Random) -> dict:
        return {"recap": " ".join(req.hints["memories"][:4])}

    # -- reflection / summary -------------------------------------------------
    def reflect_questions(req: BackendRequest, rng: random.Random) -> dict:
        who = req.hints.get("agent", "the agent")
        base = [f"What matters most to {who} right now?", f"Who has {who} been spending time with?",
                f"What is {who} planning for the coming days?"]
        return {"questions": (base * 3)[: req.hints["n"]]}

    def reflect_insights(req: BackendRequest, rng: random.Random) -> dict:
        n_st, statements = req.hints["n_statements"], req.hints["statements"]
        out = []
        for i in range(min(req.hints["n"], n_st)):
            j = (i + 1) % n_st + 1
            out.append({"insight": f"Insight about: {statements[i][:120]}", "evidence": [i + 1, j]})
        return {"insights": out}

    def summary(req: BackendRequest, rng: random.Random) -> dict:
        bio = req.hints["bio"]
        return {"core": bio[0], "occupation": bio[1] if len(bio) > 1 else bio[0],
                "recent_progress": f"{req.hints['agent']} feels things are going reasonably well."}

    # -- dialogue / interview ------------------------------------------------
    def utterance(req: BackendRequest, rng: random.Random) -> dict:
        h = req.hints
        said = " ".join(t for _, t in h["transcript"]).lower()
        news = next((m for m in h["memories"] if mentions_fact(m) and m[:40].lower() not in said), None)
        if h["turn"] == 0:
            text = f"Hi {h['listener'].split()[0]}!" + (f" Did you hear? {news}" if news else " How are you?")
        elif news:
            text = f"Oh, and {news}"
        else:
            text = "That's good to hear." if h["turn"] < 3 else "Well, I should get going. See you around!"
        return {"utterance": text, "end_conversation": h["turn"] >= 3}

    def conversation_note(req: BackendRequest, rng: random.Random) -> dict:
        text = " ".join(t for _, t in req.hints["transcript"]).lower()
        commitments = []
        for ev in events:
            fact = next((f for f in facts if f["id"] == ev.get("fact")), None)
            kws = [k.lower() for k in (fact or {}).get("keywords", [])]
            if kws and any(k in text for k in kws):
                start = str(ev["start"])
                commitments.append({"what": ev["what"], "date": start[:10], "time": start[11:16], "place": ev["place"]})
        # replan_today stays False: the engine replans when a *new* commitment falls today.
        return {"note": f"Talked with {req.hints['other']}.", "commitments": commitments, "replan_today": False}

    def interview(req: BackendRequest, rng: random.Random) -> dict:
        mem = req.hints["memories"]
        return {"answer": mem[0] if mem else "I don't know.", "cited": [1] if mem else []}

    return {
        "importance": importance, "label_actions": label_actions, "react": react, "relationship": relationship,
        "summarize_dialogue": summarize_dialogue, "plan_day": plan_day, "plan_hours": plan_blocks,
        "replan": plan_blocks, "plan_detail": plan_detail, "recap_day": recap_day,
        "reflect_questions": reflect_questions, "reflect_insights": reflect_insights, "summary": summary,
        "utterance": utterance, "conversation_note": conversation_note, "interview": interview,
    }


def _fill(blocks: list, t: int, until: int, home: str, order: list[str], i: int) -> int:
    """Fill [t, until) with ~2h outings to rotating places, bracketed by time at home."""
    while until - t >= 15:
        span = min(120, until - t)
        if until - t - span < 15:
            span = until - t
        if not blocks or blocks[-1][3] != home and (i + len(blocks)) % 3 == 0:
            place, act = home, "spending time at home"
        else:
            place = order[(i + len(blocks)) % len(order)] if order else home
            act = next((v for k, v in _ACTIVITY_AT.items() if k in place.lower()), f"spending time at {place}")
        if blocks and blocks[-1][3] == place and blocks[-1][1] == t:
            s, _, a, p = blocks.pop()
            blocks.append((s, t + span, a, p))
        else:
            blocks.append((t, t + span, act, place))
        t += span
    if blocks and t < until:  # tiny remainder: extend the last block
        s, _, a, p = blocks.pop()
        blocks.append((s, until, a, p))
        t = until
    return t


def _to_minutes(m: re.Match) -> int:
    h, mm, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return h * 60 + mm


def _parse(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
