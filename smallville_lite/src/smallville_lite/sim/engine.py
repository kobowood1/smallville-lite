"""Tick loop (DESIGN §3.1): advance transit -> PERCEIVE all -> DECIDE each -> COMMIT.

Everything an agent perceives comes from the start-of-tick state; decisions made during the
tick take effect in the world immediately but are only observable from the next tick. The
decision order is sorted by name and rotated each tick for fairness. A conversation claims both
participants for its duration, so one agent can never be in two conversations at once.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

from ..agent import Agent, CurrentAction
from ..cognition.converse import converse
from ..cognition.perceive import observe, perceive
from ..cognition.plan import ensure_decomposed, plan_new_day, recap_day, replan, resolve_action
from ..cognition.react import decide
from ..cognition.reflect import reflect, should_reflect
from ..llm import BudgetExhausted
from ..world import ActionState, Transit, WorldState
from .events import EventLog

if TYPE_CHECKING:
    from ..cognition import Mind


class Simulation:
    def __init__(self, mind: "Mind", agents: list[Agent], state: WorldState, log: EventLog, *, tick: int = 0) -> None:
        self.mind = mind
        self.agents = {a.name: a for a in agents}
        self.state = state
        self.log = log
        self.tick = tick
        self._conv_counter = 0
        # Called after every committed tick (and after a night skip); the runner saves a checkpoint here.
        self.on_commit: Callable[["Simulation"], None] | None = None

    @property
    def clock(self):
        return self.mind.clock

    @property
    def now(self) -> datetime:
        return self.clock.time_at(self.tick)

    # ------------------------------------------------------------------
    def run(self, until_tick: int) -> None:
        while self.tick < until_tick:
            skip_to = self._night_skip_target(until_tick)
            if skip_to is not None:
                self.log.set_time(self.tick, self.now)
                self.log.emit("time_skipped", {"from_tick": self.tick, "to_tick": skip_to, "reason": "all_asleep"})
                self.log.commit_tick()
                self.tick = skip_to
                if self.on_commit:
                    self.on_commit(self)
                continue
            self.step()

    def step(self) -> None:
        now = self.now
        self.mind.now = now
        self.log.set_time(self.tick, now)
        try:
            self._tick(now)
        except BudgetExhausted:
            self.log.abort_tick()
            raise
        self.log.commit_tick()
        self.tick += 1
        if self.on_commit:
            self.on_commit(self)

    # ------------------------------------------------------------------
    def _tick(self, now: datetime) -> None:
        mind, state = self.mind, self.state
        # 0. arrivals
        for name in sorted(self.agents):
            pos = state.positions[name]
            if pos.transit is not None and pos.transit.arrive_tick <= self.tick:
                pos.place, pos.area, pos.transit = pos.transit.destination, None, None
                self.agents[name].scratch.visited.add(pos.place)
                self.log.emit("move_arrived", {"place": pos.place}, agent=name)

        # 1. PERCEIVE against the start-of-tick state
        snapshot = {name: observe(state, name) for name in self.agents}
        perceived = {}
        for name in sorted(self.agents):
            agent = self.agents[name]
            if agent.is_asleep(now) or agent.is_engaged(now):
                continue
            perceived[name] = perceive(mind, agent, snapshot[name], now)

        # 2. DECIDE, in rotated order
        names = sorted(self.agents)
        k = self.tick % len(names)
        for name in names[k:] + names[:k]:
            self._decide(self.agents[name], now, perceived.get(name, []))

        # 3. COMMIT
        if self.tick % self.mind.sim.snapshot_every_ticks == 0:
            self.log.emit("state_snapshot", state.snapshot())

    def _decide(self, agent: Agent, now: datetime, perceived) -> None:
        mind = self.mind
        if agent.is_engaged(now):
            return
        plan = agent.scratch.day_plan
        if plan is None or plan.day != now.date():
            if plan is not None:
                recap_day(mind, agent, plan.day)
            mind.summaries.notify(agent.name, "new_day")
            plan_new_day(mind, agent, now.date())

        if should_reflect(mind, agent):
            reflect(mind, agent, now)

        pos = self.state.positions[agent.name]
        if perceived and not agent.is_asleep(now) and not pos.in_transit:
            reaction = decide(mind, agent, perceived, now)
            if reaction is not None:
                if reaction.kind == "talk":
                    target = self.agents.get(reaction.observation.subject)
                    if target is not None and self._can_talk(agent, target, now):
                        self._talk(agent, target, reaction.reaction, now)
                        return
                elif reaction.kind == "change_plan":
                    replan(mind, agent, now, cause=reaction.reaction, cause_ref=reaction.node.id)
        self._update_action(agent, now)

    # ------------------------------------------------------------------
    def _can_talk(self, a: Agent, b: Agent, now: datetime) -> bool:
        pa, pb = self.state.positions[a.name], self.state.positions[b.name]
        if pa.in_transit or pb.in_transit or pa.place != pb.place:
            return False
        if b.is_asleep(now) or b.is_engaged(now) or a.is_engaged(now):
            return False
        return a.scratch.cooldowns.get(b.name, now) <= now and b.scratch.cooldowns.get(a.name, now) <= now

    def _talk(self, a: Agent, b: Agent, reason: str, now: datetime) -> None:
        mind = self.mind
        self._conv_counter += 1
        conv_id = f"conv-{self.tick}-{self._conv_counter}"
        place = self.state.positions[a.name].place
        conv = converse(mind, a, b, place, reason, now, conv_id)
        minutes = max(1, conv.minutes * mind.sim.minutes_per_utterance)
        until = now + self.clock.step * self.clock.ticks_for(minutes)
        cooldown = until + timedelta(minutes=mind.sim.pair_cooldown_minutes)
        for me, other in ((a, b), (b, a)):
            me.scratch.engaged_until = until
            me.scratch.cooldowns[other.name] = cooldown
            topic = conv.topic_label or "this and that"
            action = CurrentAction(id=f"{conv_id}-{me.name}", activity=f"chatting with {other.name} about {topic}",
                                   place=place, kind="chat", emoji="💬", label="chatting", start=now, minutes=minutes)
            self._set_action(me, action, now)
        for me, other in ((a, b), (b, a)):
            if conv.replan.get(me.name):
                commitments = "; ".join(conv.commitments.get(me.name, [])) or conv.summary
                replan(mind, me, until, cause=f"after talking with {other.name}: {commitments}", cause_ref=conv_id)

    # ------------------------------------------------------------------
    def _update_action(self, agent: Agent, now: datetime) -> None:
        plan = agent.scratch.day_plan
        block = plan.block_at(now) if plan else None
        if block is not None:
            guard = 0
            while not block.is_sleep and (block.decomposed_until or block.start) <= now < block.end and guard < 20:
                ensure_decomposed(self.mind, agent, block, now)
                guard += 1
        action = resolve_action(self.mind, agent, now)
        current = agent.scratch.action
        if current is None or current.id != action.id:
            self._emit_skipped_steps(agent, action, now)
            self._set_action(agent, action, now)
        if not self._depart_early(agent, action, now):
            self._move_if_needed(agent, action, now)

    def _depart_early(self, agent: Agent, action: CurrentAction, now: datetime) -> bool:
        """Leave in time to arrive when the next action at another place starts (instead of one trip late)."""
        pos = self.state.positions[agent.name]
        if pos.in_transit or pos.place != action.place:
            return False
        world = self.mind.world
        horizon = max([world.travel_default, *world.travel.values()])
        for k in range(1, horizon + 1):
            future = resolve_action(self.mind, agent, now + self.clock.step * k)
            if future.place == pos.place or future.kind == "chat":
                continue
            if world.travel_ticks(pos.place, future.place) >= k:
                self._start_travel(agent, future.place)
                return True
            return False
        return False

    def _emit_skipped_steps(self, agent: Agent, action: CurrentAction, now: datetime) -> None:
        """Steps shorter than a tick still get an action_started event at their exact time."""
        plan = agent.scratch.day_plan
        since = agent.scratch.emitted_until
        if plan is None or since is None:
            return
        for b in plan.blocks:
            for st in b.steps:
                if since < st.start < (action.start or now) and st.id != action.id:
                    self.log.emit("action_started", {
                        "action_id": st.id, "description": f"{agent.name} is {st.activity}", "emoji": st.emoji,
                        "label": st.label, "place": b.place, "object": st.object, "object_state": st.object_state,
                        "start": st.start.isoformat(), "duration_min": st.minutes, "kind": "planned", "brief": True,
                    }, agent=agent.name, game_time=st.start.isoformat())

    def _set_action(self, agent: Agent, action: CurrentAction, now: datetime) -> None:
        state, name = self.state, agent.name
        prev = agent.scratch.action
        if prev is not None and prev.object and prev.object != action.object and state.object_users.get(prev.object) == name:
            if state.set_object_state(prev.object, None, None):
                self.log.emit("object_state_changed", {"object_path": prev.object, "state": state.object_states[prev.object], "by": None})
        agent.scratch.action = action
        agent.scratch.emitted_until = action.start or now
        pos = state.positions[name]
        state.actions[name] = ActionState(
            action_id=action.id, description=f"{name} is {action.activity}", activity=action.activity,
            emoji=action.emoji, label=action.label, object=action.object, object_state=action.object_state,
            kind=action.kind,
        )
        if action.object and not pos.in_transit and pos.place == action.place:
            self._use_object(agent, action)
        self.log.emit("action_started", {
            "action_id": action.id, "description": f"{name} is {action.activity}", "emoji": action.emoji,
            "label": action.label, "place": action.place, "object": action.object, "object_state": action.object_state,
            "start": (action.start or now).isoformat(), "duration_min": action.minutes, "kind": action.kind,
        }, agent=name, game_time=(action.start or now).isoformat())

    def _use_object(self, agent: Agent, action: CurrentAction) -> None:
        state = self.state
        pos = state.positions[agent.name]
        pos.area = "/".join(action.object.split("/")[:-1]) if action.object.count("/") > 1 else None
        if action.object_state and state.set_object_state(action.object, action.object_state, agent.name):
            self.log.emit("object_state_changed", {"object_path": action.object, "state": action.object_state, "by": agent.name})

    def _move_if_needed(self, agent: Agent, action: CurrentAction, now: datetime) -> None:
        pos = self.state.positions[agent.name]
        if pos.in_transit or pos.place == action.place:
            return
        self._start_travel(agent, action.place)

    def _start_travel(self, agent: Agent, destination: str) -> None:
        pos = self.state.positions[agent.name]
        ticks = self.mind.world.travel_ticks(pos.place, destination)
        pos.transit = Transit(origin=pos.place, destination=destination, depart_tick=self.tick, arrive_tick=self.tick + ticks)
        self.state.actions[agent.name] = ActionState(
            action_id=f"travel-{self.tick}-{agent.name}", description=f"{agent.name} is on the way to {destination}",
            activity=f"on the way to {destination}", emoji="🚶", label="walking", kind="travel",
        )
        self.log.emit("move_started", {"from": pos.transit.origin, "to": pos.transit.destination,
                                       "depart_tick": pos.transit.depart_tick, "arrive_tick": pos.transit.arrive_tick},
                      agent=agent.name)

    # ------------------------------------------------------------------
    def _night_skip_target(self, until_tick: int) -> int | None:
        """If everyone is asleep at home with today's plan made, jump to the next wake-up or midnight."""
        now = self.now
        targets = []
        for a in self.agents.values():
            plan = a.scratch.day_plan
            pos = self.state.positions[a.name]
            if plan is None or plan.day != now.date() or not a.is_asleep(now) or a.is_engaged(now):
                return None
            if pos.in_transit or pos.place != a.identity.home:
                return None
            nxt = plan.wake if now < plan.wake else datetime.combine(now.date() + timedelta(days=1), datetime.min.time())
            targets.append(nxt)
        target_tick = min(until_tick, self.clock.tick_at(min(targets)))
        return target_tick if target_tick > self.tick + 1 else None
