"""Command-line interface. Phase 2 commands: check-models, llm-demo, usage.

The simulation commands (run, resume, interview, report, serve) arrive in Phase 4.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smallville", description="smallville_lite: Generative Agents on Claude")
    parser.add_argument("--config-dir", help="directory containing default.toml and models.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-models", help="verify every configured model ID against the live Models API")

    demo = sub.add_parser("llm-demo", help="exercise routing, caching, budget and usage logging")
    mode = demo.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stub", action="store_true", help="fake LLM, zero spend")
    mode.add_argument("--live", action="store_true", help="real API calls (spends money, typically < $0.01)")
    demo.add_argument("--calls", type=int, default=4)
    demo.add_argument("--budget", type=float, default=0.05)
    demo.add_argument("--log", default=None, help="events.jsonl path (default runs/llm-demo-<time>/events.jsonl)")

    usage = sub.add_parser("usage", help="print the usage summary for an events.jsonl")
    usage.add_argument("events")

    args = parser.parse_args(argv)
    try:
        if args.command == "check-models":
            return _check_models(args)
        if args.command == "llm-demo":
            return _llm_demo(args)
        if args.command == "usage":
            return _usage(args)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 1


def _check_models(args: argparse.Namespace) -> int:
    from .llm.verify import verify_models

    cfg = load_config(args.config_dir)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("ANTHROPIC_API_KEY is not set; cannot query the Models API.", file=sys.stderr)
        return 2
    ids = {t.model for t in cfg.tasks.values()}
    results = verify_models(ids)
    for r in results:
        if r.ok:
            print(f"OK    {r.model:<30} {r.display_name or ''}  context={r.max_input_tokens} max_out={r.max_tokens}")
        else:
            print(f"FAIL  {r.model:<30} {r.error}")
    return 0 if all(r.ok for r in results) else 1


# A realistic-size static prefix so the Sonnet route can actually cache (>= 1024 tokens).
_DEMO_RULES = "\n".join(
    [
        "You are the cognition layer of a small-town social simulation. Characters perceive, remember,",
        "plan, reflect, and talk. Stay in character, use only information you are given, and never invent",
        "people or places. Answer in the JSON format requested.",
        "",
        "Town directory:",
    ]
    + [
        f"- Place {i:02d}: a {kind} with {', '.join(objs)}; open {open_}."
        for i, (kind, objs, open_) in enumerate(
            [
                ("cafe", ["a counter", "an espresso machine", "six tables", "a bulletin board"], "7am-8pm"),
                ("park", ["garden beds", "a bench", "a pond path", "a gazebo"], "all day"),
                ("market", ["produce shelves", "a checkout counter", "a notice board"], "8am-6pm"),
                ("library", ["reading tables", "bookshelves", "study carrels", "a printer"], "8am-10pm"),
                ("dorm", ["two bedrooms", "a common room", "a shared kitchen"], "residents only"),
                ("house", ["a bedroom", "a kitchen with a stove", "a living room with an armchair"], "residents only"),
            ]
            * 9
        )
    ]
)


def _llm_demo(args: argparse.Namespace) -> int:
    from .llm import BudgetExhausted, LLMFailure, PromptPrefix, build_llm
    from .llm.prompts import load_prompt
    from .llm.schemas import Ping
    from .sim.events import EventLog

    cfg = load_config(args.config_dir, overrides={"run": {"stub": bool(args.stub)}, "budget": {"max_usd": args.budget}})
    if args.live and not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2
    run_id = f"llm-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_path = Path(args.log or f"runs/{run_id}/events.jsonl")
    prompt = load_prompt("ping")

    with EventLog(log_path, run_id, autocommit=True) as log:
        llm = build_llm(cfg, sink=log)
        log.emit("run_started", {
            "run_id": run_id, "config_hash": cfg.config_hash, "seed": cfg.seed, "stub": cfg.stub,
            "models": {k: v.model for k, v in cfg.tasks.items()}, "budget_usd": cfg.budget.max_usd,
        })
        prefix = PromptPrefix(rules=_DEMO_RULES, agent_block="Agent: Demo Person (age 30). Curious and kind.")
        reason, detail = "completed", None
        try:
            for i in range(args.calls):
                task = "ping_sonnet" if i % 2 else "ping"
                label = cfg.task(task).model
                result = llm.call(task, output=Ping, prefix=prefix, prompt_id=prompt.ref,
                                  user=prompt.render(model_label=label, token=f"demo-{i}"))
                print(f"{task:<12} -> {result.value.model_dump()}  ${result.cost_usd:.6f}  "
                      f"cache_read={result.usage.cache_read_tokens} cache_write={result.usage.cache_write_tokens}")
        except BudgetExhausted as exc:
            reason, detail = "budget", str(exc)
            print(f"stopped cleanly: {exc}")
        except LLMFailure as exc:
            reason, detail = "error", str(exc)
            print(f"call failed: {exc}", file=sys.stderr)
        log.emit("run_ended", {"reason": reason, "spent_usd": round(llm.budget.spent, 6), "detail": detail})

    print()
    print(llm.usage.render())
    print(f"\nlog: {log_path}")
    return 0 if reason in ("completed", "budget") else 1


def _usage(args: argparse.Namespace) -> int:
    from .llm.usage import UsageSummary

    print(UsageSummary.from_log(args.events).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
