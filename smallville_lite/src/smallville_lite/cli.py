"""Command-line interface.

    smallville run --scenario party --days 2 --budget 5.00 [--stub]
    smallville resume runs/<id> [--budget 8] [--days 3]
    smallville report runs/<id>                   re-generate report.json / report.md
    smallville interview runs/<id> --agent "Klaus Mueller" [--tick N] "Who is running for mayor?"
    smallville serve [--runs-dir runs]            web log viewer
    smallville check-models | llm-demo | usage    model-layer utilities (Phase 2)
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

    run = sub.add_parser("run", help="run a scenario")
    run.add_argument("--scenario", default="party")
    span = run.add_mutually_exclusive_group()
    span.add_argument("--days", type=float, help="game days to simulate (default: the scenario's)")
    span.add_argument("--hours", type=float, help="game hours to simulate")
    run.add_argument("--budget", type=float, required=True, help="hard cap in USD for the whole run, report included")
    run.add_argument("--stub", action="store_true", help="fake LLM + fake embeddings: zero network, zero spend")
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--run-id")
    run.add_argument("--no-report", action="store_true")

    resume = sub.add_parser("resume", help="continue a run from its latest checkpoint")
    resume.add_argument("run_dir")
    resume.add_argument("--budget", type=float, help="new total cap in USD (default: the run's)")
    rspan = resume.add_mutually_exclusive_group()
    rspan.add_argument("--days", type=float, help="new total length in game days, from the scenario start")
    rspan.add_argument("--hours", type=float, help="new total length in game hours, from the scenario start")
    resume.add_argument("--no-report", action="store_true")

    report = sub.add_parser("report", help="re-generate the post-run report from the latest checkpoint")
    report.add_argument("run_dir")
    report.add_argument("--no-llm", action="store_true", help="memory evidence only, no interviews")
    report.add_argument("--budget", type=float, default=0.50)

    iv = sub.add_parser("interview", help="ask an agent a question against a saved memory state")
    iv.add_argument("run_dir")
    iv.add_argument("question")
    iv.add_argument("--agent", required=True)
    iv.add_argument("--tick", type=int, help="use the latest checkpoint at or before this tick (default: latest)")
    iv.add_argument("--stub", action="store_true", help="answer with the stub model even for a live run")
    iv.add_argument("--budget", type=float, default=0.10)

    serve = sub.add_parser("serve", help="web log viewer")
    serve.add_argument("--runs-dir", default="runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--stub-interviews", action="store_true", help="never spend money on viewer interviews")
    serve.add_argument("--interview-budget", type=float, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "resume":
            return _resume(args)
        if args.command == "report":
            return _report(args)
        if args.command == "interview":
            return _interview(args)
        if args.command == "serve":
            return _serve(args)
        if args.command == "check-models":
            return _check_models(args)
        if args.command == "llm-demo":
            return _llm_demo(args)
        if args.command == "usage":
            return _usage(args)
    except (ConfigError, FileNotFoundError, FileExistsError, KeyError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 1


def _needs_key(stub: bool) -> bool:
    if stub or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return False
    print("ANTHROPIC_API_KEY is not set (use --stub for a zero-cost run).", file=sys.stderr)
    return True


def _is_stub_run(run_dir: str) -> bool:
    import json

    return bool(json.loads((Path(run_dir) / "config.resolved.json").read_text(encoding="utf-8")).get("stub"))


def _exit_code(reason: str) -> int:
    # Hitting the budget is a clean, expected stop.
    return 0 if reason in ("completed", "budget") else (130 if reason == "interrupted" else 1)


def _run(args: argparse.Namespace) -> int:
    from .sim.runner import start_run

    if _needs_key(args.stub):
        return 2
    result = start_run(args.scenario, budget=args.budget, stub=args.stub, days=args.days, hours=args.hours,
                       runs_dir=args.runs_dir, run_id=args.run_id, make_report=not args.no_report,
                       config_dir=args.config_dir)
    return _exit_code(result.reason)


def _resume(args: argparse.Namespace) -> int:
    from .sim.runner import resume_run

    if _needs_key(_is_stub_run(args.run_dir)):
        return 2
    result = resume_run(args.run_dir, budget=args.budget, days=args.days, hours=args.hours,
                        make_report=not args.no_report)
    return _exit_code(result.reason)


def _report(args: argparse.Namespace) -> int:
    from .sim.runner import regenerate_report

    if not args.no_llm and _needs_key(_is_stub_run(args.run_dir)):
        return 2
    regenerate_report(args.run_dir, use_llm=not args.no_llm, budget=args.budget)
    return 0


def _interview(args: argparse.Namespace) -> int:
    from .sim.runner import interview_run

    stub = True if args.stub else None
    if _needs_key(args.stub or _is_stub_run(args.run_dir)):
        return 2
    r = interview_run(args.run_dir, args.agent, args.question, tick=args.tick, stub=stub, budget=args.budget)
    print(f"[{r['game_time']}, {r['checkpoint']}] {r['agent']}: {r['answer']}")
    for c in r["cited"]:
        print(f"  - {c['node_id']}: {c['text']}")
    print(f"cost: ${r['cost_usd']:.4f}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .web.server import create_app
    except ImportError:
        print('the viewer needs the web extra: pip install -e ".[web]"', file=sys.stderr)
        return 2
    cfg = load_config(args.config_dir)
    budget = args.interview_budget if args.interview_budget is not None else float(
        cfg.raw.get("server", {}).get("interview_budget_usd", 0.5))
    app = create_app(args.runs_dir, stub_interviews=args.stub_interviews, interview_budget=budget)
    print(f"smallville viewer: http://{args.host}:{args.port}  (runs: {Path(args.runs_dir).resolve()})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


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
