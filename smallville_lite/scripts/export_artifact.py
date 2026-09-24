"""Export one run as a static, read-only copy of the web viewer (for publishing as a shareable page).

The output is the viewer's own index.html and JS modules plus the JSON the page would have fetched from
the server, captured through the real FastAPI app. A small fetch shim in the page answers ``/api/...``
from those files. Live mode and new interviews need the server, so they are off in the export.

    .venv/Scripts/python scripts/export_artifact.py runs/party-demo out_dir
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from smallville_lite.web.server import STATIC, create_app

SHIM = """<script>
// Static export: answer the viewer's API calls from bundled JSON files.
(() => {
  const realFetch = window.fetch.bind(window);
  const json = (body, status) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  window.fetch = (url, opts) => {
    if (typeof url !== "string" || !url.startsWith("/api/")) return realFetch(url, opts);
    if (opts && opts.method === "POST") {
      return Promise.resolve(json({ detail: "New interviews need the local viewer (smallville serve). This shared page is a read-only replay." }, 400));
    }
    const path = url.split("?")[0];
    if (path === "/api/runs") return realFetch("data/runs.json");
    if (path === "/api/presets") return realFetch("data/presets.json");
    const m = path.match(/^\\/api\\/runs\\/([A-Za-z0-9_.-]+)\\/(meta|events|layout|report|interviews)$/);
    if (m) return realFetch(`data/${m[1]}/${m[2]}.json`);
    return Promise.resolve(json({ detail: "not available in the shared replay" }, 404));
  };
})();
</script>
"""

NOTE = """  <div class="bar small muted" style="padding-top:0">
    Shared replay of a 2-day test run of smallville_lite, a small reimplementation of
    Generative Agents (Park et al., 2023). This run used the offline stand-in model, so the agents' words are
    placeholder text and the dollar figures are estimates of what real calls would cost. The schedules, memory,
    reflection, and movement machinery is real. Asking new interview questions is off here.
  </div>
"""


def transform(html: str, run_id: str) -> str:
    # The publish skeleton supplies doctype/html/head/body and the charset + viewport metas.
    html = re.sub(r"<!doctype html>\s*<html[^>]*>\s*<head>\s*", "", html, flags=re.I)
    html = re.sub(r'<meta charset="utf-8">\s*<meta name="viewport"[^>]*>\s*', "", html)
    html = re.sub(r"</head>\s*<body>\s*", "", html)
    html = re.sub(r"</body>\s*</html>\s*$", "", html)
    html = html.replace("<title>Smallville Log</title>", "<title>Smallville Replay</title>")

    # Theme: honour the viewer's explicit light/dark choice, not only the OS setting.
    m = re.search(r"@media \(prefers-color-scheme: dark\) \{\s*:root \{(.*?)\}\s*\}", html, flags=re.S)
    if not m:
        raise SystemExit("dark-theme block not found in index.html")
    tokens = m.group(1)
    html = html.replace(m.group(0), "@media (prefers-color-scheme: dark) {\n    :root:not([data-theme=\"light\"]) {"
                        + tokens + "}\n  }\n  :root[data-theme=\"dark\"] {" + tokens + "}", 1)
    html = html.replace("header { position: sticky; top: 0;", "header { position: sticky; top: env(safe-area-inset-top, 0px);")

    html = html.replace('from "/static/', 'from "./static/')
    html = html.replace('<script type="module">', SHIM + '<script type="module">', 1)
    html = html.replace('  <div class="scrub">', NOTE + '  <div class="scrub">', 1)
    html = html.replace('<div class="brand">smallville_lite<small>event log</small></div>',
                        '<div class="brand">smallville_lite<small>replay</small></div>')
    html = html.replace('history.replaceState(null, "", `?run=${encodeURIComponent(id)}`);',
                        'try { history.replaceState(null, "", `?run=${encodeURIComponent(id)}`); } catch {}')
    if "./static/state.js" not in html or "data/runs.json" not in html:
        raise SystemExit("index.html no longer matches what this script expects")
    return html


def main(run_dir: str, out_dir: str) -> None:
    run = Path(run_dir).resolve()
    out = Path(out_dir)
    run_id = run.name
    client = TestClient(create_app(run.parent))

    def get(url: str):
        r = client.get(url)
        r.raise_for_status()
        return r.json()

    if out.exists():
        shutil.rmtree(out)
    (out / "data" / run_id).mkdir(parents=True)
    shutil.copytree(STATIC, out / "static", ignore=shutil.ignore_patterns("index.html"))

    runs = [r for r in get("/api/runs") if r["run_id"] == run_id]
    files = {
        "data/runs.json": runs,
        "data/presets.json": get("/api/presets"),
        # scenario_dir is a local absolute path; the page doesn't use it.
        f"data/{run_id}/meta.json": {k: v for k, v in get(f"/api/runs/{run_id}/meta").items() if k != "scenario_dir"},
        f"data/{run_id}/events.json": get(f"/api/runs/{run_id}/events?after=0"),
        f"data/{run_id}/layout.json": get(f"/api/runs/{run_id}/layout"),
        f"data/{run_id}/interviews.json": get(f"/api/runs/{run_id}/interviews"),
    }
    report = client.get(f"/api/runs/{run_id}/report")
    if report.status_code == 200:
        files[f"data/{run_id}/report.json"] = report.json()
    for rel, body in files.items():
        (out / rel).write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(transform(html, run_id), encoding="utf-8")

    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:7.2f} MB  {p.relative_to(out).as_posix()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
