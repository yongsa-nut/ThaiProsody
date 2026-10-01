"""Run candidate models through the prosody prompts.

Writes resume-safe raw generations to eval/runs/<model_key>__run<k>.jsonl
(one Call per line). Re-running skips items already completed, so an interrupted
run never re-requests finished items. Grading is a separate step (grade_runs.py).
This script calls model APIs; keys are read from the environment or a .env file.

  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.run_models --models typhoon25 --sets compose_public --limit 4 --dry-run
  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.run_models --models typhoon25,gpt54-mini --sets compose_public --limit 16
  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.run_models            # all enabled, compose_public
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../prosody/eval
REPO = HERE.parents[2]                             # eval -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv                                          # noqa: E402

for _parent in HERE.parents:
    if (_parent / ".env").exists():
        load_dotenv(_parent / ".env", override=True)   # .env is authoritative over stale OS env
        break

from pilot_rerun.clients import load_completed                          # noqa: E402
from thaiwrite.prosody.eval.promptlib import SYSTEM, load_prompts, render_prompt  # noqa: E402
from thaiwrite.prosody.eval.providers import make_generator             # noqa: E402
from thaiwrite.prosody.eval.registry import resolve                     # noqa: E402

RUNS_DIR = HERE / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run models through the prosody prompts.")
    ap.add_argument("--models", default="", help="comma-separated keys; default = all enabled")
    ap.add_argument("--sets", default="compose_public",
                    help="comma-separated: compose_public,compose_private,cloze,edit")
    ap.add_argument("--runs", type=int, default=1, help="generations per model (default 1)")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = per-model default")
    ap.add_argument("--limit", type=int, default=0, help="first N loaded items (0 = all)")
    ap.add_argument("--per-meter", type=int, default=0,
                    help="balanced subset: first N items of EACH meter_id (overrides --limit)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel in-flight requests per model (default 1 = serial)")
    ap.add_argument("--no-think", action="store_true",
                    help="ablation: disable reasoning (OpenRouter reasoning.enabled=false); "
                         "writes to <key>.nothink__run*.jsonl so it doesn't clobber native runs")
    ap.add_argument("--rules", action="store_true",
                    help="ablation: append the meter's formal rule sheet (rendered from "
                         "meter_specs via rulesheets.py) to the system prompt; writes to "
                         "<key>.rules__run*.jsonl so it doesn't clobber zero-shot runs")
    ap.add_argument("--dry-run", action="store_true", help="print plan + counts, no API calls")
    args = ap.parse_args()

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    specs = resolve(keys or None)
    items = load_prompts(sets)
    if args.per_meter:
        seen: dict[str, int] = {}
        picked = []
        for it in items:
            mid = it.get("meter_id", "")
            if seen.get(mid, 0) < args.per_meter:
                picked.append(it)
                seen[mid] = seen.get(mid, 0) + 1
        items = picked
    elif args.limit:
        items = items[:args.limit]

    print(f"Models : {', '.join(s.key for s in specs)}")
    print(f"Sets   : {sets}  |  items: {len(items)}  |  runs: {args.runs}  |  "
          f"concurrency: {args.concurrency}" + ("   [DRY RUN]" if args.dry_run else ""))
    print("note   : temperature=0 for OpenAI-compatible models (greedy decoding); "
          "reasoning models may still vary across runs.\n")

    suffix = (".rules" if args.rules else "") + (".nothink" if args.no_think else "")
    if args.rules:
        from thaiwrite.prosody.eval.rulesheets import RULE_SHEETS
    for spec in specs:
        if args.no_think:
            spec = dataclasses.replace(spec, or_reasoning={"enabled": False})
        mt = args.max_tokens or spec.max_tokens
        try:
            for k in range(1, args.runs + 1):
                log = RUNS_DIR / f"{spec.key}{suffix}__run{k}.jsonl"
                done = load_completed(log)
                todo = [it for it in items if it["id"] not in done]
                print(f"{spec.display}  run{k}: {len(todo)} to run ({len(done)} cached)", flush=True)
                if args.dry_run or not todo:
                    continue
                gen = make_generator(spec, log)

                def _run(it):
                    system = (SYSTEM + "\n\n" + RULE_SHEETS[it["meter_id"]]
                              if args.rules else SYSTEM)
                    call = gen.generate("prosody-eval", it["id"], render_prompt(it), system, mt)
                    return it, call

                if args.concurrency > 1:
                    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                        for n, (it, call) in enumerate(
                                (f.result() for f in as_completed(ex.submit(_run, it) for it in todo)), 1):
                            flag = f"ERR: {str(call.error)[:50]}" if call.error else "ok"
                            print(f"  [{n}/{len(todo)}] {it['id']:<34} {flag}", flush=True)
                else:
                    for n, it in enumerate(todo, 1):
                        _, call = _run(it)
                        flag = f"ERR: {str(call.error)[:50]}" if call.error else "ok"
                        print(f"  [{n}/{len(todo)}] {it['id']:<34} {flag}", flush=True)
        except RuntimeError as e:
            print(f"  [skip] {spec.display}: {e}")
            continue


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
