"""Per-model generation stats over eval/runs/ (token usage, reasoning length,
finish reasons, empty/error rates, latency). Diagnostic, separate from grading.
Writes gen_stats.{md,json} to --out (default eval/graded_new/).

  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.run_stats
  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.run_stats --models gemini-3.5-flash,kimi-k2.6
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                            # eval -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))

from thaiwrite.prosody.eval.promptlib import clean_poem    # noqa: E402
from thaiwrite.prosody.eval.registry import REGISTRY       # noqa: E402

RUNS_DIR = HERE / "runs"


def _stat(vals: list[int | float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "median": None, "max": None}
    return {"mean": round(statistics.mean(vals), 1),
            "median": round(statistics.median(vals), 1), "max": max(vals)}


def model_stats(recs: list[dict]) -> dict:
    n = len(recs)
    err = [r for r in recs if r.get("error")]
    nonerr = [r for r in recs if not r.get("error")]
    empty = [r for r in nonerr if not clean_poem(r.get("response", ""))]
    ok = [r for r in nonerr if clean_poem(r.get("response", ""))]
    return {
        "n": n, "ok": len(ok), "empty": len(empty), "error": len(err),
        "out_tokens": _stat([r.get("output_tokens") for r in nonerr]),
        "reasoning_chars": _stat([len(r.get("reasoning") or "") for r in nonerr]),
        "latency_s": _stat([r.get("latency_s") for r in recs]),
        "finish_reason": dict(Counter(str(r.get("finish_reason")) for r in recs)),
        "error_kinds": dict(Counter(str(r.get("error", "")).split(":")[0] for r in err)) or None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-model generation stats.")
    ap.add_argument("--models", default="", help="comma-separated keys; default = all run files")
    ap.add_argument("--out", default=str(HERE / "graded_new"),
                    help="output directory for gen_stats.{md,json} (default: eval/graded_new/)")
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    files = sorted(RUNS_DIR.glob("*__run*.jsonl"))
    by_key: dict[str, list[dict]] = {}
    for f in files:
        key = f.name.split("__run")[0]
        if keys and key not in keys:
            continue
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_key.setdefault(key, []).extend(recs)

    stats = {k: model_stats(v) for k, v in sorted(by_key.items())}

    lines = ["# Generation stats (per model)", "",
             "| Model | n | ok | empty | err | out_tok mean/max | reason_chars mean/max | lat mean(s) | finish_reasons |",
             "|---|--:|--:|--:|--:|--|--|--:|--|"]
    for k, s in stats.items():
        disp = REGISTRY[k].display if k in REGISTRY else k
        ot, rc = s["out_tokens"], s["reasoning_chars"]
        fr = ", ".join(f"{kk}:{vv}" for kk, vv in s["finish_reason"].items())
        lines.append(
            f"| {disp} | {s['n']} | {s['ok']} | {s['empty']} | {s['error']} "
            f"| {ot['mean']}/{ot['max']} | {rc['mean']}/{rc['max']} | {s['latency_s']['mean']} | {fr} |")
    md = "\n".join(lines) + "\n"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gen_stats.md").write_text(md, encoding="utf-8")
    (out_dir / "gen_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"Wrote {out_dir / 'gen_stats'}.{{md,json}}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
