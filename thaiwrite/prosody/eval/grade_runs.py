"""Grade and aggregate the composition model runs.

For each model run under eval/runs/, grade every generation with:
  L1: grader_meter.grade_text (deterministic rule layer, eval_rhyme=True for every mode)
  L2: grader_coherence.judge_coherence (optional LLM topicality/coherence judge,
      gpt-5.5 by default; needs OPENAI_API_KEY). Pass --no-coherence for a
      rules-only regrade that makes no API calls.

L2 scores written here are tagged "provisional_uncalibrated" and are reported
separately from the meter score (no blended scalar). The content-judge scores
reported in the paper come from judge_api.py, not from this L2 pass.

Outputs go to --out (default eval/graded_new/), so the archived eval/graded/ files
that the paper tables are built from are never overwritten by default:
  per-item records   -> <out>/<key>__run<k>.jsonl
  pooled aggregates  -> <out>/summary.{json,md} (item-level bootstrap CI on meter)

  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.grade_runs --models typhoon25 --runs 1 --no-coherence
  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.grade_runs --models typhoon25,gpt54-mini --runs 1 --bootstrap 500
  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.grade_runs --no-coherence --out regraded/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../prosody/eval
REPO = HERE.parents[2]                            # eval -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv                                          # noqa: E402

for _parent in HERE.parents:
    if (_parent / ".env").exists():
        load_dotenv(_parent / ".env", override=True)   # .env authoritative over stale OS env
        break

from thaiwrite.prosody.eval.promptlib import clean_poem, load_prompts, to_vrk_lines  # noqa: E402
from thaiwrite.prosody.eval.registry import REGISTRY                    # noqa: E402
from thaiwrite.prosody.grader_meter import grade_text                   # noqa: E402

RUNS_DIR = HERE / "runs"
GRADED_DIR = HERE / "graded_new"                 # default output; eval/graded/ holds the archived results
ALL_SETS = ["compose_public", "compose_private", "cloze", "edit"]
DIMS = ["count", "tone", "rhyme", "kru_lahu"]


# ---------- helpers ----------

def _dim_compliant(layers: dict) -> dict[str, bool]:
    """A dimension is compliant when it is applicable (total>0) and fully passed."""
    return {dim: (L["passed"] == L["total"])
            for dim, L in layers.items() if L["total"] > 0}


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _bootstrap_ci(values: list, n_boot: int, seed: int) -> list:
    """95% percentile CI of the mean via item resampling-with-replacement."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2 or n_boot <= 0:
        return [None, None]
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    def pct(p: float) -> float:
        return round(means[min(n_boot - 1, max(0, round(p / 100 * (n_boot - 1))))], 4)
    return [pct(2.5), pct(97.5)]


def _load_run(path: Path) -> list[dict]:
    """Records for a run file, deduped by item_id (a successful retry wins)."""
    best: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sid = str(rec["item_id"])
        good = (not rec.get("error")) and bool(rec.get("response"))
        prev = best.get(sid)
        if prev is None or good or not ((not prev.get("error")) and bool(prev.get("response"))):
            best[sid] = rec
    return list(best.values())


# ---------- grading ----------

def grade_record(rec: dict, item: dict | None, judge, seg: str = "bestof2") -> dict:
    item = item or {}
    meter_id = item.get("meter_id")
    reasoning = rec.get("reasoning")
    out: dict = {
        "model_key": None, "model_id": rec.get("model"), "display": None, "run": None,
        "item_id": str(rec["item_id"]), "task_mode": item.get("task_mode"),
        "meter_id": meter_id, "meter": item.get("meter"), "topic": item.get("topic"),
        "raw_response": rec.get("response"),       # full model output, for deep dive
        "poem": None, "layout": None, "n_vrk": None, "gen_error": None,
        # eval_status: graded | model_fail (empty/token-runout, scored 0) |
        # infra_error (402/5xx/timeout/bad-body/grader crash; excluded, not a model failure)
        "eval_status": None,
        "meter_score": None, "meter_raw": None, "meter_norm": None,
        "layer_rates": None, "dim_compliant": None, "layers": None,
        "coherence": None,
        "finish_reason": rec.get("finish_reason"),
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "latency_s": rec.get("latency_s"),
        "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
    }

    if rec.get("error"):
        out["gen_error"] = str(rec["error"])[:200]
        out["eval_status"] = "infra_error"      # API failure (402/5xx/timeout/bad-body): excluded
        return out
    if meter_id is None:
        out["gen_error"] = "prompt_not_found"
        out["eval_status"] = "infra_error"
        return out

    poem = clean_poem(rec.get("response", ""))
    out["poem"] = poem
    if not poem:
        out["gen_error"] = "empty_output"
        out["eval_status"] = "model_fail"       # model delivered no poem (runout/empty): scored 0
        return out

    # grade BOTH layouts (cheap, deterministic) and keep both scores for analysis.
    # pythainlp can raise on pathological tokens; isolate so one bad poem doesn't
    # kill the whole run.
    try:
        g_raw = grade_text(poem, meter_id, eval_rhyme=True)
        g_norm = grade_text(to_vrk_lines(poem), meter_id, eval_rhyme=True)
    except Exception as exc:  # noqa: BLE001
        out["gen_error"] = f"grade_error: {type(exc).__name__}: {exc}"[:200]
        out["eval_status"] = "infra_error"      # grader crash, not a model failure: excluded
        return out
    out["meter_raw"] = g_raw["score"]
    out["meter_norm"] = g_norm["score"]
    if seg == "raw":
        layout = "raw"
    elif seg == "normalized":
        layout = "normalized"
    else:                                          # bestof2
        layout = "raw" if g_raw["score"] >= g_norm["score"] else "normalized"
    m = g_raw if layout == "raw" else g_norm
    out["layout"] = layout
    out["n_vrk"] = len((poem if layout == "raw" else to_vrk_lines(poem)).splitlines())
    out["meter_score"] = m["score"]
    out["eval_status"] = "graded"
    out["layer_rates"] = m["layer_rates"]
    out["dim_compliant"] = _dim_compliant(m["layers"])
    out["layers"] = m["layers"]                    # full per-rule details for error analysis

    if judge is not None:
        from thaiwrite.prosody.grader_coherence import judge_coherence
        cr = judge_coherence(poem, {"id": out["item_id"], "topic": out["topic"]}, judge=judge)
        out["coherence"] = {
            "topicality": cr.topicality, "coherence": cr.coherence,
            "composite": cr.composite, "rationale": cr.rationale,
            "model": cr.model, "error": cr.error, "status": "provisional_uncalibrated",
        }
    return out


# ---------- aggregation ----------

def _status(r: dict) -> str:
    """eval_status with a fallback for records graded before the tag existed."""
    st = r.get("eval_status")
    if st:
        return st
    if r.get("meter_score") is not None:
        return "graded"
    return "model_fail" if r.get("gen_error") == "empty_output" else "infra_error"


def aggregate(records: list[dict], n_boot: int, seed: int) -> dict:
    graded = [r for r in records if _status(r) == "graded"]
    model_fail = [r for r in records if _status(r) == "model_fail"]
    infra = [r for r in records if _status(r) == "infra_error"]
    scores = [r["meter_score"] for r in graded]
    # fail-scored view: the model's own non-delivery (empty / token-runout) scores 0,
    # a real task failure. Infrastructure errors (402 credit / 5xx / timeout / bad body /
    # grader crash) are not model failures, so they are excluded rather than scored 0.
    scores_with_fail = scores + [0.0] * len(model_fail)
    n_eval = len(graded) + len(model_fail)

    dim_rates: dict[str, float | None] = {}
    for dim in DIMS:
        applicable = [r["dim_compliant"][dim] for r in graded
                      if r["dim_compliant"] and dim in r["dim_compliant"]]
        dim_rates[dim] = round(sum(applicable) / len(applicable), 4) if applicable else None

    by_meter: dict[str, list] = defaultdict(list)
    by_mode: dict[str, list] = defaultdict(list)
    for r in graded:
        by_meter[r["meter_id"]].append(r["meter_score"])
        by_mode[r["task_mode"]].append(r["meter_score"])

    coh = [r["coherence"] for r in graded if r.get("coherence")]
    coh_ok = [c for c in coh if c.get("error") is None]
    coherence = None
    if coh:
        coherence = {
            "n_judged": len(coh), "n_judge_err": len(coh) - len(coh_ok),
            "topicality": _mean([c["topicality"] for c in coh_ok]),
            "coherence": _mean([c["coherence"] for c in coh_ok]),
            "composite": _mean([c["composite"] for c in coh_ok]),
            "status": "provisional_uncalibrated",
        }

    return {
        "n": len(records),                                        # total attempted
        "n_eval": n_eval,                                         # graded + model_fail (denom for fail-score)
        "n_graded": len(graded),
        "n_model_fail": len(model_fail),                          # empty / token-runout, scored 0
        "n_infra_excluded": len(infra),                           # 402/5xx/timeout/bad-body, excluded
        "n_gen_errors": sum(1 for r in records if r["gen_error"]),
        "delivery_rate": round(len(graded) / n_eval, 4) if n_eval else None,
        "meter_mean": _mean(scores),                              # delivered-only
        "meter_mean_fail": _mean(scores_with_fail),               # model non-delivery = 0 (headline)
        "meter_ci95": _bootstrap_ci(scores, n_boot, seed),
        "meter_ci95_fail": _bootstrap_ci(scores_with_fail, n_boot, seed),
        "dim_compliance": dim_rates,
        "by_meter": {k: _mean(v) for k, v in sorted(by_meter.items())},
        "by_mode": {k: _mean(v) for k, v in sorted(by_mode.items())},
        "coherence": coherence,
    }


# ---------- report ----------

def _fmt(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def write_markdown(summary: dict, path: Path) -> None:
    meta = summary["_meta"]
    lines = [
        "# Composition results",
        "",
        f"_Generated {meta['date']} · sets: {', '.join(meta['sets'])} · runs: {meta['runs']} · "
        f"segmentation: {meta.get('segmentation', 'bestof2')} · "
        f"coherence: {'on (provisional, uncalibrated judge)' if meta['coherence'] else 'off'}_",
        "",
        "Meter score is the deterministic rule-based signal. "
        "Coherence (if on) comes from an uncalibrated LLM judge and is provisional.",
        "",
        "| Model | n | deliv% | Meter-fail [95% CI] | Meter-delivered | count | tone | rhyme | kru/lahu | Coh-C |",
        "|---|--:|--:|---|--:|--:|--:|--:|--:|--:|",
    ]
    for key in meta["model_order"]:
        a = summary["models"][key]
        d = a["dim_compliance"]
        ci = a.get("meter_ci95_fail", a["meter_ci95"])
        ci_s = "" if ci[0] is None else f" [{_fmt(ci[0])}–{_fmt(ci[1])}]"
        dr = a.get("delivery_rate")
        c = a["coherence"]
        cc = _fmt(c["coherence"], 2) if c else "—"
        lines.append(
            f"| {REGISTRY[key].display if key in REGISTRY else key} "
            f"| {a['n']} | {_fmt(dr, 2) if dr is not None else '—'} "
            f"| {_fmt(a.get('meter_mean_fail'))}{ci_s} "
            f"| {_fmt(a['meter_mean'])} "
            f"| {_fmt(d['count'])} | {_fmt(d['tone'])} | {_fmt(d['rhyme'])} | {_fmt(d['kru_lahu'])} "
            f"| {cc} |")

    lines += ["", "## Meter score by meter (per model)", "",
              "| Model | โคลงสี่สุภาพ | กลอนแปด | กาพย์ยานี ๑๑ | ฉันท์ ๑๑ |",
              "|---|--:|--:|--:|--:|"]
    order = ["khlong_si_suphap", "klon_paet", "kap_yani_11", "intharawichian_chan_11"]
    for key in meta["model_order"]:
        bm = summary["models"][key]["by_meter"]
        cells = " | ".join(_fmt(bm.get(m)) for m in order)
        lines.append(f"| {REGISTRY[key].display if key in REGISTRY else key} | {cells} |")

    lines += ["", "_Dimension columns = compliance rate (fraction of poems fully "
              "satisfying that layer). tone applies to โคลง only; kru/lahu to ฉันท์ only._", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Grade composition model runs (meter + optional coherence).")
    ap.add_argument("--models", default="", help="comma-separated keys; default = all with run files")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-coherence", action="store_true", help="skip the LLM coherence judge")
    ap.add_argument("--coherence-model", default="gpt-5.5")
    ap.add_argument("--coherence-reasoning", default="medium")
    ap.add_argument("--segmentation", choices=["bestof2", "raw", "normalized"], default="bestof2",
                    help="วรรค segmentation for the meter grader (default bestof2 = max of "
                         "literal lines and whitespace-split, format-agnostic)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--out", "--graded-dir", dest="out", default=str(GRADED_DIR),
                    help="output directory for graded records + summary "
                         "(default: eval/graded_new/; eval/graded/ holds the archived results)")
    args = ap.parse_args()
    seg = args.segmentation

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = {it["id"]: it for it in load_prompts(ALL_SETS)}

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    if not keys:
        keys = sorted({p.name.split("__run")[0] for p in RUNS_DIR.glob("*__run*.jsonl")})
    if not keys:
        print("No run files in eval/runs/. Run run_models.py first."); return

    judge = None
    if not args.no_coherence:
        from thaiwrite.prosody.grader_coherence import OpenAIJudge
        judge = OpenAIJudge(model=args.coherence_model, reasoning=args.coherence_reasoning)

    summary = {"_meta": {"date": str(date.today()), "runs": args.runs,
                         "coherence": judge is not None, "sets": ALL_SETS,
                         "segmentation": seg, "model_order": []},
               "models": {}}

    for key in keys:
        all_recs: list[dict] = []
        for k in range(1, args.runs + 1):
            run_path = RUNS_DIR / f"{key}__run{k}.jsonl"
            if not run_path.exists():
                continue
            display = REGISTRY[key].display if key in REGISTRY else key
            graded_path = out_dir / f"{key}__run{k}.jsonl"
            recs = []
            print(f"{display}  run{k}: grading {run_path.name} ...")
            for rec in _load_run(run_path):
                g = grade_record(rec, idx.get(str(rec["item_id"])), judge, seg=seg)
                g["model_key"], g["display"], g["run"] = key, display, k
                recs.append(g)
            with graded_path.open("w", encoding="utf-8") as f:
                for g in recs:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            all_recs += recs
        if not all_recs:
            print(f"  [skip] {key}: no run files for runs 1..{args.runs}")
            continue
        agg = aggregate(all_recs, args.bootstrap, args.seed)
        summary["models"][key] = agg
        summary["_meta"]["model_order"].append(key)
        c = agg["coherence"]
        print(f"  → meter {_fmt(agg['meter_mean'])} "
              f"(CI {_fmt(agg['meter_ci95'][0])}–{_fmt(agg['meter_ci95'][1])}), "
              f"n={agg['n_graded']}/{agg['n']}, gen_err={agg['n_gen_errors']}"
              + (f", coh-composite {_fmt(c['composite'],2)} (provisional)" if c else ""))

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md")
    print(f"\nWrote {out_dir / 'summary.json'} and summary.md")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
