"""Build the calibration set (v2), sampled from the full multi-model compose run.

Unlike the 2-model pilot in build_calibration_set.py, this samples from the graded
compose battery (eval/graded/*__run1.jsonl), i.e. the actual output distribution
the judge is applied to, stratified on the two axes calibration must span:

  * L2 coherence: sample across model TIERS (high / mid / low expected
    coherence). The roster is frontier-heavy, so a topic-only or uniform sample
    piles up at coherence 4-5 and leaves the judge's word-salad end untested.
    Tier is used as the proxy because per-item judge scores are what the
    calibration measures. Mean meter_score differs by tier (high 0.70 / mid 0.56
    / low 0.24), so the tiers are distinct.
  * L1 meter: every meter appears with pass AND fail; fails are abundant
    (run-wide fail rates: count 59% / rhyme 71% / tone 89% / kru-lahu 90%), and
    the scarce compliant tone/kru-lahu cases are guaranteed by the clean anchors.

Within each (meter x tier) cell we round-robin across distinct models and balance
topics, so no single model or topic dominates. Poems are then shuffled so the
annotator sees no tier/model ordering cue (the tool never shows _provenance).

The output schema is identical to build_calibration_set.py, so annotate.html and
bundle_for_experts.py consume it unchanged.

  python build_calibration_v2.py                  # ~30/meter
  python build_calibration_v2.py --per-meter 35
  python build_calibration_v2.py --modes compose cloze edit

Then:
  python bundle_for_experts.py calibration_items_v2.json \
      --experts expertA expertB
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../prosody/calibration
GRADED = HERE.parent / "eval" / "graded"          # the real run
OUT = HERE / "calibration_items_v2.json"

# reuse the anchor / rhyme / dims helpers (no side effects at import)
from build_calibration_set import (                                    # noqa: E402
    dims_for, clean_anchors, broken_anchors, rhyme_pairs,
)

# ---------- expected-coherence tiers (model roster) ----------
# Grouped by how fluent/connected their Thai output is, NOT by meter score.
# Frontier + capable generalists write connected Thai; the 8B Thai models and the
# reasoning models that exhaust their token budget produce truncated / repetitive /
# word-salad text.
TIER_HIGH = {
    "gpt55-high", "gpt55-med", "gpt55-low", "gemini-3.5-flash",
    "opus-4.7", "sonnet-4.6", "deepseek-v4-pro", "gpt54-mini",
}
TIER_MID = {"gemma-4-31b", "deepseek-v4-flash", "typhoon25"}
# everything else → low: openthaigpt, typhoon-s, thalle, pathumma,
#                        qwen3.6-35b-a3b, glm-5.1, minimax-m2.7, kimi-k2.6
TIERS = ("high", "mid", "low")


def tier_of(model_key: str) -> str:
    if model_key in TIER_HIGH:
        return "high"
    if model_key in TIER_MID:
        return "mid"
    return "low"


# ---------- load the real graded run ----------

def load_graded(modes: set[str]) -> list[dict]:
    rows = []
    for f in sorted(GRADED.glob("*__run1.jsonl")):
        if ".nothink" in f.name:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task_mode") not in modes:
                continue
            if r.get("eval_status") != "graded":
                continue                              # skip non-delivery / infra errors
            poem = (r.get("poem") or "").strip()
            if not poem:
                continue
            rows.append(r)
    return rows


# ---------- stratified pick: round-robin over models, topic-balanced ----------

def pick_cell(cands: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Pick up to `target` items, one model at a time (round-robin) and within a
    model preferring the least-seen topic: spreads across models AND topics."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_model[c["model_key"]].append(c)
    models = list(by_model)
    rng.shuffle(models)
    for m in models:
        rng.shuffle(by_model[m])

    picked: list[dict] = []
    seen_topic: Counter = Counter()
    while len(picked) < target:
        progressed = False
        for m in models:
            if len(picked) >= target:
                break
            lst = by_model[m]
            if not lst:
                continue
            lst.sort(key=lambda c: seen_topic[c.get("topic", "")])   # least-seen topic first
            c = lst.pop(0)
            picked.append(c)
            seen_topic[c.get("topic", "")] += 1
            progressed = True
        if not progressed:
            break
    return picked


def to_item(r: dict) -> dict:
    return {
        "id": f"cal2-{r['model_key']}-{r['item_id']}",
        "kind": "poem",
        "task_mode": r.get("task_mode", "compose"),
        "meter": r.get("meter"),
        "meter_id": r.get("meter_id"),
        "topic": r.get("topic", ""),
        "applicable_dims": dims_for(r["meter_id"]),
        "poem_text": (r.get("poem") or "").strip(),
        "_provenance": {
            "kind": "run_sample",
            "model": r.get("model_key"),
            "display": r.get("display"),
            "tier": tier_of(r["model_key"]),
            "src_item_id": r.get("item_id"),
            "meter_score": r.get("meter_score"),
        },
    }


def build(per_meter: int, modes: set[str], seed: int) -> None:
    rng = random.Random(seed)
    rows = load_graded(modes)

    # group by meter → tier
    by_meter_tier: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_meter_tier[r["meter_id"]][tier_of(r["model_key"])].append(r)

    per_tier = [per_meter // len(TIERS)] * len(TIERS)
    for i in range(per_meter - sum(per_tier)):       # spread the remainder
        per_tier[i] += 1
    tier_target = dict(zip(TIERS, per_tier))

    sampled: list[dict] = []
    grid: dict[str, dict[str, int]] = {}             # meter → tier → n picked
    for meter_id, tiers in sorted(by_meter_tier.items()):
        grid[meter_id] = {}
        for t in TIERS:
            got = pick_cell(tiers.get(t, []), tier_target[t], rng)
            grid[meter_id][t] = len(got)
            sampled += [to_item(r) for r in got]

    # dedup identical poems (rare, but trivial outputs can collide)
    seen_text: set[str] = set()
    uniq = []
    for it in sampled:
        if it["poem_text"] in seen_text:
            continue
        seen_text.add(it["poem_text"])
        uniq.append(it)

    poems = uniq + clean_anchors() + broken_anchors()
    rng.shuffle(poems)                               # no tier/model ordering cue
    pairs = rhyme_pairs()
    items = poems + pairs
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report ----
    name_th = {r["meter_id"]: r["meter"] for r in rows}
    print(f"Sampled from {len(rows)} graded items "
          f"({', '.join(sorted(modes))}); target {per_meter}/meter, split {tier_target}.\n")
    print(f"  {'meter':22s} {'high':>5} {'mid':>5} {'low':>5}")
    for mid, tt in grid.items():
        print(f"  {name_th.get(mid, mid):22s} {tt['high']:5d} {tt['mid']:5d} {tt['low']:5d}")

    # coherence-spread sanity: mean meter_score of the SAMPLED poems by tier
    ms: dict[str, list[float]] = defaultdict(list)
    for it in uniq:
        ms[it["_provenance"]["tier"]].append(it["_provenance"]["meter_score"] or 0.0)
    print("\n  sampled mean meter_score by tier (coherence proxy — should spread):")
    for t in TIERS:
        v = ms[t]
        print(f"    {t:5s}: {sum(v)/len(v):.3f}  (n={len(v)})" if v else f"    {t:5s}: —")

    by_kind: Counter = Counter()
    for p in poems:
        by_kind[p["_provenance"]["kind"]] += 1
    print(f"\n  poems:       {len(poems):3d}  ({dict(by_kind)})")
    print(f"  rhyme pairs: {len(pairs):3d}")
    print(f"  TOTAL items: {len(items):3d}  → {OUT.name}")
    print(f"\n  each expert labels all {len(poems)} poems + {len(pairs)} pairs (full overlap, 2-rater alpha).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-meter", type=int, default=30,
                    help="model poems per meter, split across high/mid/low tiers (default 30)")
    ap.add_argument("--modes", nargs="+", default=["compose"],
                    choices=["compose", "cloze", "edit"],
                    help="task modes to sample (default compose only)")
    ap.add_argument("--seed", type=int, default=20260525)
    args = ap.parse_args()
    build(args.per_meter, set(args.modes), args.seed)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
