"""Calibration scoring: turns the expert label files into agreement numbers.

Computes, from calibration_items.json and the per-expert labels_<id>.json files:

  Layer 2 (coherence judge)
    - human-human Krippendorff alpha (ordinal), per axis: the reliability ceiling
    - [if --judge] judge-vs-gold alpha (ordinal), per axis: the 0.65 threshold
  Layer 1 (meter grader)
    - human-human alpha (nominal) per meter dimension: labeling reliability
    - [if --gold] grader-vs-gold per-dimension accuracy + violation precision/recall
    - [if --gold] rhyme-fallback precision/recall, broken out by hard-case category

Krippendorff's alpha is implemented inline (ordinal + nominal), exact for 2 raters,
with no third-party dependency.

Expert labels ship in labels/ (e.g. labels/labels_expertA.json).

  python score_calibration.py --items calibration_items_v2.json --labels labels/labels_expertA.json labels/labels_expertB.json
  python score_calibration.py --labels ... --gold gold.json
  python score_calibration.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                  # calibration -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))

from thaiwrite.prosody.grader_meter import grade_text, rhymes   # noqa: E402

ITEMS_DEFAULT = HERE / "calibration_items.json"


# ============================ Krippendorff's α ============================

def krippendorff_alpha(units: list[list], level: str = "nominal") -> float:
    """α from reliability units (each = list of ratings, missing dropped).
    level: 'nominal' or 'ordinal'. Returns nan if undefined (e.g. <2 values)."""
    o: dict[tuple, float] = defaultdict(float)
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(vals[i], vals[j])] += 1.0 / (m - 1)
    values = sorted({c for c, _ in o} | {k for _, k in o})
    if len(values) < 2:
        return float("nan")
    n_c = {v: sum(o.get((v, k), 0.0) for k in values) for v in values}
    n = sum(n_c.values())
    if n < 2:
        return float("nan")

    if level == "nominal":
        def d2(c, k):
            return 0.0 if c == k else 1.0
    elif level == "ordinal":
        def d2(c, k):
            lo, hi = (c, k) if c <= k else (k, c)
            s = sum(n_c[g] for g in values if lo <= g <= hi)
            return (s - (n_c[c] + n_c[k]) / 2.0) ** 2
    else:
        raise ValueError(level)

    do = sum(cnt * d2(c, k) for (c, k), cnt in o.items())
    de = sum(n_c[c] * n_c[k] * d2(c, k) for c in values for k in values)
    if de == 0:
        return float("nan")
    return 1.0 - (n - 1) * do / de


# ============================ data loading ============================

def load_items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(paths: list[Path]) -> list[tuple[str, dict]]:
    out = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        out.append((data.get("annotator", p.stem), data.get("labels", {})))
    return out


# ============================ Layer 2: judge axes ============================

def human_alpha_axes(poems, labelsets):
    res = {}
    for axis in ("topicality", "coherence"):
        units = []
        for it in poems:
            vals = [ls.get(it["id"], {}).get(axis) for _, ls in labelsets]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if len(vals) >= 2:
                units.append(vals)
        res[axis] = (krippendorff_alpha(units, "ordinal"), len(units))
    return res


# ============================ Layer 1: meter dims ============================

def human_alpha_dims(poems, labelsets):
    res = {}
    for dim in ("count", "tone", "rhyme", "kru_lahu"):
        units = []
        for it in poems:
            if dim not in it.get("applicable_dims", []):
                continue
            vals = []
            for _, ls in labelsets:
                mv = ls.get(it["id"], {}).get("meter", {})
                if dim in mv:
                    vals.append(1 if mv[dim] else 0)
            if len(vals) >= 2:
                units.append(vals)
        if units:
            res[dim] = (krippendorff_alpha(units, "nominal"), len(units))
    return res


def grader_dim_verdict(item) -> dict[str, bool]:
    """Grader's compliant?(✓) per applicable dimension: passed == total (total>0)."""
    g = grade_text(item["poem_text"], item["meter_id"])
    out = {}
    for dim, layer in g["layers"].items():
        if layer["total"] > 0:
            out[dim] = (layer["passed"] == layer["total"])
    return out


def grader_vs_gold_dims(poems, gold):
    """Per dimension: accuracy + precision/recall of detecting a VIOLATION (positive=violated)."""
    stats = {}
    for it in poems:
        gl = gold.get(it["id"], {}).get("meter")
        if not gl:
            continue
        gv = grader_dim_verdict(it)
        for dim, gold_ok in gl.items():
            if dim not in gv:
                continue
            s = stats.setdefault(dim, {"acc_n": 0, "acc_c": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
            grader_ok = gv[dim]
            s["acc_n"] += 1
            s["acc_c"] += int(grader_ok == gold_ok)
            gold_viol, grader_viol = (not gold_ok), (not grader_ok)   # positive = violated
            if gold_viol and grader_viol: s["tp"] += 1
            elif (not gold_viol) and grader_viol: s["fp"] += 1
            elif gold_viol and (not grader_viol): s["fn"] += 1
            else: s["tn"] += 1
    return stats


# ============================ rhyme micro-set ============================

def human_alpha_rhyme(pairs, labelsets):
    units = []
    for it in pairs:
        vals = [ls.get(it["id"], {}).get("valid") for _, ls in labelsets]
        vals = [1 if v else 0 for v in vals if isinstance(v, bool)]
        if len(vals) >= 2:
            units.append(vals)
    return krippendorff_alpha(units, "nominal"), len(units)


def rhyme_fallback_vs_gold(pairs, gold):
    """grader.rhymes(a,b) vs gold valid: overall + by hard-case category."""
    by = defaultdict(lambda: {"n": 0, "c": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
    overall = {"n": 0, "c": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for it in pairs:
        g = gold.get(it["id"], {})
        if "valid" not in g:
            continue
        gold_v = bool(g["valid"])
        pred = bool(rhymes(it["word_a"], it["word_b"]))
        for bucket in (overall, by[it.get("_cat", "?")]):
            bucket["n"] += 1
            bucket["c"] += int(pred == gold_v)
            if gold_v and pred: bucket["tp"] += 1            # positive = "rhymes"
            elif (not gold_v) and pred: bucket["fp"] += 1
            elif gold_v and (not pred): bucket["fn"] += 1
            else: bucket["tn"] += 1
    return overall, dict(by)


# ============================ reporting ============================

def _pr(s):
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r


def _fmt(x):
    return "  n/a" if (isinstance(x, float) and math.isnan(x)) else f"{x:5.3f}"


def report(items, labelsets, gold, judge_results):
    poems = [it for it in items if it.get("kind") == "poem"]
    pairs = [it for it in items if it.get("kind") == "rhyme_pair"]
    print(f"\nItems: {len(poems)} poems · {len(pairs)} rhyme pairs · {len(labelsets)} annotators "
          f"({', '.join(a for a, _ in labelsets)})\n")

    print("── Layer 2 · coherence judge ───────────────────────────────")
    print("  human–human α (ordinal) — the reliability ceiling:")
    for axis, (a, n) in human_alpha_axes(poems, labelsets).items():
        print(f"    {axis:<12} α = {_fmt(a)}   (n={n})")
    if judge_results:
        print("  judge vs gold α (ordinal) — gate ≥ 0.65:")
        for model, axes in judge_results.items():
            line = "   ".join(f"{ax}={_fmt(a)}" for ax, a in axes.items())
            print(f"    {model:<16} {line}")

    print("\n── Layer 1 · meter grader ──────────────────────────────────")
    print("  human–human α (nominal) per dimension:")
    for dim, (a, n) in human_alpha_dims(poems, labelsets).items():
        print(f"    {dim:<10} α = {_fmt(a)}   (n={n})")
    if gold:
        print("  grader vs gold — accuracy · violation precision/recall:")
        for dim, s in grader_vs_gold_dims(poems, gold).items():
            acc = s["acc_c"] / s["acc_n"] if s["acc_n"] else float("nan")
            p, r = _pr(s)
            print(f"    {dim:<10} acc={_fmt(acc)}  P={_fmt(p)} R={_fmt(r)}  (n={s['acc_n']})")

    print("\n── Rhyme micro-set ─────────────────────────────────────────")
    a, n = human_alpha_rhyme(pairs, labelsets)
    print(f"  human–human α (nominal) = {_fmt(a)}   (n={n})")
    if gold:
        overall, by = rhyme_fallback_vs_gold(pairs, gold)
        if overall["n"]:
            p, r = _pr(overall)
            acc = overall["c"] / overall["n"]
            print(f"  fallback vs gold: acc={_fmt(acc)}  P={_fmt(p)} R={_fmt(r)}  (n={overall['n']})")
            print("  by category (en/ev easy · an อักษรนำ · ka การันต์ · sv สระ-variant · ei easy-invalid):")
            for cat, s in sorted(by.items()):
                ca = s["c"] / s["n"] if s["n"] else float("nan")
                print(f"    {cat:<4} acc={_fmt(ca)}  (n={s['n']})")
    print()


# ============================ judge bake-off (optional) ============================

def run_judge(poems, model, gold):
    from thaiwrite.prosody.grader_coherence import judge_coherence
    pred = {"topicality": [], "coherence": []}
    units = {"topicality": [], "coherence": []}
    for it in poems:
        g = gold.get(it["id"], {})
        if "topicality" not in g and "coherence" not in g:
            continue
        res = judge_coherence(it["poem_text"], it, model=model)
        if res.error:
            print(f"    [judge err] {it['id']}: {res.error}")
            continue
        for axis, val in (("topicality", res.topicality), ("coherence", res.coherence)):
            if isinstance(g.get(axis), (int, float)) and val is not None:
                units[axis].append([g[axis], val])
    return {ax: krippendorff_alpha(u, "ordinal") for ax, u in units.items()}


# ============================ selftest ============================

def selftest():
    assert krippendorff_alpha([[1, 1], [2, 2], [3, 3]], "nominal") == 1.0
    a = krippendorff_alpha([[1, 1], [2, 2], [1, 2], [2, 1]], "nominal")
    assert round(a, 3) == 0.125, a
    assert krippendorff_alpha([[5, 5], [4, 4], [3, 3]], "ordinal") == 1.0
    # near-miss is rewarded under ordinal vs nominal
    units = [[5, 4], [4, 5], [1, 1], [5, 5], [4, 4]]
    a_ord = krippendorff_alpha(units, "ordinal")
    a_nom = krippendorff_alpha(units, "nominal")
    assert a_ord > a_nom, (a_ord, a_nom)
    print(f"selftest OK  (balanced-2x2 α=0.125; ordinal {a_ord:.3f} > nominal {a_nom:.3f} on near-miss)")


# ============================ main ============================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=ITEMS_DEFAULT)
    ap.add_argument("--labels", type=Path, nargs="+", help="labels_<id>.json files (≥2)")
    ap.add_argument("--gold", type=Path, help="consensus gold.json (enables grader validation)")
    ap.add_argument("--judge", nargs="+", help="judge model id(s) to bake off vs gold (calls API)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.labels or len(args.labels) < 2:
        ap.error("need ≥2 --labels files (or --selftest)")

    items = load_items(args.items)
    labelsets = load_labels(args.labels)
    gold = json.loads(args.gold.read_text(encoding="utf-8")).get("labels", {}) if args.gold else None

    judge_results = None
    if args.judge:
        if not gold:
            ap.error("--judge requires --gold")
        poems = [it for it in items if it.get("kind") == "poem"]
        judge_results = {m: run_judge(poems, m, gold) for m in args.judge}

    report(items, labelsets, gold, judge_results)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
