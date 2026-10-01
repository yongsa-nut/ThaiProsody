"""Compare the coherence judge against one human expert (expert A).

Expert A's labels on the calibration set serve as gold, and the question is how
well the LLM judge tracks the human on topicality and coherence.

Because expert A's labels are skewed high (most poems are on-topic and coherent),
a bare Krippendorff alpha understates agreement. We therefore report a fuller
picture per axis: alpha (ordinal), Pearson r, exact-agree %, within-1 %, MAE, and
the safety metric: recall on the poems the human rated low (<= 2), i.e. does the
judge also catch them?

Resumable: judge calls are cached to judge_cache_<model>.json (a re-run is free).
Items are judged concurrently (threads), since each is an independent API call.

  python judge_vs_expert.py                        # gpt-5.5 medium
  python judge_vs_expert.py --model gpt-5.5 --workers 6
  python judge_vs_expert.py --limit 8              # quick check on 8 items
  python judge_vs_expert.py --model opus --score-only   # score an existing cache
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                  # calibration -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))

ITEMS = HERE / "calibration_items_v2.json"
LABELS = HERE / "labels" / "labels_expertA.json"


def load_env() -> None:
    """Load the nearest .env (searching up from the working directory), overriding OS env."""
    import os
    p = Path.cwd()
    for d in [p, *p.parents]:
        if (d / ".env").exists():
            for line in (d / ".env").read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
            return


def krippendorff_alpha(units, level="ordinal"):
    from score_calibration import krippendorff_alpha as ka
    return ka(units, level)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N items")
    ap.add_argument("--score-only", action="store_true",
                    help="score whatever is already in judge_cache_<model>.json; no API calls "
                         "(for caches produced by another judge backend, e.g. judge_cache_opus.json)")
    args = ap.parse_args()

    load_env()
    sys.path.insert(0, str(HERE))           # score_calibration

    items = {it["id"]: it for it in json.loads(ITEMS.read_text(encoding="utf-8"))
             if it.get("kind") == "poem"}
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]

    # items the expert scored on at least one judge axis
    todo = [iid for iid, lab in labels.items()
            if iid in items and (("topicality" in lab) or ("coherence" in lab))]
    if args.limit:
        todo = todo[: args.limit]

    cache_path = HERE / f"judge_cache_{args.model.replace('.', '').replace('-', '')}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    lock = threading.Lock()

    pending = [iid for iid in todo if iid not in cache]
    print(f"items expert-labeled: {len(todo)} · cached: {len(todo)-len(pending)} · to call: {len(pending)}")

    if args.score_only:
        pending = []

    if pending:
        sys.path.insert(0, str(HERE.parent))    # grader_coherence (prosody dir)
        from grader_coherence import judge_coherence, OpenAIJudge
        judge = OpenAIJudge(model=args.model, reasoning=args.reasoning)

        def call(iid):
            it = items[iid]
            res = judge_coherence(it["poem_text"], it, judge=judge)
            return iid, {"topicality": res.topicality, "coherence": res.coherence,
                         "error": res.error}

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(call, iid) for iid in pending]
            for fut in as_completed(futs):
                iid, rec = fut.result()
                with lock:
                    cache[iid] = rec
                    done += 1
                    if rec["error"]:
                        print(f"  [err] {iid}: {rec['error']}")
                    if done % 20 == 0:
                        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
                        print(f"  …{done}/{len(pending)}")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- metrics per axis ----
    L = [f"# Coherence judge vs expert A — {args.model} ({args.reasoning})\n",
         "Gold: 1 human rater (expertA); judge-vs-human agreement.\n"]
    print()
    for axis in ("topicality", "coherence"):
        pairs = []
        for iid in todo:
            h = labels[iid].get(axis)
            j = cache.get(iid, {}).get(axis)
            if isinstance(h, (int, float)) and isinstance(j, (int, float)):
                pairs.append((h, j))
        if not pairs:
            continue
        hs = [h for h, _ in pairs]
        js = [j for _, j in pairs]
        n = len(pairs)
        alpha = krippendorff_alpha([[h, j] for h, j in pairs], "ordinal")
        exact = sum(1 for h, j in pairs if h == j) / n
        within1 = sum(1 for h, j in pairs if abs(h - j) <= 1) / n
        mae = statistics.mean(abs(h - j) for h, j in pairs)
        # Pearson
        mh, mj = statistics.mean(hs), statistics.mean(js)
        cov = sum((h - mh) * (j - mj) for h, j in pairs)
        sh = math.sqrt(sum((h - mh) ** 2 for h in hs))
        sj = math.sqrt(sum((j - mj) ** 2 for j in js))
        r = cov / (sh * sj) if sh and sj else float("nan")
        # safety: low-flag recall (expert ≤2)
        low = [(h, j) for h, j in pairs if h <= 2]
        low_recall = (sum(1 for h, j in low if j <= 2) / len(low)) if low else float("nan")
        # judge mean per expert level
        per_level = {}
        for lv in range(1, 6):
            v = [j for h, j in pairs if h == lv]
            if v:
                per_level[lv] = statistics.mean(v)

        block = [
            f"## {axis}  (n={n})",
            f"- Krippendorff α (ordinal): **{alpha:.3f}**",
            f"- Pearson r: **{r:.3f}**",
            f"- exact agreement: **{exact*100:.0f}%** · within ±1: **{within1*100:.0f}%** · MAE: **{mae:.2f}**",
            f"- low-flag recall (expert ≤2, n={len(low)}): "
            + (f"**{low_recall*100:.0f}%**" if not math.isnan(low_recall) else "n/a"),
            "- judge mean by expert score: "
            + " · ".join(f"{lv}→{per_level[lv]:.1f}" for lv in sorted(per_level)),
            "",
        ]
        L += block
        print("\n".join(block))

    out = HERE / f"judge-vs-expert-{args.model.replace('.', '').replace('-', '')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ wrote {out.relative_to(HERE)}  (cache: {cache_path.name})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
