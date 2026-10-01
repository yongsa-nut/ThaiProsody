"""Judge-vs-expert agreement on a BINARY coherence/topicality decision.

Loads expert A's 1-5 labels and the judge caches (gpt-5.5, Opus) and, for each
cut point t (good = score >= t), computes judge-vs-expert binary agreement: raw %
and Cohen's kappa. This complements the ordinal alpha on the 1-5 scale.

  python binary_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = HERE / "labels" / "labels_expertA.json"


def cohen_kappa(pairs):
    """binary (0/1) Cohen's kappa."""
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    k = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return po, k


def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    caches = {
        "gpt-5.5": json.loads((HERE / "judge_cache_gpt55.json").read_text(encoding="utf-8")),
        "Opus":    json.loads((HERE / "judge_cache_opus.json").read_text(encoding="utf-8")),
    }

    for axis in ("coherence", "topicality"):
        print(f"\n================  {axis}  ================")
        print(f"{'cut (good= >=t)':>16} | {'model':<8} | {'n':>3} | {'%bad(exp)':>9} | {'agree%':>7} | {'kappa':>6}")
        for t in (2, 3, 4):   # good = score >= t  → bad = score <= t-1
            for model, cache in caches.items():
                pairs = []
                for iid, lab in labels.items():
                    h = lab.get(axis)
                    j = cache.get(iid, {}).get(axis)
                    if isinstance(h, (int, float)) and isinstance(j, (int, float)):
                        pairs.append((1 if h >= t else 0, 1 if j >= t else 0))
                po, k = cohen_kappa(pairs)
                pbad = sum(1 for a, _ in pairs if a == 0) / len(pairs) if pairs else 0
                print(f"{'>= '+str(t):>16} | {model:<8} | {len(pairs):>3} | {pbad*100:>8.0f}% | "
                      f"{po*100:>6.0f}% | {k:>6.3f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
