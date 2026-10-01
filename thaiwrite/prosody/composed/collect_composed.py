"""Re-verify all LLM-composed source poems, dedup, and print them for inspection.

Loads composed_{khlong,chan}_*.json, re-scores each poem with the rule grader
(an independent check of the composer's own verification), flags any score
< 1.0, detects duplicate texts, and prints the joined วรรค so coherence and
segmentation can be inspected by eye. Writes the merged, verified, deduplicated
pool to all_composed.json.

  python collect_composed.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent       # composed -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))
from thaiwrite.prosody.grader_meter import score_meter      # noqa: E402


def main() -> None:
    seen: dict[str, str] = {}
    kept: list[dict] = []
    by_meter: dict[str, int] = {}
    by_prov: dict[str, int] = {}
    fails = 0
    for f in sorted(glob.glob(str(HERE / "composed_*.json"))):
        for p in json.loads(Path(f).read_text(encoding="utf-8")):
            lines = [tuple(v) for v in p["lines"]]
            mid = p["meter_id"]
            prov = p.get("provenance", "Opus 4.7 (composed)")
            r = score_meter(lines, mid, eval_rhyme=p.get("eval_rhyme", True))
            joined = " / ".join("".join(v) for v in lines)
            key = "".join("".join(v) for v in lines)
            dup = key in seen
            tag = "DUP" if dup else ("OK " if r["score"] >= 0.9999 else "FAIL")
            if r["score"] < 0.9999:
                fails += 1
            print(f"[{tag}] {r['score']:.3f} {mid[:6]} {('real' if 'real' in prov else 'opus')} "
                  f"{p.get('source','')!s:<20} | {joined}")
            if not dup and r["score"] >= 0.9999:
                seen[key] = p.get("source", "")
                kept.append({"meter_id": mid, "source": p.get("source", ""),
                             "lines": [list(v) for v in lines], "provenance": prov})
                by_meter[mid] = by_meter.get(mid, 0) + 1
                pk = "real" if "real" in prov else "opus"
                by_prov[f"{mid}:{pk}"] = by_prov.get(f"{mid}:{pk}", 0) + 1
    (HERE / "all_composed.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nkept {len(kept)} unique 1.0 poems: " +
          ", ".join(f"{k}={v}" for k, v in by_meter.items()) +
          f" | fails={fails} | -> all_composed.json")
    print("provenance:", ", ".join(f"{k}={v}" for k, v in sorted(by_prov.items())))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
