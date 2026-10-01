"""Rule-grader check for composed source poems.

Reads a JSON array of poems, each {meter_id, source, lines, [eval_rhyme]} where
`lines` = list of วรรค, each วรรค = list of pronounced-syllable strings (the GOLD
segmentation). Scores each with the rule-based meter grader and, for any failing
layer, prints the exact failing position. Because the SUPPLIED segmentation is
graded, the count layer is exact (no tokenizer): a count failure means a วรรค
has the wrong number of syllables.

  python verify_composed.py mypoems.json      # exit 0 iff every poem scores 1.0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent                   # composed -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))
from thaiwrite.prosody.grader_meter import score_meter   # noqa: E402

_LAYER_FMT = {
    "count": lambda d: f"วรรค{d[0]}: has {d[1]} พยางค์, need {d[2]}",
    "tone": lambda d: f"(วรรค{d[0]},พยางค์{d[1]}) '{d[2]}': need {d[3]}, got {d[4]}",
    "kru_lahu": lambda d: f"(วรรค{d[0]},พยางค์{d[1]}) '{d[2]}': need {d[3]}, got {d[4]}",
    "rhyme": lambda d: f"src '{d[0]}' must rhyme one of {d[1]} — matched {d[2]}",
}


def check(path: str) -> int:
    poems = json.loads(Path(path).read_text(encoding="utf-8"))
    all_ok = True
    for k, p in enumerate(poems):
        lines = [tuple(v) for v in p["lines"]]
        r = score_meter(lines, p["meter_id"], eval_rhyme=p.get("eval_rhyme", True))
        ok = r["score"] >= 0.9999
        all_ok &= ok
        print(f"\n=== poem {k} [{p['meter_id']}] {p.get('source','')!r} "
              f"score={r['score']}  {'PASS ✅' if ok else 'FAIL ❌'}")
        for i, v in enumerate(lines):
            print(f"   วรรค{i} ({len(v)}): {''.join(v)}   [{'.'.join(v)}]")
        print("   rates:", r["layer_rates"])
        if not ok:
            for lname, L in r["layers"].items():
                if L["total"] and L["passed"] < L["total"]:
                    print(f"   ── {lname} {L['passed']}/{L['total']}:")
                    for d in L["details"]:
                        if not d[-1]:
                            print("       ✗", _LAYER_FMT[lname](d))
    print("\n✅ ALL POEMS PASS (1.0)" if all_ok else "\n❌ SOME FAILED — revise & re-run")
    return 0 if all_ok else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(check(sys.argv[1]))
