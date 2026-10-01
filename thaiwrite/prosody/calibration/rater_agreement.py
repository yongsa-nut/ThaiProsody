"""Agreement between the two human raters, and of the content judges and rhyme checker with them.

Reads the 176 calibration items (calibration_items_v2.json), both raters' labels
(labels/labels_expertA.json = rater 1, labels/labels_expertB.json = rater 2), the adjudicated
binary labels (labels/labels_adjudicated.json: where the raters fall on opposite sides of a
benchmark cut, the label rater 1 set after re-examining the poem with both scores in view),
and the judge scores on the same poems (judge_cache_gpt55.json = GPT-5.5,
judge_cache_opus.json = Claude Opus 4.7, judge_api_calib_claude-opus-5-5.json = Claude Opus
5.5, the judge used for the leaderboard). Reports, for the 132 poems, the score distributions
and binary Cohen's kappa at topicality >= 3 (on-topic), coherence >= 2 (not word salad),
coherence >= 3 and the >= 4 cuts; for the 44 word pairs, rater agreement and the rhyme
checker's precision and recall against each rater. The ordinal Krippendorff alpha on the 1-5
scales comes from score_calibration.py:

  PYTHONIOENCODING=utf-8 python score_calibration.py --items calibration_items_v2.json \
      --labels labels/labels_expertA.json labels/labels_expertB.json
  PYTHONIOENCODING=utf-8 python rater_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from thaiwrite.prosody.grader_meter import rhymes  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_json(path):
    return json.loads((HERE / path).read_text(encoding="utf-8"))


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po


def main():
    r1 = load_json("labels/labels_expertA.json")["labels"]
    r2 = load_json("labels/labels_expertB.json")["labels"]
    adj = load_json("labels/labels_adjudicated.json")["labels"]
    judges = {"GPT-5.5": load_json("judge_cache_gpt55.json"),
              "Opus 4.7": load_json("judge_cache_opus.json"),
              "Opus 5.5": load_json("judge_api_calib_claude-opus-5-5.json")["scores"]}
    items = load_json("calibration_items_v2.json")
    poems = [x["id"] for x in items if x.get("kind") == "poem"]
    pairs = [x for x in items if x.get("kind") != "poem"]

    for axis, cuts in (("topicality", (3, 4)), ("coherence", (2, 3, 4))):
        print(f"{axis} (n={len(poems)})")
        for name, R in (("rater 1", r1), ("rater 2", r2)):
            print(f"  {name} scores 1..5: {[sum(1 for i in poems if R[i][axis] == k) for k in range(1, 6)]}")
        for cut in cuts:
            k12, po12 = kappa([(int(r1[i][axis] >= cut), int(r2[i][axis] >= cut)) for i in poems])
            line = f"  >= {cut}: rater 1 vs rater 2 kappa {k12:.2f} (agreement {po12:.3f})"
            for jn, J in judges.items():
                for rn, R in (("r1", r1), ("r2", r2)):
                    jp = [(int(R[i][axis] >= cut), int(J[i][axis] >= cut)) for i in poems
                          if isinstance(J.get(i, {}).get(axis), int)]
                    line += f"; {jn} vs {rn} {kappa(jp)[0]:.2f}"
            print(line)
        bench_cut = 3 if axis == "topicality" else 2
        for jn, J in list(judges.items()) + [("rater 1", r1), ("rater 2", r2)]:
            jp = [(int(adj[i][axis]["positive"]), int(J[i][axis] >= bench_cut)) for i in poems
                  if isinstance(J.get(i, {}).get(axis), int)]
            k, po = kappa(jp)
            print(f"  >= {bench_cut} vs adjudicated labels: {jn} kappa {k:.2f} (agreement {po:.3f})")

    agree = sum(r1[p["id"]]["valid"] == r2[p["id"]]["valid"] for p in pairs)
    print(f"rhyme pairs: raters agree on {agree}/{len(pairs)}")
    for name, R in (("rater 1", r1), ("rater 2", r2)):
        tp = fp = fn = tn = 0
        for p in pairs:
            g, h = bool(rhymes(p["word_a"], p["word_b"])), bool(R[p["id"]]["valid"])
            tp += g and h
            fp += g and not h
            fn += h and not g
            tn += not g and not h
        print(f"  rhyme checker vs {name}: TP {tp} FP {fp} FN {fn} TN {tn}; "
              f"precision {tp / (tp + fp):.3f}, recall {tp / (tp + fn):.3f}")


if __name__ == "__main__":
    main()
