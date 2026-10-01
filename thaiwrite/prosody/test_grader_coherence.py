"""Live API test for grader_coherence (not a calibration of the judge).

Proves the gpt-5.5 path works end-to-end: structured scores come back, axes clamp
to [1,5], composite == mean, and gibberish scores below a coherent poem.

Run:  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/test_grader_coherence.py
Skips cleanly if OPENAI_API_KEY2 (or OPENAI_API_KEY) is unset. Makes a few real
gpt-5.5 calls (~$0.01).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thaiwrite.prosody import grader_coherence as gc

# A real, coherent กาพย์พระไชยสุริยา stanza about fleeing the city into hardship.
COHERENT = {
    "id": "smoke-coherent",
    "topic": "ความลำบากเมื่อต้องจากบ้านเมืองหนีเข้าป่า",
    "poem": "ขึ้นกกตกทุกข์ยาก แสนลำบากจากเวียงชัย\nมันเผือกเลือกเผาไฟ กินผลไม้ได้เป็นแรง",
}
# Meter-valid but semantically empty (single word repeated).
GIBBERISH = {
    "id": "smoke-gibberish",
    "topic": "สายฝน",
    "poem": "น้ำ น้ำ น้ำ ตก ตก ตก ตก ตก",
}


def main() -> int:
    if not (os.environ.get("OPENAI_API_KEY2") or os.environ.get("OPENAI_API_KEY")):
        print("SKIP: no OPENAI_API_KEY2 / OPENAI_API_KEY in environment.")
        return 0

    print(f"prompt version: {gc.PROMPT_VERSION}")
    coh = gc.judge_coherence(COHERENT["poem"], COHERENT)
    gib = gc.judge_coherence(GIBBERISH["poem"], GIBBERISH)

    for label, r in (("coherent", coh), ("gibberish", gib)):
        print(f"  [{label:9}] model={r.model} {r.latency_s}s  "
              f"top={r.topicality} coh={r.coherence} composite={r.composite}  "
              f"err={r.error}")
        print(f"             rationale: {r.rationale[:140]}")

    fails = []
    for label, r in (("coherent", coh), ("gibberish", gib)):
        if r.error:
            fails.append(f"{label}: API error: {r.error}")
            continue
        if not (r.topicality and 1 <= r.topicality <= 5):
            fails.append(f"{label}: topicality out of range: {r.topicality}")
        if not (r.coherence and 1 <= r.coherence <= 5):
            fails.append(f"{label}: coherence out of range: {r.coherence}")
        if r.composite != (r.topicality + r.coherence) / 2:
            fails.append(f"{label}: composite != mean")

    if not coh.error and not gib.error:
        if not (gib.coherence < coh.coherence):
            fails.append(f"gibberish coherence ({gib.coherence}) not < coherent ({coh.coherence})")
        # soft expectations (warn only)
        if coh.coherence < 4:
            print(f"  NOTE: coherent coherence {coh.coherence} < 4 (soft expectation)")
        if gib.coherence > 2:
            print(f"  NOTE: gibberish coherence {gib.coherence} > 2 (soft expectation)")

    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS: structured output valid; gibberish scores below coherent.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
