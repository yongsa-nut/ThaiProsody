"""Prepare a word set for the LLM syllable-count audit.

Mines distinct real-orthography Thai words from the verse corpus and a sample of
model compose outputs, adds the 91 hand-probe words (tagged with known gold so
the LLM counts can be checked against them), dedupes, caps, and splits the set
into N batch files (audit_words/batch_NN.json). Each batch was counted by
Claude Opus 4.7; the counts are in audit_words/counts_NN.json and are scored by
audit_words_score.py.

  python audit_words_prep.py --cap 120 --batches 10
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))        # repo root

from pythainlp.tokenize import word_tokenize                            # noqa: E402
from thaiwrite.prosody.data.real_source_poems import REAL_POEMS         # noqa: E402

OUTDIR = HERE / "audit_words"
GRADED = HERE / "eval" / "graded"
THAI = re.compile(r"[ก-๙]")
TONE_MARKS = "่้๊๋"

# the 91 hand-probe words with hand counts (to check the LLM counts against), from audit_count.py
PROBE_GOLD: dict[str, int] = {
    "น้ำ":1,"ใจ":1,"ไป":1,"มา":1,"รัก":1,"ทอง":1,"ลม":1,"ฝน":1,"ดาว":1,"ป่า":1,"เขา":1,
    "ดี":1,"ปี":1,"เห็น":1,"คิด":1,"เรา":1,"ของ":1,"ใหม่":1,"ไกล":1,"กก":1,
    "ศักดิ์":1,"สิทธิ์":1,"จันทร์":1,"พันธุ์":1,"สงฆ์":1,"เล่ห์":1,"สัตว์":1,"ทรัพย์":1,"พงศ์":1,
    "เมล์":1,"รักษ์":1,"ลักษณ์":1,"โทรศัพท์":3,
    "หนา":1,"หมด":1,"หลง":1,"หนาว":1,"หมาย":1,"อย่า":1,"อยู่":1,"ไหล":1,"ใหญ่":1,"หรือ":1,
    "หญิง":1,"หรู":1,
    "ขนม":2,"ถนน":2,"สนาม":2,"จมูก":2,"ตลาด":2,"สบาย":2,"อร่อย":2,"ฉลาด":2,
    "ปลา":1,"กลาง":1,"ความ":1,"ขวาน":1,"ครวญ":1,"กลอน":1,"เพลง":1,"พราว":1,"ปราย":1,"กรง":1,
    "ปรับ":1,"ครับ":1,"พลาด":1,"ขลาด":1,
    "มนุษย์":2,"กรุณา":3,"ศรัทธา":2,"ประเทศ":2,"ธรรมชาติ":3,"สมุทร":2,"อาทิตย์":2,"สวรรค์":2,
    "มหา":2,"ราชา":2,"วิญญาณ":2,"สังเกต":2,"ปรารถนา":3,"วิทยา":3,
    "ดอกไม้":2,"ต้นไม้":2,"แม่น้ำ":2,"ภูเขา":2,"ทะเล":2,"ตะวัน":2,"ฤดู":2,"ครอบครัว":2,
    "เพื่อนบ้าน":2,"ความรัก":2,
}


def clean(tok: str) -> str:
    return tok.strip()


def is_word(tok: str) -> bool:
    if not tok or " " in tok or "\n" in tok:
        return False
    if not THAI.search(tok):
        return False
    # drop a token that is only a tone mark / combining char
    if all(c in TONE_MARKS for c in tok):
        return False
    return True


def mine_corpus(n_poems: int, seed: int) -> set[str]:
    words: set[str] = set()
    # real source poems (already word-token tuples, real orthography)
    for poem in REAL_POEMS:
        for vrk in poem.lines:
            for w in vrk:
                if is_word(w):
                    words.add(w)
    # sample of model compose outputs → newmm word tokens (real orthography)
    files = [f for f in glob.glob(str(GRADED / "*__run1.jsonl")) if ".nothink" not in f]
    rng = random.Random(seed)
    poems: list[str] = []
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("task_mode") == "compose" and (r.get("poem") or "").strip():
                poems.append(r["poem"])
    rng.shuffle(poems)
    for poem in poems[:n_poems]:
        flat = poem.replace("\n", " ")
        for w in word_tokenize(flat, engine="newmm"):
            w = clean(w)
            if is_word(w) and len(w) >= 2:
                words.add(w)
    return words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--n-poems", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260525)
    args = ap.parse_args()

    mined = mine_corpus(args.n_poems, args.seed)
    probe = set(PROBE_GOLD)
    # prioritise multi-char (likely multi-syllable) mined words, then fill
    mined_only = sorted(mined - probe, key=lambda w: (-len(w), w))

    # build set: all probe words (have gold) + mined, capped
    selected = list(probe) + [w for w in mined_only]
    selected = selected[:args.cap]
    rng = random.Random(args.seed)
    rng.shuffle(selected)

    OUTDIR.mkdir(exist_ok=True)
    meta = {w: {"probe_gold": PROBE_GOLD.get(w)} for w in selected}
    (OUTDIR / "words_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # split into batches
    batches: list[list[str]] = [[] for _ in range(args.batches)]
    for i, w in enumerate(selected):
        batches[i % args.batches].append(w)
    for i, b in enumerate(batches):
        (OUTDIR / f"batch_{i:02d}.json").write_text(
            json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")

    n_probe = sum(1 for w in selected if w in probe)
    print(f"mined distinct corpus words: {len(mined)}")
    print(f"selected: {len(selected)}  ({n_probe} probe w/ known gold, "
          f"{len(selected)-n_probe} new mined)")
    print(f"wrote {args.batches} batches → {OUTDIR.relative_to(HERE)}/batch_NN.json")
    print(f"      + words_meta.json")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
