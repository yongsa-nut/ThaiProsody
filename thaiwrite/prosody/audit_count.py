"""Audit the syllable-COUNT accuracy of the grader's tokenizer on a larger gold set.

The rule grader's fallback syllable counter is `split_syllables` = pythainlp
`syllable_tokenize(engine="dict")` (grader_meter.py). A 26-line (วรรค)
hand-segmented audit (docs/khavee-audit.md, syllable tokenization) put its
accuracy at about 92% on classical verse. n=26 is small, so this script:

  1. Reproduces the LINE-level number from the 26 hand-segmented audit_poems
     วรรค, with a bootstrap 95% CI (so the width at n=26 is explicit).
  2. Measures a larger WORD-level gold probe (words whose pronounced-syllable
     count is a dictionary fact), stratified by the constructs that drive the
     error (การันต์ silent finals, อักษรนำ, consonant clusters, Pali/Sanskrit
     polysyllables, native compounds) plus a monosyllabic control. This gives a
     tighter overall number and shows WHERE the tokenizer fails.

Gold provenance: the 26 lines are hand-segmented against ราชบัณฑิต/สารานุกรมไทยฯ
(audit_poems.py). The word counts are standard pronounced-syllable counts; the
genuinely ambiguous ones are excluded to keep gold clean. The harness reads any
additional (วรรค, gold-count) rows added to the gold lists.

  python audit_count.py
  python audit_count.py --engines dict han_solo --boot 5000
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

from pythainlp.tokenize import syllable_tokenize

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # repo root
from thaiwrite.prosody.data.audit_poems import POEMS as AUDIT_POEMS      # noqa: E402
from thaiwrite.prosody import meter_specs as M                          # noqa: E402


def count(text: str, engine: str) -> int:
    return len(syllable_tokenize(text, engine=engine)) if text.strip() else 0


# ---------- (1) line-level gold: the 26 hand-segmented audit วรรค ----------

def line_gold() -> list[dict]:
    """Each audit วรรค: text = joined syllables, gold = hand-segmented length."""
    rows = []
    for poem in AUDIT_POEMS:
        meter = M.METER_SPECS_BY_ID[poem.meter_id].name_th
        for vrk in poem.lines:
            rows.append({"meter": meter, "text": "".join(vrk), "gold": len(vrk)})
    return rows


# ---------- (2) word-level gold probe (pronounced-syllable counts) ----------
# (word, syllables). Counts are standard pronounced-syllable counts; ambiguous
# words deliberately omitted. `cat` groups by the construct under test.
WORD_GOLD: dict[str, list[tuple[str, int]]] = {
    "mono (control)": [
        ("น้ำ", 1), ("ใจ", 1), ("ไป", 1), ("มา", 1), ("รัก", 1), ("ทอง", 1),
        ("ลม", 1), ("ฝน", 1), ("ดาว", 1), ("ป่า", 1), ("เขา", 1), ("ดี", 1),
        ("ปี", 1), ("เห็น", 1), ("คิด", 1), ("เรา", 1), ("ของ", 1), ("ใหม่", 1),
        ("ไกล", 1), ("กก", 1),
    ],
    "การันต์ (silent final)": [
        ("ศักดิ์", 1), ("สิทธิ์", 1), ("จันทร์", 1), ("พันธุ์", 1), ("สงฆ์", 1),
        ("เล่ห์", 1), ("สัตว์", 1), ("ทรัพย์", 1), ("พงศ์", 1), ("เมล์", 1),
        ("รักษ์", 1), ("ลักษณ์", 1), ("โทรศัพท์", 3),
    ],
    "อักษรนำ silent-ห (1 syl)": [
        ("หนา", 1), ("หมด", 1), ("หลง", 1), ("หนาว", 1), ("หมาย", 1), ("อย่า", 1),
        ("อยู่", 1), ("ไหล", 1), ("ใหญ่", 1), ("หรือ", 1), ("หญิง", 1), ("หรู", 1),
    ],
    "อักษรนำ (2 syl)": [
        ("ขนม", 2), ("ถนน", 2), ("สนาม", 2), ("จมูก", 2), ("ตลาด", 2),
        ("สบาย", 2), ("อร่อย", 2), ("ฉลาด", 2),
    ],
    "consonant cluster (1 syl)": [
        ("ปลา", 1), ("กลาง", 1), ("ความ", 1), ("ขวาน", 1), ("ครวญ", 1), ("กลอน", 1),
        ("เพลง", 1), ("พราว", 1), ("ปราย", 1), ("กรง", 1), ("ปรับ", 1), ("ครับ", 1),
        ("พลาด", 1), ("ขลาด", 1),
    ],
    "Pali/Sanskrit polysyllable": [
        ("มนุษย์", 2), ("กรุณา", 3), ("ศรัทธา", 2), ("ประเทศ", 2), ("ธรรมชาติ", 3),
        ("สมุทร", 2), ("อาทิตย์", 2), ("สวรรค์", 2), ("มหา", 2), ("ราชา", 2),
        ("วิญญาณ", 2), ("สังเกต", 2), ("ปรารถนา", 3), ("วิทยา", 3),
    ],
    "native compound (2 syl)": [
        ("ดอกไม้", 2), ("ต้นไม้", 2), ("แม่น้ำ", 2), ("ภูเขา", 2), ("ทะเล", 2),
        ("ตะวัน", 2), ("ฤดู", 2), ("ครอบครัว", 2), ("เพื่อนบ้าน", 2), ("ความรัก", 2),
    ],
}


# ---------- scoring ----------

def boot_ci(hits: list[int], n_boot: int, seed: int) -> tuple[float, float]:
    if not hits:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(hits)
    means = []
    for _ in range(n_boot):
        s = sum(hits[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (lo, hi)


def acc(hits: list[int]) -> float:
    return sum(hits) / len(hits) if hits else 0.0


def run(engines: list[str], n_boot: int, seed: int) -> None:
    print("=" * 64)
    print("SYLLABLE-COUNT TOKENIZER AUDIT")
    print("=" * 64)

    lines = line_gold()

    # --- (1) line-level, per engine ---
    print(f"\n(1) LINE-LEVEL — {len(lines)} hand-segmented audit วรรค\n")
    for eng in engines:
        hits = [1 if count(r["text"], eng) == r["gold"] else 0 for r in lines]
        within1 = [1 if abs(count(r["text"], eng) - r["gold"]) <= 1 else 0 for r in lines]
        lo, hi = boot_ci(hits, n_boot, seed)
        print(f"  {eng:9s}  exact {sum(hits)}/{len(lines)} = {acc(hits)*100:4.1f}%  "
              f"[95% CI {lo*100:.0f}-{hi*100:.0f}]   within±1 {acc(within1)*100:4.1f}%")

    # per-meter (dict only)
    eng0 = engines[0]
    by_meter: dict[str, list[int]] = defaultdict(list)
    for r in lines:
        by_meter[r["meter"]].append(1 if count(r["text"], eng0) == r["gold"] else 0)
    print(f"\n  per-meter ({eng0}):")
    for m, h in by_meter.items():
        print(f"    {m:22s} {sum(h)}/{len(h)} = {acc(h)*100:4.1f}%")

    # line mismatches
    print(f"\n  line mismatches ({eng0}):")
    any_miss = False
    for r in lines:
        c = count(r["text"], eng0)
        if c != r["gold"]:
            any_miss = True
            print(f"    {r['text']:<22} gold {r['gold']}  got {c}")
    if not any_miss:
        print("    (none)")

    # --- (2) word-level probe ---
    flat = [(w, g, cat) for cat, ws in WORD_GOLD.items() for (w, g) in ws]
    print(f"\n(2) WORD-LEVEL PROBE — {len(flat)} words, dictionary counts\n")
    for eng in engines:
        hits = [1 if count(w, eng) == g else 0 for (w, g, _) in flat]
        lo, hi = boot_ci(hits, n_boot, seed)
        print(f"  {eng:9s}  {sum(hits)}/{len(flat)} = {acc(hits)*100:4.1f}%  "
              f"[95% CI {lo*100:.0f}-{hi*100:.0f}]")

    print(f"\n  per-category ({eng0}):")
    miss_words = []
    for cat, ws in WORD_GOLD.items():
        h = []
        for (w, g) in ws:
            c = count(w, eng0)
            ok = c == g
            h.append(1 if ok else 0)
            if not ok:
                miss_words.append((cat, w, g, c))
        print(f"    {cat:28s} {sum(h)}/{len(h)} = {acc(h)*100:5.1f}%")

    print(f"\n  word mismatches ({eng0}):")
    for cat, w, g, c in miss_words:
        print(f"    {w:<12} gold {g}  got {c}   [{cat}]")
    if not miss_words:
        print("    (none)")

    print("\n" + "=" * 64)
    print("Gold note: 26 lines hand-segmented vs ราชบัณฑิต; word counts are standard")
    print("pronounced-syllable counts.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="+", default=["dict", "han_solo"])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260525)
    args = ap.parse_args()
    run(args.engines, args.boot, args.seed)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
