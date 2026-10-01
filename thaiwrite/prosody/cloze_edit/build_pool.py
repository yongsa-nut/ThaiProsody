"""Build the cloze/edit SOURCE POOL (contamination-free stanzas only).

Sources (both never published):
  * student: stanzas written by a native-speaker secondary-school (ม.3) student as
             coursework and used with the author's consent (`sources/student_raw/`).
             The student's words are never changed; only a gold syllable
             segmentation is supplied.
  * model:   stanzas composed by Claude Opus 4.7 and rule-verified
               (`sources/model_composed_all_composed.json`, provenance "Opus 4.7").
Web-found published poems and public-domain classics are excluded (memorization risk).

Every kept stanza must score exactly 1.0 under `grader_meter.score_meter` with its gold
segmentation. Each stanza records, per วรรค, the SURFACE text (orthographic, what the
model is shown) and the gold syllables, aligned into units (a unit is one written
syllable, or a respelled multi-syllable word such as สราญ = สะ.ราญ).

Per-form pool = the benchmark's per-form source-pool size (`TARGET`), student stanzas
first, model-composed stanzas filling the remainder (`in_main`). The remaining verified
model-composed stanzas are kept as a provenance-control set (`in_main = False`).

  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/build_pool.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from thaiwrite.prosody.grader_meter import score_meter  # noqa: E402

SRC = HERE / "sources"
RAW = SRC / "student_raw"
OUT = HERE / "data" / "pool.json"
REPORT = HERE / "data" / "pool_report.md"

TARGET = {"khlong_si_suphap": 12, "klon_paet": 14, "kap_yani_11": 20,
          "intharawichian_chan_11": 12}
SLUG = {"khlong_si_suphap": "khlong", "klon_paet": "klon",
        "kap_yani_11": "yani", "intharawichian_chan_11": "chan"}

# ---------------------------------------------------------------------------
# Student โคลง: the gold file student_khlong_gold_original.json substitutes words in
# 19 วรรค so that they satisfy the tone/rhyme rules (e.g. หล่อหลอม to หล่อพร้อม, ปัญญา
# to ปันคิด). This table restores the student's own words (from the raw text) with
# their segmentation. Key = (stanza 1-based, grader-order วรรค index).
KHLONG_FAITHFUL = {
    (2, 0): ["งาน", "หนัก", "จัก", "หล่อ", "หลอม"],
    (3, 2): ["ปัน", "ยา", "ส่อง", "ติด", "ตาม"],
    (5, 3): ["แน่ว", "แน่"],
    (5, 7): ["ด้วย", "รู้", "จัด", "การ"],
    (6, 0): ["งาน", "สอน", "คน", "รู้", "ค่า"],
    (6, 4): ["สอน", "มือ", "สร้าง", "สัด", "ทา"],
    (6, 6): ["สอน", "ฝัน", "ให้", "ไป", "ต่อ"],
    (6, 7): ["จน", "ถึง", "ปลาย", "ทาง"],
    (7, 0): ["ร่วม", "แรง", "ร่วม", "ร้อย", "ถ้อย"],
    (7, 5): ["ช่วย", "เสริม"],
    (7, 7): ["สำ", "เร็จ", "โดย", "พลัน"],
    (8, 0): ["ผู้", "ใด", "ทำ", "งาน", "ด้วย"],
    (8, 3): ["หยุด", "ฝัน"],
    (8, 6): ["วัน", "หนึ่ง", "ฟ้า", "ย่อม", "ปัน"],
    (8, 7): ["เกียรติ", "ให้", "ผู้", "เพียร"],
    (9, 2): ["หาก", "คือ", "ทาง", "ฝึก", "ตน"],
    (9, 7): ["แม้", "ภาร", "มาก", "มาย"],
    (10, 2): ["ขอ", "แรง", "อย่า", "ร้าว", "ราน"],
    (10, 6): ["ขอ", "ผล", "แห่ง", "เพียร", "กล้า"],
}

# ---------------------------------------------------------------------------
# Model-composed stanzas: manual audit (syllabification by pronunciation + real words).
MODEL_EXCLUDE = {
    "ความสงบในธรรม": "สงบ is two syllables (สะ-หงบ): first วรรค has 6 syllables, not 5",
    "พระคุณครู": "วิทยา is three syllables: first วรรค has 6 syllables, not 5",
    "กาลเวลา (time)": "เวลา is two syllables: first วรรค has 6 syllables, not 5",
    "การให้ทาน": "บุญ respelled บุ.ญะ to force two light syllables (not a valid reading)",
    "ความหวัง (hope)": "non-word fillers (วิมานา, สุมล)",
    "ความรักแท้ (true love)": "non-word fillers (ทุระ, มุทะนาว)",
    "การพลัดพราก (parting)": "non-word fillers (ระทะ, ทุขม)",
    "ดอกไม้ผลิบาน": "ผกา is two syllables: วรรค 3 has 6 syllables and the last วรรค 5",
    "มิตรภาพแท้": "เผชิญ is two syllables: วรรค 3 has 6 syllables",
    "ชีวิต (ความเพียร)": "ทีละ is two syllables: the วรรค-4 rhyme target shifts off the rhyme",
    "ธรรมชาติ (ทะเลยามเช้า)": "ทะเล is two syllables: the วรรค-4 rhyme target shifts off the rhyme",
    "ฤดูกาล (ฝนแรกของปี)": "misspelling (ครื้ม) and a non-word (ผืนดา)",
    "ดอกไม้ในสวน": "ผลิ is two syllables: first วรรค has 6 syllables",
    "ชีวิตและความเพียร": "เผชิญ is two syllables: second วรรค has 7 syllables",
    "เด็กน้อยกับการเรียน": "สมุด is two syllables: second วรรค has 7 syllables",
    "ดวงดาวยามค่ำคืน": "นภา is two syllables: second วรรค has 7 syllables",
}
#: segmentation fixes that keep the stanza valid (a multi-syllable token split).
MODEL_RESEG = {("ชีวิต (กาลเวลา)", 0, "เวลา"): ["เว", "ลา"]}
#: orthographic surface for respelled ฉันท์ tokens (ขะ.ณะ = ขณะ, ชะ.ละ = ชล …).
MODEL_SURFACE = {
    "สายฝน": ["เมฆดำผิหม่นมัว", "ขณะรัวพิรุณมา", "ฝนพรำผิพรมนา", "ชลรินชโลมดิน"],
    "ทะเลกว้าง": ["เวิ้งชลผิครามตา", "คณะคราระเริงลม", "คลื่นลมผิโชยชม", "ระยะเรือระเริงลอย"],
    "ค่ำคืนและดวงดาว": ["คืนนี้ผิพราวดาว", "ระยะวาวสกาวราย", "ดวงดาวผิพราวพราย", "นภยับระยับวับ"],
    "รุ่งอรุณ": ["รุ่งทองอรุณฉาย", "คณะพรายระยับทอง", "แสงทองผิทาผอง", "ระยะทองผิทาแดง"],
    "ความเมตตากรุณา": ["เมตตาผดุงใจ", "บ่ละใครจะทุกข์ตรอม", "โอบเอื้อวนียอม", "ผิจะทนบ่ถือตรอม"],
}


# ---------------------------------------------------------------------------
def align_units(surface: str, sylls: list[str]) -> list[dict]:
    """Align gold syllables to the written วรรค. Returns units
    [{"s": surface_piece, "syl": [syllables]}]; a syllable written as-is is its own
    unit, a respelled syllable (สะ in สราญ) is merged with the next unit, and a run of
    respelled syllables (ปัน.ยา = ปัญญา) forms one unit."""
    units, pos, i = [], 0, 0
    while i < len(sylls):
        t = sylls[i]
        if surface.startswith(t, pos):
            units.append({"s": t, "syl": [t]})
            pos += len(t)
            i += 1
            continue
        # respelled run: extend until a later syllable is found verbatim ahead
        j, q = i + 1, -1
        while j < len(sylls):
            q = surface.find(sylls[j], pos + 1)
            if q != -1:
                break
            j += 1
        piece = surface[pos:q] if j < len(sylls) else surface[pos:]
        grp = {"s": piece, "syl": sylls[i:j]}
        pos += len(piece)
        i = j
        if len(grp["syl"]) == 1 and i < len(sylls) and surface.startswith(sylls[i], pos):
            # single respelled syllable → part of the next written word (ส+ราญ = สราญ)
            nxt = sylls[i]
            grp = {"s": grp["s"] + nxt, "syl": grp["syl"] + [nxt]}
            pos += len(nxt)
            i += 1
        units.append(grp)
    if "".join(u["s"] for u in units) != surface or pos != len(surface):
        raise ValueError(f"alignment failed: {surface!r} vs {sylls}")
    return units


def _raw_stanzas(meter: str) -> list[list[str]]:
    txt = (RAW / f"raw_{meter}.txt").read_text(encoding="utf-8").strip()
    return [[w for w in re.split(r"\s+", s.strip()) if w]
            for s in re.split(r"\n\s*\n", txt) if s.strip()]


def _khlong_display_order(n_lines: int, raw_lines: list[str]) -> list[int]:
    """Grader order is 8 core วรรค + คำสร้อย at the end; the display (traditional)
    order puts a บาท-3 คำสร้อย right after วรรค 6 (index 5)."""
    return [0, 1, 2, 3, 4, 5, 8, 6, 7] if n_lines == 9 else list(range(n_lines))


def load_student() -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for meter, mid in (("khlong", "khlong_si_suphap"), ("klon", "klon_paet"),
                       ("yani", "kap_yani_11"), ("chan", "intharawichian_chan_11")):
        gold = json.loads((SRC / f"student_{meter}_gold_original.json").read_text(encoding="utf-8"))
        raws = _raw_stanzas(meter)
        assert len(raws) == len(gold), (meter, len(raws), len(gold))
        for k, (p, rw) in enumerate(zip(gold, raws), 1):
            lines = [list(v) for v in p["lines"]]
            if meter == "khlong":
                for (sn, li), toks in KHLONG_FAITHFUL.items():
                    if sn == k:
                        lines[li] = toks
                # raw order: v0 v1 | v2 v3 | v4 v5 soi | v6 v7  → grader order
                order = _khlong_display_order(len(lines), rw)
                surface = [None] * len(lines)
                for disp_pos, gidx in enumerate(order):
                    surface[gidx] = rw[disp_pos]
            else:
                order = list(range(len(lines)))
                surface = rw
            rec = {"id": f"stu-{SLUG[mid]}-{k:02d}", "meter_id": mid,
                   "provenance": "student", "topic": p["source"].split(" (")[0],
                   "source_label": p["source"], "lines": lines, "surface": surface,
                   "display_order": order}
            r = score_meter([tuple(v) for v in lines], mid)
            rec["gold_score"] = r["score"]
            if r["score"] < 0.9999:
                fails = []
                for lname, L in r["layers"].items():
                    for d in L["details"]:
                        if not d[-1]:
                            fails.append(f"{lname}:{d[:-1]}")
                rec["fail_reasons"] = fails
                dropped.append(rec)
            else:
                kept.append(rec)
    return kept, dropped


def load_model() -> tuple[list[dict], list[dict]]:
    data = json.loads((SRC / "model_composed_all_composed.json").read_text(encoding="utf-8"))
    kept, dropped, ctr = [], [], {}
    for p in data:
        if not p["provenance"].startswith("Opus"):
            continue                      # web-found published poems: excluded
        mid, src = p["meter_id"], p["source"]
        key = next((k for k in MODEL_EXCLUDE if src.startswith(k)), None)
        ctr[mid] = ctr.get(mid, 0) + 1
        rid = f"mdl-{SLUG[mid]}-{ctr[mid]:02d}"
        lines = [list(v) for v in p["lines"]]
        for (s, li, tok), rep in MODEL_RESEG.items():
            if src.startswith(s):
                i = lines[li].index(tok)
                lines[li][i:i + 1] = rep
        skey = next((k for k in MODEL_SURFACE if src.startswith(k)), None)
        surface = MODEL_SURFACE[skey] if skey else ["".join(v) for v in lines]
        rec = {"id": rid, "meter_id": mid, "provenance": "model",
               "topic": src.split(" (")[0].split(" / ")[0].strip(), "source_label": src,
               "lines": lines, "surface": surface, "display_order": list(range(len(lines)))}
        rec["gold_score"] = score_meter([tuple(v) for v in lines], mid)["score"]
        if key:
            rec["fail_reasons"] = [MODEL_EXCLUDE[key]]
            dropped.append(rec)
        elif rec["gold_score"] < 0.9999:
            rec["fail_reasons"] = ["gold segmentation < 1.0"]
            dropped.append(rec)
        else:
            kept.append(rec)
    return kept, dropped


def main() -> None:
    stu, stu_drop = load_student()
    mdl, mdl_drop = load_model()
    pool = []
    for rec in stu + mdl:
        rec["units"] = [align_units(s, v) for s, v in zip(rec["surface"], rec["lines"])]
        pool.append(rec)
    # main pool: per-form TARGET, student first
    count: dict[str, int] = {}
    for rec in pool:                      # student records come first
        mid = rec["meter_id"]
        rec["in_main"] = count.get(mid, 0) < TARGET[mid]
        if rec["in_main"]:
            count[mid] = count.get(mid, 0) + 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = ["# Source pool", "",
            "| form | student kept / total | model kept / total | main pool (student + model) | control (model) |",
            "|---|--:|--:|--:|--:|"]
    for mid in TARGET:
        sk = sum(1 for r in stu if r["meter_id"] == mid)
        st = sk + sum(1 for r in stu_drop if r["meter_id"] == mid)
        mk = sum(1 for r in mdl if r["meter_id"] == mid)
        mt = mk + sum(1 for r in mdl_drop if r["meter_id"] == mid)
        ms = sum(1 for r in pool if r["meter_id"] == mid and r["in_main"] and r["provenance"] == "student")
        mm = sum(1 for r in pool if r["meter_id"] == mid and r["in_main"] and r["provenance"] == "model")
        cc = sum(1 for r in pool if r["meter_id"] == mid and not r["in_main"])
        rows.append(f"| {mid} | {sk} / {st} | {mk} / {mt} | {ms} + {mm} | {cc} |")
    rows += ["", "## Excluded stanzas", ""]
    for r in stu_drop + mdl_drop:
        rows.append(f"- `{r['id']}` ({r['source_label']}, gold score {r['gold_score']}): "
                    + "; ".join(r["fail_reasons"][:6])
                    + (f" … (+{len(r['fail_reasons']) - 6} more)" if len(r["fail_reasons"]) > 6 else ""))
    rows += ["", "## Unit alignments with respelled syllables (surface ← gold)", ""]
    for rec in pool:
        for li, us in enumerate(rec["units"]):
            for u in us:
                if u["syl"] != [u["s"]]:
                    rows.append(f"- `{rec['id']}` วรรค{li}: {u['s']} ← {'.'.join(u['syl'])}")
    REPORT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows[:8]))
    print(f"\nkept {len(pool)} stanzas → {OUT}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
