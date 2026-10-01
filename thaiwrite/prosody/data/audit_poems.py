"""Gold-segmented canonical poems for the khavee capability audit.

Each `Poem.lines` is the hand-segmented **pronounced-syllable** (พยางค์) split
per วรรค, ordered to match `meter_specs.METER_SPECS[meter_id].lines`. The audit
derives *expected* tone / weight / rhyme from `meter_specs` and compares khavee's
output against the gold syllable actually sitting at each constrained position.

Segmentation notes
------------------
- Constrained positions (โคลง เอก/โท, ฉันท์ ครุ/ลหุ, all rhyme endpoints) sit on
  clean monosyllabic written words, so khavee receives the exact word.
- อักษรนำ / polysyllabic written words are split into pronounced syllables so
  `len(line)` equals the prosodic count (e.g. อร่อย → อะ·ร่อย, มนุษย์ → มะ·นุด,
  กรุณา → กะ·รุ·ณา). These never coincide with a constrained position here.
- Verified against the royal สารานุกรมไทยฯ (see docs/meter-specs.md, section 5b).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Poem:
    meter_id: str
    source: str
    lines: tuple[tuple[str, ...], ...]
    note: str = ""
    eval_rhyme: bool = True   # False for synthetic วรรค sets (no real บท rhyme)


POEMS: tuple[Poem, ...] = (
    # --- โคลงสี่สุภาพ: ลิลิตพระลอ (all 11 tone positions + 3 rhymes exercised) ---
    Poem(
        meter_id="khlong_si_suphap",
        source="ลิลิตพระลอ — เสียงลือเสียงเล่าอ้าง",
        lines=(
            ("เสียง", "ลือ", "เสียง", "เล่า", "อ้าง"),   # 0 บาท๑ หน้า
            ("อัน", "ใด"),                                # 1 บาท๑ หลัง
            ("เสียง", "ย่อม", "ยอ", "ยศ", "ใคร"),        # 2 บาท๒ หน้า
            ("ทั่ว", "หล้า"),                             # 3 บาท๒ หลัง
            ("สอง", "เขือ", "พี่", "หลับ", "ใหล"),       # 4 บาท๓ หน้า
            ("ลืม", "ตื่น"),                              # 5 บาท๓ หลัง
            ("สอง", "พี่", "คิด", "เอง", "อ้า"),         # 6 บาท๔ หน้า
            ("อย่า", "ได้", "ถาม", "เผือ"),              # 7 บาท๔ หลัง
            ("พี่", "เอย"),                               # 8 คำสร้อย บาท๑
            ("ฤๅ", "พี่"),                                # 9 คำสร้อย บาท๓
        ),
    ),
    # --- กลอนแปด: นิราศภูเขาทอง ---
    Poem(
        meter_id="klon_paet",
        source="นิราศภูเขาทอง — ถึงบางพูดพูดดีเป็นศรีศักดิ์",
        lines=(
            ("ถึง", "บาง", "พูด", "พูด", "ดี", "เป็น", "ศรี", "ศักดิ์"),       # สดับ 8
            ("มี", "คน", "รัก", "รส", "ถ้อย", "อะ", "ร่อย", "จิต"),          # รับ 8 (อร่อย→อะ·ร่อย)
            ("แม้น", "พูด", "ชั่ว", "ตัว", "ตาย", "ทำ", "ลาย", "มิตร"),       # รอง 8
            ("จะ", "ชอบ", "ผิด", "ใน", "มะ", "นุด", "เพราะ", "พูด", "จา"),   # ส่ง 9 (มนุษย์→มะ·นุด)
        ),
        note="คำท้ายสดับ ศักดิ์ → รับ คำ๓ รัก; รับท้าย จิต → รองท้าย มิตร; รองท้าย มิตร → ส่ง คำ๓ ผิด.",
    ),
    # --- กลอนแปด: พระอภัยมณี ---
    Poem(
        meter_id="klon_paet",
        source="พระอภัยมณี — แล้วสอนว่าอย่าไว้ใจมนุษย์",
        lines=(
            ("แล้ว", "สอน", "ว่า", "อย่า", "ไว้", "ใจ", "มะ", "นุด"),        # สดับ 8 (มนุษย์→มะ·นุด)
            ("มัน", "แสน", "สุด", "ลึก", "ล้ำ", "เหลือ", "กำ", "หนด"),       # รับ 8 (กำหนด→กำ·หนด)
            ("ถึง", "เถา", "วัล", "พัน", "เกี่ยว", "ที่", "เลี้ยว", "ลด"),     # รอง 8 (เถาวัลย์→เถา·วัล)
            ("ก็", "ไม่", "คด", "เหมือน", "หนึ่ง", "ใน", "น้ำ", "ใจ", "คน"),  # ส่ง 9
        ),
        note="สดับท้าย นุด → รับ คำ๓ สุด; รับท้าย หนด → รองท้าย ลด; รองท้าย ลด → ส่ง คำ๓ คด.",
    ),
    # --- กาพย์ยานี ๑๑: กาพย์พระไชยสุริยา (clean, all monosyllabic) ---
    Poem(
        meter_id="kap_yani_11",
        source="กาพย์พระไชยสุริยา — ขึ้นกกตกทุกข์ยาก",
        lines=(
            ("ขึ้น", "กก", "ตก", "ทุกข์", "ยาก"),               # วรรค๑ 5
            ("แสน", "ลำ", "บาก", "จาก", "เวียง", "ชัย"),        # วรรค๒ 6
            ("มัน", "เผือก", "เลือก", "เผา", "ไฟ"),             # วรรค๓ 5
            ("กิน", "ผล", "ไม้", "ได้", "เป็น", "แรง"),         # วรรค๔ 6
        ),
        note="วรรค๑ท้าย ยาก → วรรค๒ คำ๓ บาก; วรรค๒ท้าย ชัย → วรรค๓ท้าย ไฟ.",
    ),
    # --- อินทรวิเชียรฉันท์ ๑๑: scan-verified วรรค (NOT one canonical บท) ---
    # Used to test check_karu_lahu against the ครุ/ลหุ pattern + tokenization.
    # Each วรรค independently scans to the meter; rhyme NOT evaluated (eval_rhyme=False).
    Poem(
        meter_id="intharawichian_chan_11",
        source="ฉันท์ทดสอบ (แต่งขึ้นทดสอบการสแกน ครุ/ลหุ) — รำลึกพระคุณแม่",
        lines=(
            ("รำ", "ลึก", "พระ", "คุณ", "แม่"),                 # วรรคหน้า: ค ค ล ค ค
            ("กะ", "รุ", "ณา", "และ", "อา", "ทร"),             # วรรคหลัง: ล ล ค ล ค ค (กรุณา·และ·อาทร)
            ("สา", "ยัณห์", "ตะ", "วัน", "ยาม"),               # วรรคหน้า: ค ค ล ค ค
            ("ขะ", "ณะ", "ข้าม", "ทิ", "คัม", "พร"),           # วรรคหลัง: ล ล ค ล ค ค (ขณะ·ข้าม·ทิฆัมพร)
        ),
        note="วรรคแยกที่สแกนถูกต้องตามแต่ละ ครุ/ลหุ; ไม่ใช่บทเดียวกัน จึงไม่ตรวจสัมผัส.",
        eval_rhyme=False,
    ),
)


# Clean labeled syllables with unambiguous ครุ/ลหุ, for a tokenizer-independent
# check_karu_lahu accuracy signal.
KARU_LAHU_PROBES: dict[str, tuple[str, ...]] = {
    # ครุ: long vowel, or any closed syllable, or สระ อำ ใอ ไอ เอา
    "karu": ("รำ", "ลึก", "คุณ", "แม่", "สา", "วัน", "ยาม", "ข้าม",
             "มา", "ไป", "ใจ", "ดาว", "กก", "จิต"),
    # ลหุ: short open vowel, no final
    "lahu": ("พระ", "ตะ", "ขะ", "ณะ", "ทิ", "และ", "จะ", "มิ", "ธุ", "ระ"),
}
