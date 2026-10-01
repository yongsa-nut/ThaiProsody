# Meter Formal Specifications

Formal, cited specifications for the four classical Thai meters covered by the
prosody benchmark. This is the **ground-truth definition of meter correctness**
for the rule-based grading layer; the grader (`thaiwrite/prosody/grader_meter.py`) is
built against the machine-readable twin of this document,
`thaiwrite/prosody/meter_specs.py`. This `.md` is the human-facing spec and the `.py`
is authoritative for code.

**Verification.** All four meters are cross-checked against royal-endorsed primary
sources (สารานุกรมไทยฯ + ราชบัณฑิตยสภา, §6); โคลง's 11 tone positions match exactly.
โคลงสี่สุภาพ is verified word-by-word against the ลิลิตพระลอ exemplar; the other three
are hand-counted against canonical poems (§5). กลอนแปด rhyme is encoded as the discrete
set {คำ๓, คำ๕}.

---

## 0. Conventions

**Coordinates.** Each poem is flattened to a list of **lines**, one line per
วรรค (the smallest sequential syllable segment). บาท / บท are kept as metadata,
not used for addressing. A syllable is `(line_index, syllable_index)`, both
**0-based**. A **negative** `syllable_index` counts from the end (`-1` = last),
so rules survive within-tolerance count drift (matters for กลอนแปด's 7–9 window).

**Tone marks.** "เอก" = a syllable bearing ไม้เอก (อ่); "โท" = ไม้โท (อ้), per the
รูปวรรณยุกต์ (written-mark) convention of classical โคลง, graded via
`khavee.check_aek_too`.

**Rhyme (สัมผัส).** A link's source syllable must rhyme (สัมผัสสระ) with **at
least one** syllable in the target range; a single-position target is a
length-1 range. External (cross-บท) links reference the **next stanza's** line
block.

**ครุ/ลหุ.** ครุ = heavy syllable (long vowel, or any syllable closed by a final
/ ending in สระ อำ ไ ใ เา); ลหุ = light (short open vowel). Used only by ฉันท์.

> **Two rule values that sources render differently.**
> 1. **อินทรวิเชียรฉันท์ ๑๑**: the canonical weight pattern is `ค ค ล ค ค | ล ล ค ล ค ค`
>    on the 5+6 วรรค split (§4). A 7+4 rendering such as `ค-ล-ค-ค-ล-ล-ค | ค-ล-ค-ค` is
>    non-canonical.
> 2. **กาพย์ยานี ๑๑**: the bound rhymes are วรรค๑(ท้าย)→วรรค๒ คำ๓ and
>    วรรค๒(ท้าย)→วรรค๓(ท้าย) (§3), tighter than the loose gloss "last syllable of
>    บาทเอก → syllables 1–3 of บาทโท".
>
> Both values are verified against the royal-endorsed สารานุกรมไทยฯ (§6).

---

## 1. โคลงสี่สุภาพ  (`khlong_si_suphap`)

**Structure.** 1 บท = 4 บาท. Each บาท = วรรคหน้า (5 คำ) + วรรคหลัง. วรรคหลัง = 2 คำ
for บาท ๑–๓, **4 คำ** for บาท ๔. Total 30 คำ. Optional **คำสร้อย** (2 คำ) may close
บาท ๑ and บาท ๓. → 8 mandatory lines (+2 optional สร้อย lines).

**Tone marks: เอก ๗ แห่ง, โท ๔ แห่ง** (positions as บาท·คำ, and the 0-based
`(line, syll)` used in code):

| Rule | บาท·คำ | (line, syll) | | Rule | บาท·คำ | (line, syll) |
|---|---|---|---|---|---|---|
| เอก | ๑·๔ | (0, 3) | | โท | ๑·๕ | (0, 4) |
| เอก | ๒·๒ | (2, 1) | | โท | ๒·๗ | (3, 1) |
| เอก | ๒·๖ | (3, 0) | | โท | ๔·๕ | (6, 4) |
| เอก | ๓·๓ | (4, 2) | | โท | ๔·๗ | (7, 1) |
| เอก | ๓·๗ | (5, 1) | | | | |
| เอก | ๔·๒ | (6, 1) | | | | |
| เอก | ๔·๖ | (7, 0) | | | | |

**Rhyme (สัมผัสบังคับ).**
- คำสุดท้ายบาท ๑ (คำ๗) → คำที่ ๕ ของบาท ๒: `(1,1) → (2,4)`
- คำสุดท้ายบาท ๑ (คำ๗) → คำที่ ๕ ของบาท ๓: `(1,1) → (4,4)`
- คำสุดท้ายบาท ๒ (คำ๗) → คำที่ ๕ ของบาท ๔: `(3,1) → (6,4)`

**คำสร้อย.** Optional 2-word add-on at the end of บาท ๑ and บาท ๓; used only when
the sense is incomplete. Encoded as `optional` lines appended after the
mandatory 8 so they never renumber the addressed positions. No rule references
คำสร้อย, so its absence is not penalized.

**Source.** th.wikipedia.org/wiki/โคลงสี่สุภาพ (diagram template), verified
word-by-word against the ลิลิตพระลอ exemplar; สัมผัส cross-checked with
digitalschool.club (thai2_12). **Primary-source verified:** the royal
สารานุกรมไทยฯ (oldchild/1424) lists **all 11 tone positions identical** to the
above; ราชบัณฑิตยสภา (orst.go.th) corroborates เอก·บาท๑คำ๔ / บาท๒คำ๒ and the
บาท๑→คำ๕·บาท๒/๓ rhymes.

---

## 2. กลอนแปด / กลอนสุภาพ  (`klon_paet`)

**Structure.** 1 บท = 4 วรรค: **สดับ / รับ / รอง / ส่ง**. ~8 พยางค์ per วรรค,
tolerance **7–9**. No required เอก/โท positions are graded (end-of-วรรค tone
*preferences* exist but are advisory; see note). → 4 lines.

**Rhyme (สัมผัสนอก).**
- สดับ(ท้าย) → รับ **คำ ๓ หรือ คำ ๕**: `(0,-1) → {(1,2), (1,4)}`
- รับ(ท้าย) → รอง(ท้าย): `(1,-1) → (2,-1)`
- รอง(ท้าย) → ส่ง **คำ ๓ หรือ คำ ๕**: `(2,-1) → {(3,2), (3,4)}`
- **ระหว่างบท:** ส่ง(ท้าย) → รับ(ท้าย) ของบทถัดไป: `(3,-1) → next.(1,-1)` *(external, optional)*

*Encoded as the discrete set {คำ๓, คำ๕} (two length-1 target ranges), per the
royal สารานุกรมไทยฯ wording "คำที่ ๓ **หรือ** คำที่ ๕", **not** the contiguous
range ๓–๕; คำ ๔ is not a valid receiver. In code, `RhymeLink.targets` is a tuple of
position ranges, which supports such disjoint receivers.*

**Note (advisory, not graded).** เสียงวรรณยุกต์ท้ายวรรค:
วรรครับ avoids สามัญ (prefers เอก/โท/จัตวา); วรรคส่ง prefers สามัญ/ตรี. Recorded
for reference; **not** enforced by the grader.

**Source.** dhanitarampaigul.wordpress.com (ฉันทลักษณ์ + เสียงวรรณยุกต์);
khuntong52.blogspot.com (สัมผัสกลอนแปด). **Primary-source verified:** the royal
สารานุกรมไทยฯ (oldchild/1423) states the rhyme verbatim as "คำสุดท้ายของวรรคสลับ
สัมผัสกับคำที่ ๓ **หรือ** คำที่ ๕ ของวรรครับ" (→ discrete encoding).

---

## 3. กาพย์ยานี ๑๑  (`kap_yani_11`)

**Structure.** 1 บท = 2 บาท. Each บาท = วรรคหน้า (5) + วรรคหลัง (6) = 11 พยางค์.
→ 4 lines: `[5, 6, 5, 6]`. No เอก/โท, no ครุ/ลหุ.

**Rhyme (สัมผัสบังคับ).**
- คำสุดท้ายวรรค ๑ (คำ๕) → คำที่ ๓ ของวรรค ๒: `(0,4) → (1,2)`
- คำสุดท้ายวรรค ๒ (คำ๖) → คำสุดท้ายวรรค ๓ (คำ๕): `(1,5) → (2,4)`
- **ระหว่างบท:** คำสุดท้ายวรรค ๔ → คำสุดท้ายวรรค ๒ ของบทถัดไป: `(3,-1) → next.(1,-1)` *(external, optional)*

**Source.** th.wikipedia.org/wiki/กาพย์ยานี; trueplookpanya.com/learning/detail/32162.
**Primary-source verified:** the royal สารานุกรมไทยฯ (oldchild/1422) confirms
5+6 structure and the two internal rhymes (วรรค๑ท้าย→วรรค๒คำ๓; วรรค๒ท้าย→วรรค๓ท้าย)
plus the between-บท link.

---

## 4. อินทรวิเชียรฉันท์ ๑๑  (`intharawichian_chan_11`)

**Structure.** Identical line layout + rhyme to กาพย์ยานี ๑๑ (`[5, 6, 5, 6]`),
**plus** a fixed ครุ/ลหุ pattern.

**ครุ/ลหุ pattern** (ค = ครุ, ล = ลหุ):

| วรรค | คำ ๑ | ๒ | ๓ | ๔ | ๕ | ๖ |
|---|---|---|---|---|---|---|
| วรรคหน้า (5) | ค | ค | ล | ค | ค | n/a |
| วรรคหลัง (6) | ล | ล | ค | ล | ค | ค |

(วรรคหน้า rule: คำ ๑,๒,๔,๕ = ครุ, คำ ๓ = ลหุ. วรรคหลัง rule: คำ ๑,๒,๔ = ลหุ,
คำ ๓,๕,๖ = ครุ. 14 ครุ + 8 ลหุ per บท.)

**Rhyme.** Same three links as กาพย์ยานี ๑๑ (§3).

**Source.** phkruthai.in.th (4.3 อินทรวิเชียรฉันท์ ๑๑); krupiyarerk.wordpress.com.
Pattern verified against **two** exemplars (§5b). **Primary-source caveat:** the
royal สารานุกรมไทยฯ ฉันท์ entry (oldchild/1425) exists but its diagram did not
auto-extract reliably (returned an implausible strict-alternating `ค-ล-ค-ล-ค`
that fails real lines, e.g. "ลึก", a closed syllable, must be ครุ not ลหุ). The
encoded pattern `ค ค ล ค ค | ล ล ค ล ค ค` is the canonical form and is
hand-confirmed against both exemplars below.

---

## 5. Validation

### 5a. Self-consistency check  (`python thaiwrite/prosody/meter_specs.py`)

```
meter                      lines  syll  tone  rhyme   wt
โคลงสี่สุภาพ                  10    30    11      3    0
กลอนแปด                        4    32     0      4    0
กาพย์ยานี ๑๑                   4    22     0      3    0
อินทรวิเชียรฉันท์ ๑๑           4    22     0      3    4

check_specs: OK, all 4 meters internally consistent.
```

All assertions pass: every tone/rhyme/weight position resolves within its line's
syllable window; every ครุ/ลหุ pattern length equals its line's count; meter_ids
unique; `to_dict()` round-trips through `json.dumps(ensure_ascii=False)`.

### 5b. Hand-count against canonical poems

| Meter | Exemplar | Result |
|---|---|---|
| โคลงสี่สุภาพ | ลิลิตพระลอ: *เสียงลือเสียงเล่าอ้าง…* | **PASS.** Syllables 5+2 / 5+2 / 5+2 / 5+4. All 7 เอก (เล่า, ย่อม, ทั่ว, พี่, ตื่น, พี่, อย่า) and 4 โท (อ้าง, หล้า, อ้า, ได้) land on the spec positions. Rhymes ใด→ใคร→ใหล and หล้า→อ้า hold. |
| กลอนแปด | นิราศภูเขาทอง *…ศรีศักดิ์…* + พระอภัยมณี + สุภาษิตสุนทรภู่ (3 stanzas) | **PASS.** Syllables 8/8/8/9 (within 7–9). The สัมผัสรับ lands on **คำ๓** in every sampled stanza (ศักดิ์→รัก, นุด→สุด, ปาก→ยาก); none on คำ๔, supporting the discrete {๓,๕} encoding. |
| กาพย์ยานี ๑๑ | *เรื่อยเรื่อยมารอนรอน / ทิพากรจะตกต่ำ…* | **PASS.** Syllables 5/6/5/6. รอน→กร (วรรค๒ คำ๓), ต่ำ→ค่ำ (วรรค๓ ท้าย) hold. |
| อินทรวิเชียรฉันท์ ๑๑ | (1) *รำลึกพระคุณแม่ … กรุณาและอาทร*; (2) *สายัณห์ตะวันยาม / ขณะข้ามทิฆัมพร* | **PASS ×2.** Both: วรรคหน้า = ค-ค-ล-ค-ค (รำลึกพระคุณแม่ / สายัณห์ตะวันยาม), วรรคหลัง = ล-ล-ค-ล-ค-ค (กรุณาและอาทร / ขณะข้ามทิฆัมพร); both match the encoded pattern exactly. |

---

## 6. Sources

All accessed **2026-05-23**.

**Primary / authoritative sources** (royal-endorsed encyclopedia + ราชบัณฑิตยสภา),
used for verification:

| Meter | Source | URL | Verified |
|---|---|---|---|
| โคลงสี่สุภาพ | สารานุกรมไทยสำหรับเยาวชนฯ | saranukromthai.or.th/oldchild/1424 | all 11 tone positions exact |
| โคลงสี่สุภาพ | ราชบัณฑิตยสภา | orst.go.th (DRAWER004/…/00004229.FLP) | corroborates เอก + rhyme |
| กลอนแปด | สารานุกรมไทยสำหรับเยาวชนฯ | saranukromthai.or.th/oldchild/1423 | "คำ๓ หรือ ๕" rhyme |
| กาพย์ยานี ๑๑ | สารานุกรมไทยสำหรับเยาวชนฯ | saranukromthai.or.th/oldchild/1422 | 5+6 + rhymes |
| อินทรวิเชียรฉันท์ ๑๑ | สารานุกรมไทยสำหรับเยาวชนฯ | saranukromthai.or.th/oldchild/1425 | diagram auto-extract unreliable; pattern hand-verified vs 2 exemplars |

**Secondary / teaching sources** (initial rule extraction):

| # | Meter | Source | URL |
|---|---|---|---|
| 1 | โคลงสี่สุภาพ | วิกิพีเดีย: โคลงสี่สุภาพ (diagram template, verified vs ลิลิตพระลอ) | th.wikipedia.org/wiki/โคลงสี่สุภาพ |
| 2 | โคลงสี่สุภาพ (สัมผัส) | Digital School Thailand: ภาษาไทย thai2_12 | digitalschool.club/digitalschool/thai2_4_1/thai2_12/page2.php |
| 3 | โคลงสี่สุภาพ | TruePlookpanya: โคลงสี่สุภาพ | trueplookpanya.com/learning/detail/310 |
| 4 | กลอนแปด | ฉันทลักษณ์กลอนแปดสุภาพ | dhanitarampaigul.wordpress.com/2009/07/14/ |
| 5 | กลอนแปด (สัมผัส) | สัมผัสกลอนแปด | khuntong52.blogspot.com/2017/05/blog-post_7.html |
| 6 | กลอนแปด | สารานุกรมไทยสำหรับเยาวชน เล่ม ๑๔๒๓ | saranukromthai.or.th/oldchild/1423 |
| 7 | กาพย์ยานี ๑๑ | วิกิพีเดีย: กาพย์ยานี | th.wikipedia.org/wiki/กาพย์ยานี |
| 8 | กาพย์ยานี ๑๑ | TruePlookpanya: ข้อบังคับในการแต่งกาพย์ยานี ๑๑ | trueplookpanya.com/learning/detail/32162 |
| 9 | อินทรวิเชียรฉันท์ ๑๑ | ครูภาษาไทย: 4.3 อินทรวิเชียรฉันท์ ๑๑ | phkruthai.in.th/.../4-3-อินทรวเชยรฉนท-11 |
| 10 | อินทรวิเชียรฉันท์ ๑๑ | ครูปิยะฤกษ์ บุญโกศล | krupiyarerk.wordpress.com/2011/10/02/อินทรวิเชียรฉันท์-๑๑/ |

**Methodological precedents** (not meter-rule sources): CCiV (2025, two-layer rule+judge Chinese Ci benchmark); Yi et al.
(EMNLP 2018, Chinese 4-dim rubric, κ=0.67).
