# Khavee Capability Audit

**pythainlp:** 5.3.1 · **Harness:** `thaiwrite/prosody/audit_khavee.py` ·
**Corpus:** `thaiwrite/prosody/data/audit_poems.py` (6 poems / 26 วรรค).

This audit measures what `pythainlp.khavee` can and cannot verify at meter scale, and
motivates which checks the meter grader (`grader_meter.py`) takes from khavee and which it
implements itself. Verifier functions are tested on hand-segmented **gold** syllables (so a
tokenizer error cannot mask a verifier result); tokenization is audited separately. The
grader's final tone and weight handling (a dead-syllable fallback for tone and two weight
overrides) is described in `methods.md` §5.2.

---

## 1. Capability matrix

| khavee surface | Tested against | Result | Verdict |
|---|---|---|---|
| `check_aek_too` | 11/11 โคลง เอก+โท positions (plain **and** `dead_syllable_as_aek`) | **11/11** | **Reliable**: delegate the โคลง tone layer |
| `check_karu_lahu` | 22/22 ฉันท์ pattern positions + 24/24 clean labeled probes | **46/46** | **Reliable**: delegate the ฉันท์ ครุ/ลหุ layer |
| `is_sumpus` | 11 canonical rhyme links | **8/11** | **Partial**: fails on อักษรนำ / การันต์ words |
| `check_klon` (built-in กลอน validator) | 2 canonical กลอนแปด บท | both flagged false errors | **Not used**: mis-tokenizes and inherits the `is_sumpus` gaps |
| `syllable_tokenize(engine="dict")` | 26 gold วรรค | **24/26** count (92%) | Primary, but over-splits some words |
| `syllable_tokenize(engine="han_solo")` | 26 gold วรรค | **23/26** count (88%) | Cross-check; merges adjacent syllables |

**Bottom line:** the two **verification primitives the grader most needs are solid**:
tone-mark classification (โคลง) and ครุ/ลหุ (ฉันท์) are 100%. The **rhyme layer is
not delegable** as-is, and syllable tokenization is ~90% on classical verse.
**No meter fails badly enough to be dropped.**

---

## 2. What khavee verifies reliably

- **`check_aek_too`, โคลง tone marks: 11/11.** Every เอก (7) and โท (4) position in
  ลิลิตพระลอ classified correctly; `dead_syllable_as_aek=True` also 11/11, so the
  dead-syllable-as-เอก rule is handled. Wrapped directly in the grader.
- **`check_karu_lahu`, ฉันท์ ครุ/ลหุ: 46/46.** All 22 อินทรวิเชียรฉันท์ pattern
  positions correct, including tricky sub-syllables (กะ, รุ, ทร, ยัณห์, ขะ, ณะ,
  คัม, พร), plus 24/24 on a clean labeled probe set. **Must be fed one syllable
  at a time** (returns a single value per call). Wrapped directly.

## 3. What khavee does not handle

### 3a. `is_sumpus`: 8/11, fails on อักษรนำ / การันต์
The 3 misses share one root cause: **a leading silent consonant or complex
final defeats khavee's rime extraction**, while plain syllables rhyme fine:

| Miss | Why | Evidence |
|---|---|---|
| `ศักดิ์` ~ `รัก` | การันต์: real final is ก (/sàk/) but written ศ-ัก-ด-ิ์ | `is_sumpus('สัก','รัก')→True`; `('ศักดิ์','รัก')→False`; `handle_karun_sound_silence('ศักดิ์')→'ศักด'` (still wrong) |
| `หนด` ~ `ลด` | อักษรนำ หน (silent ห, /nòt/) | `is_sumpus('นด','ลด')→True`; `('หนด','ลด')→False` |
| `ใด` ~ `ใหล` | อักษรนำ ห in ใหล (/laj/) | `is_sumpus('ใด','ใคร')→True`; `('ใด','ใหล')→False`, `('ใด','ไหล')→False` |

`handle_karun_sound_silence` is **insufficient** (ศักดิ์→ศักด, not สัก). The grader
therefore compares the **true phonetic rime** (initial-cluster-stripped vowel+final by
sound, via `pythainlp` romanization) rather than relying on `is_sumpus` over raw
orthography: `is_sumpus` is the fast path, with a **fallback to a phonetic-rime
comparison on any miss** (`grader_meter.py`).

### 3b. `check_klon`: not used
On the two canonical กลอนแปด บท it raised *false* errors:
- *นิราศ:* `"Can't find rhyme ('ศักดิ์', ['คน','รัก','รส','ถ้อย'])"`: the valid
  ศักดิ์→รัก rhyme is missed (same การันต์ gap as 3a).
- *พระอภัย:* `"('ด', ['แสน','สุด',…])"`: it took **'ด'** as สดับ's last word, i.e.
  it **over-split มนุษย์** with its own internal tokenizer; plus `'หนด'~'ลด'` miss.

So `check_klon` both mis-tokenizes and inherits the `is_sumpus` gaps. The grader
implements its own กลอน rhyme check over `meter_specs.rhyme_links` instead.

### 3c. Syllable tokenization: ~90%, neither engine trustworthy alone
| Engine | Count acc. | Characteristic error |
|---|---|---|
| `dict` | 24/26 | **over-splits** มนุษย์ → มะ·นุ·ด (3, should be 2) |
| `han_solo` | 23/26 | **merges** adjacent syllables: ยอยศ, รุณา |

Errors are systematic (dict: การันต์/cluster over-split; han_solo: compound
merge). **`dict` is used as the primary engine** (better count, correctable errors),
with `han_solo` as a cross-check. Automatic syllable counts are treated as
**approximate**: the grader relies on the `meter_specs` count tolerances and, where the
model controls วรรค structure, on explicit line breaks rather than inferred ones. For the
reported count dimension, the tokenizer count is replaced by a model-based
count-of-record (`thaiwrite/prosody/count_api.py`; see `methods.md` §5.2).

---

## 4. Implications for the grader

| Layer | Source | Notes |
|---|---|---|
| Tone (เอก/โท) | **khavee** `check_aek_too` | reliable; use `dead_syllable_as_aek=True` |
| ครุ/ลหุ | **khavee** `check_karu_lahu` | reliable; one syllable per call |
| Rhyme (สัมผัส) | **ours** over `meter_specs` | `is_sumpus` fast-path + phonetic-rime fallback for อักษรนำ/การันต์ |
| Syllable count | **ours** (`dict` + post-rules) | counts approximate; spec tolerances; no hard fail on borderline counts; reported counts use the count-of-record (`count_api.py`) |
| Position mapping | **ours** (`meter_specs`) | khavee offers nothing here |
| `check_klon` | n/a | not used |
