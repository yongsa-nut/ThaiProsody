# Thai Prosody Benchmark: Grading Methodology

_How the benchmark is run and scored. This document consolidates the rule definitions
(`meter-specs.md`, `khavee-audit.md`), the annotation procedure (`annotation-protocol.md`)
and the evaluation harness (`thaiwrite/prosody/eval/`). Every table in the paper is
reproduced from the released files by `reproduce/make_paper_tables.py`, with no API calls._

---

## 0. Pipeline overview

```
prompt (system + task)  →  model  →  raw output
   → clean_poem        (strip reasoning / markdown / asides)
   → best-of-2          (two วรรค segmentations, keep the higher-scoring)
   → rule-based meter grader   (4 dimension checks → per-meter weighted score ∈ [0,1])
   → output-status classification   (graded | model_fail=0 | infra_error=excluded)
   → aggregation        (item-level bootstrap 95% CI; fail-scored headline)
```
Layer 2 (content judge) is reported as a **binary** topicality/coherence flag (a Claude Opus
judge, validated against expert labels) and is **excluded from the headline meter score**.
The meter score is the primary, deterministic signal.

---

## 1. Tasks and items

Three task modes share the same four classical meters:

| Mode | Prompt gives | Model must | Graded |
|---|---|---|---|
| **compose** | meter name + topic + #stanzas | write a new poem from scratch | full meter grade |
| **cloze** | a valid poem with `____` blanks | fill blanks, reproduce the complete poem | full meter grade |
| **edit** | a poem with an injected meter break | repair it, rewrite the whole poem | full meter grade |

**Composition** (the headline task): 160 items (80 public + 80 private), crossing the four
meters with 40 topics (`thaiwrite/prosody/data/prompts_compose_{public,private}.jsonl`).

**Cloze and edit** (`thaiwrite/prosody/cloze_edit/`): 178 cloze and 74 edit items
(`cloze_edit/data/items_{cloze,edit}.jsonl`), built from a pool of unpublished stanzas:
stanzas the second author wrote as secondary-school coursework and model-composed
stanzas. Every pool stanza scores 1.0 under the meter grader with its gold syllable
segmentation, and model-composed stanzas with invalid readings (respellings, non-word
fillers, miscounted syllables) are excluded; the pool composition and every exclusion are
listed in `cloze_edit/data/pool_report.md`. Items blank or break different hard positions of
a stanza, so several items come from one stanza. **Edit is โคลง + ฉันท์ only** (กลอน/ยานี bind
no tone or weight, so there is no meter rule to break), and an injection is kept only if it
actually lowers the rule-grader score. The pipeline is documented in `cloze_edit/README.md`.

---

## 2. Prompts

Zero-shot and minimal: no few-shot examples, no rule explanations, no syllable counts. This
tests whether the model *knows* the meter, not whether it can follow a handed-over spec.

**System (all modes):**
> คุณเป็นกวีผู้เชี่ยวชาญฉันทลักษณ์ไทย ตอบเป็นบทร้อยกรองเท่านั้น ไม่ต้องอธิบาย
> _(“You are an expert in Thai prosody. Respond with verse only, no explanation.”)_

**User (compose):**
> จงแต่ง{meter} จำนวน {n} บท ในหัวข้อ "{topic}" ให้ถูกต้องตามฉันทลักษณ์ เขียนแต่ละวรรคขึ้นบรรทัดใหม่

**User (cloze):**
> จงเติมคำที่หายไป (____) ในบทร้อยกรองต่อไปนี้ ให้ถูกต้องตามฉันทลักษณ์และความหมาย แล้วเขียนบทที่สมบูรณ์ แต่ละวรรคขึ้นบรรทัดใหม่:
> \n\n{poem_with_blanks}

**User (edit):**
> บทร้อยกรองต่อไปนี้มีข้อผิดพลาดด้านฉันทลักษณ์ จงแก้ไขให้ถูกต้อง แล้วเขียนบทที่แก้แล้วใหม่ทั้งบท แต่ละวรรคขึ้นบรรทัดใหม่:
> \n\n{input_poem}

The instruction **"เขียนแต่ละวรรคขึ้นบรรทัดใหม่"** ("write each วรรค on a new line") is the layout
contract the grader relies on; because models honour it unevenly, grading is robust to two
layouts via best-of-2 (§4). (Source: `thaiwrite/prosody/eval/promptlib.py`.)

---

## 3. Output normalization (`clean_poem`)

The system prompt asks for verse only, but models still decorate output, so the raw
response is normalized before grading. Stripped, in order:

1. **Reasoning blocks**: text inside `<think>…</think>` (keep what follows the last
   `</think>`; an *unclosed* `<think>` means the answer was never produced, so the output is empty).
2. **Code fences**: ` ``` ` lines.
3. **Markdown emphasis markers**: `** __ \` ~~` (removed inline).
4. **Label / header lines**: markdown `#` headers; horizontal rules (`--- *** ___`);
   stanza labels (`บทที่ N`, `บท/วรรค/บาท/ตอน/ท่อน` + number, `stanza/verse/part`); bare
   ordinals (`1.` `๑)` `(3)`).
5. **Colon-terminated lead-ins**, e.g. `Here is the poem:`, `ขอแก้ใหม่อีกครั้ง:`.
6. **Latin-only / no-Thai lines**: English asides, separators.

Blank lines are kept (display spacing; the grader ignores them). **Why this matters:** an
un-stripped *leading* line (e.g. a `**บทที่ 1**` header) becomes a phantom วรรค and shifts
every subsequent line off its spec slot, producing spurious count/rhyme verdicts; stripping
it is necessary for a faithful grade. (Source: `promptlib.clean_poem`.)

_Known limitation: a Thai prose annotation with an inline colon (e.g. `หมายเหตุ: …`) can
survive; it is positionally harmless when trailing (grading maps onto a fixed วรรค count, so
extra trailing lines are ignored), and rare under the no-explanation system prompt._

---

## 4. วรรค segmentation (best-of-2)

The grader maps poem lines positionally onto the meter spec, so it needs **one วรรค per
line**. Thai writes no spaces *within* a วรรค, so any space inside a line is a true วรรค
boundary. The cleaned poem is graded under **two** segmentations and the higher meter score
is kept:

- **raw**: split on newlines only (correct when the model writes one วรรค per line).
- **normalized** (`to_vrk_lines`): split on newlines **and** whitespace runs (correct for
  the traditional two-วรรค-per-line layout).

Because intra-วรรค spaces do not occur, normalized splitting can only *recover* the correct
division, never wrongly chop a real วรรค; keeping raw as well protects the rare case of a
stray intra-วรรค space. Taking the max means a valid poem is never penalized for which
conventional layout it used. (Source: `promptlib.to_vrk_lines`, `grade_runs.grade_record`.)

---

## 5. Rule-based meter grader (Layer 1)

### 5.1 The four meters

| Meter | วรรค (required + optional) | Syllables / วรรค | Bound dimensions |
|---|---|---|---|
| **โคลงสี่สุภาพ** | 8 required + 2 optional คำสร้อย | 5·2·5·2·5·2·5·4 (สร้อย 2·2) | count, **tone** (11 positions), rhyme (3 links) |
| **กลอนแปด** | 4 (one บท) | 7–9 each | count, rhyme (3 links) |
| **กาพย์ยานี ๑๑** | 4 | 5·6·5·6 | count, rhyme (2 links) |
| **อินทรวิเชียรฉันท์ ๑๑** | 4 | 5·6·5·6 | count, rhyme (2 links), **ครุ/ลหุ** (weight pattern ค-ค-ล-ค-ค \| ล-ล-ค-ล-ค-ค) |

Specs are verified against ราชบัณฑิตยสภา / สารานุกรมไทยฯ. (Source: `docs/meter-specs.md`,
`meter_specs.py`.)

### 5.2 The four dimension checks

Each applicable dimension yields a **pass rate ∈ [0,1]** = fraction of that dimension's
rules satisfied. Where possible the check delegates to `pythainlp.khavee`
(`KhaveeVerifier`), audited reliable (`docs/khavee-audit.md`):

- **count (นับคำ)**: no rule-based Thai syllabifier is reliable on classical verse. The
  dictionary tokenizer (`syllable_tokenize(engine="dict")`, used inside `grader_meter.py`)
  reaches **~84% per-line accuracy** (audited against Claude Opus 4.7 counts over all 20,064
  generated lines) and systematically *under*-counts implicit-vowel อักษรนำ / Sanskrit-Pali
  syllables. The reported count dimension therefore uses a **count-of-record**: syllable
  counts from a Claude Opus syllable counter (`count_api.py`, Claude Opus 5.5 through the
  Claude API, prompt in `vrk_counter.py`), cached in `vrk_count_cache_opus55.json` and
  `vrk_count_cache_opus55_bcd.json`. `reproduce/make_paper_tables.py` recomputes the count
  layer from these caches on the stored best-of-2 layout (a line's count is the sum of the
  cached counts of its whitespace-separated pieces). Count carries the lowest weight, and
  substituting the dictionary tokenizer leaves the ranking unchanged (see the paper's
  robustness appendix).
- **tone (เอก/โท)**: `check_aek_too` (audit 11/11 reliable), **plus an เอก-ตาย fallback**: a
  คำตาย (dead syllable, `check_marttra ∈ {กก, กด, กบ}`) satisfies a บังคับเอก position, covering
  polysyllabic / การันต์ / silent-ร words `check_aek_too` mishandles.
- **rhyme (สัมผัส)**: `is_sumpus` fast-path **plus fallbacks** for the words it misses:
  strip a silent leading ห นำ, compare the **romanized rime** (from the first vowel onward)
  for อักษรนำ / การันต์ words, and handle open ไ-/ใ- syllables, which the library misreads.
  Audit: `is_sumpus` alone 8/11, with fallbacks 14/14. Internal (within-บท) links only.
- **ครุ/ลหุ (kru/lahu)**: `check_karu_lahu` (audit 46/46 reliable), with two overrides
  where the library is wrong: an open syllable in long -อ is heavy, and the negation บ is
  light.

All khavee calls are wrapped (`_safe`) to **degrade to None instead of crashing** on
pathological tokens (e.g. a consonant-less วรรค in a reformatted poem); a single bad syllable
never fails the whole grade. (Source: `grader_meter.py`.)

### 5.3 Composite meter score

The meter score is a **weighted average of the applicable dimension pass rates**, weights
renormalized over the dimensions a meter actually binds:

| Meter | count | tone | rhyme | kru/lahu |
|---|--:|--:|--:|--:|
| โคลงสี่สุภาพ | 0.25 | 0.40 | 0.35 | n/a |
| กลอนแปด | 0.40 | n/a | 0.60 | n/a |
| กาพย์ยานี ๑๑ | 0.40 | n/a | 0.60 | n/a |
| อินทรวิเชียรฉันท์ ๑๑ | 0.20 | n/a | 0.30 | 0.50 |

Weights are **fixed a priori, not tuned** (count carries the least weight; the
meter-defining constraint, tone for โคลง and weight for ฉันท์, carries the most). Equal
weights reproduce the same ranking. Validation checks the sub-rule *correctness*, not the
weights. Each dimension also reports a **full-compliance** boolean (all its rules
satisfied); a **valid stanza** is one in which every bound check passes.
(Source: `grader_meter.LAYER_WEIGHTS`.)

---

## 6. Output-status classification and fail-scoring

Every graded record carries an `eval_status`, which determines how a non-delivered output
is scored, distinguishing the **model's** failure from an **infrastructure** failure:

| `eval_status` | Trigger | Scoring |
|---|---|---|
| **graded** | a poem was produced and scored | its meter score ∈ [0,1] |
| **model_fail** | empty output / token-runout (reasoning consumed the whole budget) | **0**, a real task failure |
| **infra_error** | exhausted API credit, server error, timeout, malformed body, or grader crash | **excluded**, not the model's fault |

The headline **fail-scored mean** is taken over `graded + model_fail` (model non-delivery
counts as 0); `infra_error` items are dropped from the denominator. A reasoning model that
burns its budget and emits nothing **fails** (0), but a model whose calls were rejected by
the serving side is **not** penalized. In the reported runs, infrastructure errors were
retried until the item completed, so every configuration is evaluated on all 160
composition items. `delivery_rate` = graded / (graded + model_fail).
(Source: `grade_runs.aggregate`, `_status`.)

---

## 7. Aggregation and confidence intervals

- **Primary metric:** per-model fail-scored mean meter score, with a **non-parametric
  item-level bootstrap 95% CI** (1000 resamples, seed 20260524).
- Also reported: valid-stanza rate, delivered-only mean, per-dimension compliance,
  per-meter mean, per-mode mean, delivery rate, and counts (`n_graded`, `n_model_fail`,
  `n_infra_excluded`).
- **Repeated runs.** A subset of configurations was run three times on the full composition
  set to measure run-to-run variation; the graded files are in `eval/graded/replication/`
  and the replication table is produced by `reproduce/make_paper_tables.py`.
  (Source: `grade_runs.py`.)

---

## 8. Dual-mode (thinking / no-think) protocol

Reasoning models are evaluated in **both** modes:

- **thinking = native**: the model's default. Token-runout (budget consumed before a poem
  is emitted) is scored as a **fail (0)**, and delivery rate is reported alongside.
- **no-think = ablation**: reasoning disabled (`reasoning.enabled=false` for OpenRouter
  models); written to separate `*.nothink` files so it never overwrites the native run.

This balances fairness (a thinking model is allowed to think) with honesty (runaway
reasoning that delivers nothing is a real failure). Empirically, disabling thinking restores
100% delivery for the runaway open reasoners. (Source: `run_models.py --no-think`.)

A second ablation places the meter rules in the prompt (`*.rules` files; rule sheets in
`eval/rulesheets.py`).

---

## 9. Content judge (Layer 2): binary, Claude Opus

A 2-axis pointwise judge (**topicality** + **coherence**) scores semantic quality,
separating *meter adherence* from *is-it-a-real-poem*. It is **excluded from the headline
meter score**. The judge (`judge_api.py`, Claude Opus 5.5 through the Claude API; rubric in
`grader_coherence_prompt.txt`) sees only the topic and the poem, blind to the model that wrote
it. Two design choices follow from a comparison of Claude Opus 4.7 and GPT-5.5 against
expert labels on 132 calibration poems (`calibration/labels/labels_expertA.json`); the same
rubric and binary cuts are used with Claude Opus 5.5, and `reproduce/make_paper_tables.py`
reports each judge's agreement with the expert:

- **Binary, not 1–5.** On the native 1–5 scale agreement is weak (coherence ordinal
  α≈0.29 for both GPT-5.5 and Claude Opus 4.7); the disagreement lies mainly between scores
  3, 4 and 5 of well-formed poems, a distinction the benchmark does not use. Scores are
  therefore binarized: topical = topicality ≥ 3; coherent = coherence ≥ 2 (not word salad).
- **Claude Opus, not GPT-5.5.** Holding the rubric fixed, GPT-5.5 (medium reasoning) never
  assigns the lowest coherence score: poems the expert rated 1 receive a mean of 3.5 from
  GPT-5.5 and 2.0 from Claude Opus 4.7, so GPT-5.5 cannot flag word-salad output. Binary
  Cohen's κ against the expert: topicality ≥ 3, Opus 4.7 **0.74** vs GPT-5.5 0.72;
  coherence ≥ 2, Opus 4.7 **0.74** vs GPT-5.5 0.00.

No blended L1/L2 scalar is reported (the meter score stays the precise signal; the content
judge is a binary degenerate/off-topic flag). (Source: `judge_api.py`, `grader_coherence.py`,
`calibration/judge_vs_expert.py`, `calibration/binary_agreement.py`.)

---

## 10. Calibration against expert labels

- **Two raters** labeled the full 176-item calibration set (`calibration/calibration_items_v2.json`)
  independently: 132 model poems rated 1–5 for topicality and coherence, and 44 word pairs
  judged as rhyming or not by sound. Rater 1's labels are in `calibration/labels/labels_expertA.json`,
  rater 2's (the second author) in `calibration/labels/labels_expertB.json`; the procedure is
  described in `docs/annotation-protocol.md`. The judge comparison in §9 uses rater 1.
- **Inter-rater agreement** (`calibration/rater_agreement.py`, `score_calibration.py`): the raters
  agree on 42/44 rhyme pairs (α = 0.88). They use the 1–5 scales differently (rater 2 scores
  lower throughout), so ordinal α is near zero; on the binary cuts they agree on 91% (topicality
  ≥ 3) and 95% (coherence ≥ 2) of poems, Cohen's κ 0.49 and 0.44; at a cut of 4 their agreement
  falls to 0.13 and 0.08.
- **Adjudicated labels** (`calibration/labels/labels_adjudicated.json`): where the raters fall on
  opposite sides of a benchmark cut (12 on-topic, 7 not-word-salad decisions), rater 1 re-examined
  the poem with both scores in view and set the label (siding with rater 2 in 8 of 19). Against
  these labels Claude Opus 5.5 reaches κ 0.64 (on-topic) and 0.56 (not word salad), Claude Opus 4.7
  0.78 and 0.66, GPT-5.5 0.64 and 0.00. Against the individual raters Opus 5.5 reaches 0.72 / 0.66
  (rater 1) and 0.39 / 0.35 (rater 2).
- **Agreement** is reported as binary Cohen's κ (§9) plus, for reference, ordinal
  Krippendorff α on the raw 1–5 scale (`score_calibration.py`, `judge_vs_expert.py`).
- **Rhyme checker** validated on the 44 labeled pairs: precision 1.000 / recall 0.941 against
  rater 1, 0.969 / 0.969 against rater 2.

---

## 11. Reproducibility and limitations

**Reproducibility.** Evaluation is fully scripted and resume-safe: model registry
(`eval/registry.py`), provider seam (`eval/providers.py`), runner (`eval/run_models.py`),
grader (`eval/grade_runs.py`), syllable count-of-record (`count_api.py`), content judge
(`judge_api.py`). Raw generations (`eval/runs/`) and graded records (`eval/graded/`) are
included, and `reproduce/make_paper_tables.py` rebuilds every paper table from them, the
count-of-record caches and the judge outputs, with no API calls. Rule grading is
deterministic and free.

**Limitations.**
1. **Count** relies on a model-based count-of-record because the dictionary tokenizer
   reaches only ~84% per line on classical verse (systematic under-count of อักษรนำ /
   Sanskrit-Pali syllables); count is therefore weighted least, and the ranking is
   unchanged under the tokenizer count.
2. **Rhyme** is graded on internal links only; cross-บท links are out of scope for single-บท
   grading.
3. **The content judge** is binary and validated against one expert's labels; it is a
   floor check for degenerate or off-topic output, not an aesthetic judgment.
4. **Composition** is the headline mode: the scaffold in cloze/edit supports weaker models
   and compresses the spread between models.
