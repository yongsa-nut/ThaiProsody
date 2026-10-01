# Human Calibration Protocol

This document describes how the two grading layers were validated against human labels.
The annotator instructions are in `annotation-guidelines.md`; the resulting agreement
figures are reported in the paper (instrument-validation appendix) and reproduced by
`reproduce/make_paper_tables.py`. How the calibration results shaped the content judge
is summarized in `methods.md` §9–10.

---

## 0. What calibration measures

The calibration answers two structurally different questions, one per grading layer:

| Layer | Nature | Calibration question | Metric |
|---|---|---|---|
| **L1: meter grader** | deterministic | **Validity**: does the rhyme check match the expert's judgment by sound? | precision / recall on 44 labeled rhyme pairs (tone and weight: audit positions, `khavee-audit.md`) |
| **L2: content judge** | stochastic + subjective | **Agreement**: does the judge match the expert? | binary Cohen's κ per axis; ordinal Krippendorff α on the 1–5 scale for reference |

---

## 1. Design

1. **The calibration set is sampled from real model output.** Poems are drawn from the graded
   composition runs, so the judge is validated on the output distribution it is applied to,
   and supplemented with known-good and known-broken anchors.
2. **Agreement metric.** Krippendorff's α uses an **ordinal** distance for the 1–5 axes and a
   **nominal** distance for binary labels such as the rhyme verdicts. The content judge is
   reported as a binary flag, so its agreement is also reported as binary Cohen's κ at the
   cuts the benchmark uses (topicality ≥ 3; coherence ≥ 2).
3. **Layer 1 weights are fixed, not tuned.** The per-dimension breakdown is the
   diagnostic output and the weighted meter score is the ranking measure. The annotation
   interface also has optional per-dimension meter fields; the released labels cover
   topicality, coherence and the rhyme pairs.

---

## 2. The calibration set (176 items)

`thaiwrite/prosody/calibration/calibration_items_v2.json`, built by `build_calibration_v2.py`:

| Source | n | Role |
|---|---|---|
| Model poems from the composition runs | 120 | realistic output across the score range |
| Clean anchor poems (score 1.0) | 5 | meter compliant, high coherence |
| Injected-break anchor poems | 7 | meter specifically broken |
| **Poems, total** | **132** | 33 โคลง / 34 กลอน / 32 ยานี / 33 ฉันท์ |
| Rhyme word pairs | 44 | rhyme / no rhyme, judged by sound |

- **Sampling.** Model poems are stratified by meter and by model tier (high / mid / low
  expected coherence), so the word-salad end of the coherence scale is represented. Within
  each meter × tier cell, poems are drawn round-robin across models with topics balanced,
  then shuffled so the annotator sees no tier or model ordering cue (the tool never shows
  provenance).
- **Rhyme micro-set.** The rhyme layer is L1's highest validity risk (khavee fails
  อักษรนำ / การันต์ words; see `khavee-audit.md`). A separate set of 44 valid/invalid rhyme
  pairs, deliberately loaded with อักษรนำ, การันต์ and vowel-variant cases, measures the
  fallback's precision and recall directly.
- Each poem item carries `id`, `meter`, `task_mode`, `topic`, `poem_text` (newline-delimited
  วรรค) and `applicable_dims` (the meter dimensions that meter binds).

`calibration_items.json` (66 poems + 44 rhyme pairs, built by `build_calibration_set.py`
from two models' outputs) is the smaller set that `build_calibration_v2.py` extends; it uses
the same schema.

---

## 3. Annotator

The annotator is a native Thai speaker with a doctoral degree and the standard
Thai secondary-school training in classical prosody. The annotation covers judgments a
competent native speaker makes reliably (topicality, coherence, rhyme by sound); it does not
ask for aesthetic judgment.

The guidelines (`annotation-guidelines.md`) mirror the instruments exactly:
- *Content axes*: the **same 1/3/5 anchors** as `grader_coherence_prompt.txt`, so the human
  and the judge share a rubric, with the same hard constraints (do not judge meter; do not
  penalize archaic vocabulary).
- *Meter dimensions*: defined against `meter-specs.md`; for each dimension, compliant ✓ /
  violated ✗, with a one-line definition and a worked example.

---

## 4. The annotation tool

A **single self-contained HTML file** (`calibration/annotate.html`): no server, no account,
no build. `bundle_for_experts.py` bakes the items and the annotator id into one file per
annotator (`annotate_<id>.html`).

- **One item per screen**: meter + topic header and the poem, one วรรค per line.
- **Layer 2**: two 1–5 rows (Topicality, Coherence) with the rubric anchors shown inline;
  keys **1–5** set the focused axis.
- **Layer 1**: a ✓/✗ toggle per *applicable* dimension (count / tone / rhyme / ครุ-ลหุ) plus
  an optional note.
- **Rhyme pairs**: a second item type showing a word pair with one valid / invalid toggle.
- Prev / Next, a progress bar and **autosave to `localStorage`** (the annotator can stop and
  resume).
- **Export** downloads `labels_<id>.json` (all ratings and per-item latency), the only output.

The tool makes no network calls and records nothing beyond the annotator id and the labels.

---

## 5. Procedure

1. **Build the set** (`build_calibration_v2.py`) and bundle the annotation file
   (`bundle_for_experts.py`).
2. **Labeling.** The expert labeled all 176 items. The labels are released as
   `calibration/labels/labels_expertA.json`.
3. **Judge comparison.** Candidate judges (Claude Opus and GPT-5.5 with medium reasoning)
   were run under an identical prompt over the 132 poems and compared with the expert
   labels per axis (`judge_vs_expert.py`, `binary_agreement.py`); the judge that detects
   degenerate output was selected (`methods.md` §9).
4. **L1 validation.** The rhyme checker was run over the 44 labeled pairs and compared with
   the expert's verdicts (precision / recall; `reproduce/make_paper_tables.py`). The tone and
   weight checks were validated on hand-annotated audit positions (`khavee-audit.md`).

---

## 6. Metrics

**Layer 2 (content judge):**
- Judge vs expert, per axis: binary Cohen's κ at the benchmark's cuts, and ordinal
  Krippendorff α on the 1–5 scale for reference.
- Score-distribution check: whether the judge uses the full 1–5 range (in particular, whether
  it ever assigns the lowest coherence score).

**Layer 1 (meter grader):**
- Rhyme micro-set **precision/recall** against the expert's verdicts.
- Tone and weight: agreement with hand-annotated audit positions (`khavee-audit.md`).
- L1 is deterministic, so a disagreement with the expert is a grader error to inspect, not
  a model property.

---

## 7. Out of scope

- Aesthetic quality and register (the benchmark measures form, topicality and coherence).
