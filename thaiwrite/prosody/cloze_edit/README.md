# Cloze and edit evaluation

Two extra task modes on the same four verse forms and the same meter grader as the
composition benchmark:

* **Cloze**: a valid stanza with one or two meter-bearing positions blanked
  (`____`): rhyme sources, *mai-ek* slots (Khlong), light/heavy slots (Chan). The model
  returns the completed stanza.
* **Edit**: a valid stanza with one injected violation: a required tone mark removed
  (Khlong) or the heavy syllable มา placed in a light slot (Chan). The model returns the
  corrected stanza. Klon and Kap bind no tone or weight rule, so edit covers Khlong and
  Chan only.

Prompts, system role, providers and per-model generation settings are those of the
composition runs (`eval/promptlib.py`, `eval/registry.py`).

## Pipeline

Run from the repository root with `PYTHONIOENCODING=utf-8`.

| step | command | output |
|---|---|---|
| 1. source pool | `python thaiwrite/prosody/cloze_edit/build_pool.py` | `data/pool.json`, `data/pool_report.md` |
| 2. items | `python thaiwrite/prosody/cloze_edit/make_items.py` | `data/items_cloze.jsonl`, `data/items_edit.jsonl` |
| 3. generation (API calls) | `python thaiwrite/prosody/cloze_edit/run_ce.py --models <key>[,<key>...] --concurrency 8` | `runs/<key>__run1.jsonl` (resume-safe) |
| 4. grading | `python thaiwrite/prosody/cloze_edit/grade_ce.py [--out DIR]` | `graded_new/<key>__run1.jsonl`, `graded_new/summary.json` |
| 5. report | `python thaiwrite/prosody/cloze_edit/report_ce.py [--graded DIR] [--out DIR]` | `report/results.json`, `report/results.md`, `report/table_rows.tex` |
| spend | `python thaiwrite/prosody/cloze_edit/cost_ce.py` | stdout |

The released raw generations are in `runs/` and their grading in `graded/`, so steps 4
and 5 run without any API call. `grade_ce.py` writes to `graded_new/` by default so the
archived `graded/` files are not overwritten; `report_ce.py` reads `graded/` by default
(pass `--graded thaiwrite/prosody/cloze_edit/graded_new` to report a fresh regrade).
Steps 1 and 2 rebuild `data/pool.json` and `data/items_*.jsonl` in place.

Model keys in the paper table: `gpt55-med`, `gemini-3.5-flash`, `gemma-4-31b`,
`deepseek-v4-flash`, `typhoon-s`, `openthaigpt`, `qwen3-8b`, `typhoon25`, all with their
composition settings from `eval/registry.py`. `run_ce.py` requests OpenRouter
reasoning explicitly for `deepseek-v4-flash` (OpenRouter serves it without reasoning unless
requested; the composition run reasoned natively), and defines two supplementary
configurations reported only in `results.md`:

* `deepseek-v4-pro`: DeepSeek V4 Pro through the official DeepSeek API (native
  thinking, 24k-token budget, as in the composition run).
* `gemma-4-31b-think`: Gemma 4 31B through the Google AI API, which always runs Gemma 4
  in thinking mode (16k-token budget). The composition run of Gemma used a
  non-thinking endpoint, so this is a different configuration of the same weights.

API keys are read from the environment or the nearest `.env` (`OPENAI_API_KEY`,
`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `THAILLM_API_KEY`,
`TYPHOON_API_KEY`).

## Source pool (`build_pool.py`)

Only stanzas that were never published are used:

* **student**: stanzas the paper's second author wrote as secondary-school (ม.3)
  coursework (`sources/student_raw/`). The student's
  words are used verbatim; we supply only a syllable segmentation
  (`sources/student_*_gold_original.json`; for Khlong, the word lists in `build_pool.py`
  (`KHLONG_FAITHFUL`) carry the student's original wording and take precedence over the
  segmentation file where the two differ).
* **model**: stanzas composed by Claude Opus 4.7 and rule-verified
  (`sources/model_composed_all_composed.json`, entries with provenance "Opus 4.7"),
  after a manual audit of syllabification and vocabulary (`MODEL_EXCLUDE` lists the
  stanzas removed and why). Web-found published poems in that file are not used.

A stanza enters the pool only if it scores exactly 1.0 under
`grader_meter.score_meter` with its gold segmentation. Per form, the pool takes up to the
benchmark's source-pool size (Khlong 12, Klon 14, Kap 20, Chan 12), student stanzas
first (`in_main = true`); the remaining verified model-composed stanzas form a control
set used only for the provenance comparison. Each stanza stores, per วรรค, the written
text shown to the model and its gold syllables, aligned into units (a respelled
multi-syllable word such as สราญ = สะ.ราญ is one unit). A Khlong คำสร้อย is shown in its
traditional place (after วรรค 6).

## Items (`make_items.py`)

Cloze blanks up to two hard positions per item,
positions partitioned across up to four variants per stanza (six for Chan); edit injects
one violation per item, at a different slot per variant (up to four per Khlong stanza,
every light slot per Chan stanza), and keeps an injection only if it actually breaks the
meter. A blank covers the written unit holding the target syllable. Each item records its
source-stanza id, provenance, the source text and the reference segmentation.

## Grading (`grade_ce.py`)

* Cleaning as in composition (`promptlib.clean_poem`), plus: a Gemma thinking block
  (`<thought>...</thought>`) is removed, and parenthetical commentary lines are dropped.
  If an output has more วรรค than the stanza (e.g. a preface explaining the fix), the
  contiguous window that best matches the prompt's stanza is graded.
* **meter**: `grader_meter.score_meter` on the returned stanza, best of the two layouts
  (as written / whitespace-split), non-delivery = 0. The meter score alone checks
  neither the blanks nor the extent of changes; **faithful** below does.
* **segmentation**: gold-anchored; syllables the output shares with the prompt keep the
  source stanza's verified segmentation (a fill identical to the source word also keeps
  it); all new material is segmented with the grader's dictionary tokenizer. The
  count-of-record is not used for these outputs. `meter_tok` is the pure-tokenizer score.
* **valid**: meter = 1.0.
* **faithful**: cloze: every given syllable of every วรรค reproduced and every blank
  filled; edit: every วรรค except the faulty one unchanged and the faulty one changed
  (whitespace ignored).
* **solved**: valid and faithful.
* CIs: 95% cluster bootstrap over source stanzas (2,000 resamples).

Sanity checks: returning the source stanza solves every item (1.0, faithful); returning
an edit item's input unchanged solves none; both layouts are handled.
