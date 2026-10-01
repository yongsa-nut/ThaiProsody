# ThaiProsody: A Benchmark for Thai Classical Poetry in Large Language Models

Code, data and model outputs for the paper

> **Composing Within Constraints: A Benchmark for Thai Classical Poetry in Large Language Models**
> Nutchanon Yongsatianchot (Faculty of Medicine Siriraj Hospital, Mahidol University) and
> Phuvis Kerdpramote (Department of Computer Science, School of Science, King Mongkut's Institute of Technology Ladkrabang (KMITL)).
> AACL-IJCNLP 2026. [Paper (PDF)](paper/ThaiProsody_AACL2026.pdf)

Classical Thai verse asks a poet to satisfy several formal constraints at once: a fixed
number of syllables per line, written tone marks (เอก/โท) at fixed positions, light and
heavy syllables (ครุ/ลหุ) in a fixed pattern, and an interlocking rhyme scheme (สัมผัส),
all while staying on topic. ThaiProsody measures how well large language models compose
under these constraints. It covers four canonical verse forms, asks for one stanza per
item in a zero-shot prompt that names only the form and the topic, and grades every
output with deterministic rule checks plus a model-based syllable count. The repository
contains the benchmark items, the grader, the harness used to run the models, every
model output and graded record behind the paper, the expert annotations used to
validate the grader and the content judge, and a script that rebuilds every table of
the paper from these files without calling any model.

## Citation

```bibtex
@inproceedings{yongsatianchot2026composing,
  title     = {Composing Within Constraints: A Benchmark for Thai Classical Poetry in Large Language Models},
  author    = {Yongsatianchot, Nutchanon and Kerdpramote, Phuvis},
  booktitle = {Proceedings of the 2026 Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the International Joint Conference on Natural Language Processing (AACL-IJCNLP 2026)},
  year      = {2026}
}
```

## What is in the benchmark

| Form | Thai | Constraints graded | Meter weights |
|---|---|---|---|
| *Khlong* | โคลงสี่สุภาพ | count, tone, rhyme | .25 / .40 / .35 |
| *Klon* | กลอนแปด | count, rhyme | .40 / .60 |
| *Kap* | กาพย์ยานี ๑๑ | count, rhyme | .40 / .60 |
| *Chan* | อินทรวิเชียรฉันท์ ๑๑ | count, weight, rhyme | .20 / .50 / .30 |

* **Composition (main task): 160 items** = 4 forms x 40 topics. The topics fall into six
  categories (nature, abstract, emotion, everyday life, seasons, festivals). The items are
  stored by topic in two files of 80 items each
  (`thaiwrite/prosody/data/prompts_compose_public.jsonl` and `prompts_compose_private.jsonl`);
  the paper reports results on all 160. We ask that the items and model outputs not be used
  as training data, so that the benchmark stays informative.
* **Cloze: 178 items.** A valid stanza with one or two meter-bearing positions blanked;
  the model returns the completed stanza.
* **Edit: 74 items** (*Khlong* and *Chan*). A valid stanza with one injected violation
  (a missing tone mark or a heavy syllable in a light slot); the model returns the
  corrected stanza.

The cloze and edit items are built from unpublished stanzas: stanzas the second author
wrote as secondary-school coursework and rule-verified stanzas
composed by Claude Opus 4.7 (`thaiwrite/prosody/cloze_edit/`, with its own README).

Model outputs are included for all configurations in the paper: 19 model configurations
on the 160 composition items, five of them also with thinking disabled, four with the
form's rules in the system prompt, and eight with two further generation passes for the
replication study, plus the cloze and edit runs.

## How grading works

Each output is cleaned (reasoning traces, code fences, markdown and stanza labels are
removed) and graded on the dimensions its form binds. Each dimension is a pass rate in
[0, 1], the fraction of that dimension's rule checks the poem satisfies.

* **Tone, weight and rhyme** are deterministic (`thaiwrite/prosody/grader_meter.py`, with
  the form definitions in `meter_specs.py`). Tone and weight use `pythainlp.khavee` with a
  dead-syllable fallback for tone and two weight corrections; rhyme uses the `khavee` rime
  check plus fallbacks for leading-consonant (อักษรนำ) and silent-final (การันต์) words and
  open ไ-/ใ- syllables. No model is involved.
* **Syllable count** uses a count-of-record: every line piece the grader scores is counted
  once by Claude Opus 5.5 through the Claude API with a fixed prompt
  (`thaiwrite/prosody/count_api.py`, prompt in `vrk_counter.py`; `vrk` stands for วรรค, a
  line of verse), and the counts are cached
  (`thaiwrite/prosody/vrk_count_cache_opus55_main.json` for the composition runs, native
  and with rules in the prompt; `vrk_count_cache_opus55_bcd.json` for the thinking-disabled
  runs, the replication runs and the cloze and edit outputs). A line is counted as the sum of the cached counts of its
  whitespace-separated pieces. The grader's own dictionary-tokenizer count is stored in
  every graded record and serves as a robustness comparison.
* **Meter score** = weighted mean of the bound dimensions (weights in the table above).
  Models lay poems out differently (one line per วรรค or two), so each output is graded under
  both segmentations and the higher score is kept.
* **Valid stanza** = the poem passes every check of its form.
* **Non-delivery** (empty output or token runout) scores 0. Infrastructure failures are
  not scored. Means come with item-bootstrap 95% confidence intervals (1000 resamples,
  seed 20260524).
* **Content judge** (reported separately, not part of the meter score): Claude Opus 5.5
  through the Claude API (`thaiwrite/prosody/judge_api.py`, rubric in
  `grader_coherence_prompt.txt`) rates topicality and coherence from 1 to 5; a poem counts
  as on-topic at topicality >= 3 and as coherent at coherence >= 2. A poem the judge
  declines to rate (for example a leaked reasoning trace) counts as neither.

## Reproducing the paper's tables (no API calls)

```bash
pip install -r requirements.txt
PYTHONIOENCODING=utf-8 python reproduce/make_paper_tables.py
```

This reads the graded records in `thaiwrite/prosody/eval/graded/`, the count-of-record
caches and the content-judge outputs, and writes to `reproduce/out/`:

* `tables.md`: every table in readable form (leaderboard with valid-stanza rate and
  Top./Coh., per-dimension error rates, count-of-record versus tokenizer, thinking
  disabled, rules in the prompt, topic categories, replication, rhyme checker versus the
  expert labels, content-judge agreement, count-of-record checks);
* `tables.tex`: the same tables in LaTeX (booktabs);
* `numbers.json`: every number, machine-readable;
* `uncounted_pieces.json`: line pieces missing from the count caches (empty when the
  caches cover every graded line).

The script only reads the released files; it never modifies `eval/graded/`. Options:
`--cache` (one or more count caches; for a piece present in several, the first listed
wins), `--judge` (one or more judge directories; for each item the first directory
holding a judgment of the same poem text wins; `tables.md` lists which directory judged
each configuration), `--graded-dir` (one or more directories
of graded files) and `--out`. For example, to compute Top./Coh. with a second judge,
Claude Opus 4.7 on the same rubric, instead of Claude Opus 5.5:

```bash
python reproduce/make_paper_tables.py --judge thaiwrite/prosody/eval/judged_archive
```

The grader's unit tests also run without API access:

```bash
PYTHONIOENCODING=utf-8 python thaiwrite/prosody/test_grader_meter.py
PYTHONIOENCODING=utf-8 python thaiwrite/prosody/test_grader_fixes.py
```

## Evaluating a new model

Generating, counting and judging call model APIs; set the keys you need in the
environment or in a `.env` file (for example `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
`ANTHROPIC_API_KEY`). Keep the new model's files in new directories so that the released
records stay unchanged.

1. Register the model in `thaiwrite/prosody/eval/registry.py` (provider, model id,
   token budget, reasoning settings) and generate the 160 composition outputs:

   ```bash
   python -m thaiwrite.prosody.eval.run_models --models <key> --sets compose_public,compose_private
   ```

   Raw outputs go to `thaiwrite/prosody/eval/runs/<key>__run1.jsonl`; re-running resumes
   where it stopped.

2. Grade them into a new directory (`--no-coherence` grades the meter only; step 4 adds
   the content judge):

   ```bash
   python thaiwrite/prosody/eval/grade_runs.py --models <key> --no-coherence --out thaiwrite/prosody/eval/graded_new
   ```

3. Count the new line pieces with the count-of-record model, into a separate cache
   (`--scope A` selects the native composition outputs in the given directory; the other
   scopes are B for thinking-disabled and rules-in-prompt runs, C for replication runs and
   D for cloze and edit outputs):

   ```bash
   python thaiwrite/prosody/count_api.py run --scope A --graded-dir thaiwrite/prosody/eval/graded_new \
       --cache thaiwrite/prosody/vrk_count_cache_new.json
   ```

4. Judge topicality and coherence:

   ```bash
   python thaiwrite/prosody/judge_api.py run --scope all --graded-dir thaiwrite/prosody/eval/graded_new \
       --out thaiwrite/prosody/eval/judged_new
   ```

5. Build the tables with the new model next to the released ones:

   ```bash
   python reproduce/make_paper_tables.py \
       --graded-dir thaiwrite/prosody/eval/graded thaiwrite/prosody/eval/graded_new \
       --cache thaiwrite/prosody/vrk_count_cache_opus55_main.json thaiwrite/prosody/vrk_count_cache_opus55_bcd.json \
               thaiwrite/prosody/vrk_count_cache_new.json \
       --judge thaiwrite/prosody/eval/judged_api_all thaiwrite/prosody/eval/judged_api thaiwrite/prosody/eval/judged_new \
       --out reproduce/out_new
   ```

The cloze and edit tasks have their own pipeline (generation, grading and report); see
`thaiwrite/prosody/cloze_edit/README.md`.

## Repository layout

```
paper/ThaiProsody_AACL2026.pdf     the paper
reproduce/make_paper_tables.py     rebuilds every paper table from the files below (no API calls)
thaiwrite/prosody/
  meter_specs.py                   the four forms: line structure, tone and weight positions, rhyme links
  grader_meter.py                  rule-based grader (count, tone, weight, rhyme; meter score)
  count_api.py, vrk_counter.py     syllable count-of-record through the Claude API, and its prompt
  vrk_count_cache_opus55*.json     count-of-record caches (line piece -> syllable count)
  judge_api.py                     content judge through the Claude API
  grader_coherence_prompt.txt      content-judge rubric
  grader_coherence.py              judge client used by grade_runs.py when a judge is requested
  make_prompts.py                  builds the composition prompts from data/topics.py
  test_grader_meter.py, test_grader_fixes.py   grader unit tests
  audit_*.py, audit_words/         audits of the khavee checks and of syllable counting
  data/                            composition prompts (public and private), topics, audit poems
  composed/                        rule-verified stanzas composed by Claude Opus 4.7
  eval/
    registry.py, providers.py, run_models.py, promptlib.py, rulesheets.py   generation harness
    grade_runs.py                  grades raw outputs
    runs/                          raw model outputs (<config>__run<k>.jsonl)
    graded/                        graded records used in the paper; graded/replication/ for runs 1-3
    judged_api_all/, judged_api/   content judgments used in the paper (Claude Opus 5.5)
    judged_archive/                content judgments by a second judge, Claude Opus 4.7, same rubric
  calibration/                     calibration items, both raters' labels and the adjudicated labels
                                   (labels/), annotation interface, judge scores on the calibration
                                   poems (Claude Opus 4.7, GPT-5.5, and judge_api.py calib output),
                                   agreement scripts (rater_agreement.py, score_calibration.py)
  cloze_edit/                      cloze and edit tasks: source stanzas, items, runs, graded outputs
pilot_rerun/clients.py             HTTP client (retries, OpenAI-compatible endpoints) used by the harness
docs/                              form specifications, methods, khavee audit, annotation guidelines
```

Configuration names: `<model>` is the native run, `<model>.nothink` the run with thinking
disabled, and `<model>.rules` the run with the form's rules in the system prompt.

## Requirements

Python 3.10 or later (tested with 3.11). Install with `pip install -r requirements.txt`:
`pythainlp` 5.3.1 for the grader and the table script, `anthropic` for `count_api.py` and
`judge_api.py`, and `openai`, `requests` and `python-dotenv` for the generation harness.
On Windows, set `PYTHONIOENCODING=utf-8` when a script prints Thai text.

## License

The source code is released under the MIT License (`LICENSE`). The benchmark data, the
model outputs and the annotations are released under the Creative Commons Attribution 4.0
International License (CC BY 4.0, `LICENSE-DATA`).
