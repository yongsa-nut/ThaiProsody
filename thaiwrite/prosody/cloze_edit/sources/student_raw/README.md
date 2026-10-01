# Student-authored source stanzas (originals)

This folder holds the original Thai stanzas that Phuvis Kerdpramote, the paper's
second author, wrote as secondary-school (ม.3) coursework. They form the student part
of the cloze/edit source pool and are released under CC BY 4.0 with the rest of the data. They were not published
elsewhere, which keeps these source stanzas out of model training data.

## Files

- `raw_khlong.txt`, `raw_klon.txt`, `raw_yani.txt`, `raw_chan.txt`: the stanzas of
  each verse form as written by the student (one line of verse per line, stanzas
  separated by blank lines).
- `original_coursework.md`: the original document as received, covering all four forms.
- `SEGMENTATION_INSTRUCTIONS.md`: the conventions used to produce the gold syllable
  segmentation.

## Processing

1. Each stanza is segmented into pronounced syllables (the gold segmentation), stored
   in `../student_<form>_gold_original.json` with provenance `student-authored (ม.3)`.
2. Each segmented stanza is checked with `composed/verify_composed.py`, which scores it
   with the meter grader and reports any failing วรรค or position.
3. `../../build_pool.py` keeps the stanzas that score exactly 1.0 under
   `grader_meter.score_meter` with their gold segmentation; the student's words are
   never changed (see `KHLONG_FAITHFUL` in `build_pool.py`).
4. `../../make_items.py` builds the cloze and edit items from the pool.
