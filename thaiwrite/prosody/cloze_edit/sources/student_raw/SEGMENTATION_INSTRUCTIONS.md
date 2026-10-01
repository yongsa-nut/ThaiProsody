# Segmentation instructions for the student-authored stanzas

These are the instructions used to convert the raw student stanzas of one verse form
into the benchmark's gold JSON (`../student_<form>_gold_original.json`) and to check
every stanza against the meter grader. They document the syllable conventions of the
gold segmentation.

## Task

You are given a file of **raw Thai stanzas** for ONE meter. Convert each stanza into the
benchmark's gold JSON and **grade-gate every poem to a perfect score of 1.0** using the
provided grader.

## Output format (a JSON array, one object per stanza)
```json
{ "meter_id": "<the meter id you are told>",
  "source": "<short topic/title, e.g. งาน (stanza 3)>",
  "provenance": "student-authored (ม.3)",
  "lines": [ ["พยางค์","พยางค์", ...], ... ] }
```
`lines` = the list of วรรค; **each วรรค is a list of its pronounced syllables**
(one syllable per string). This supplied segmentation IS the gold: the count layer
counts your tokens, so the token count of each วรรค must equal the meter's required
syllable count.

## Syllable conventions (count by PRONUNCIATION, not characters)
1. การันต์ (◌์) silenced final adds no syllable: จันทร์ = 1, ศักดิ์ = 1.
2. อักษรนำ / อักษรควบไม่แท้ with an inserted half-vowel = SEPARATE syllables:
   ขนม = ขะ.หนม (2), สบาย = สะ.บาย (2), สถาน = สะ.ถาน (2), ปัญญา = ปัน.ยา (2).
3. True cluster (อักษรควบแท้) = ONE syllable: ปลา = 1, ครวญ = 1, พริ้ว = 1.
4. Silent leading ห/อ = ONE syllable: หนา = 1, อยู่ = 1.
5. Sanskrit/Pali by Thai pronunciation: มนุษย์ = 2, วิทยา = 3, ศรัทธา = สัด.ทา (2),
   ภาวนา = 2 or 3 (use the reading that makes the line scan).
6. Write each token as the syllable's text. When one written word spans 2 syllables,
   split it (e.g. "สถาพร" becomes ["สะ","ถา","พร"]).

## The meter (count per วรรค)
Use the given `meter_id` and its exact วรรค counts. (For โคลงสี่สุภาพ: 8 core วรรค of
5,2,5,2,5,2,5,4 syllables, **plus** any 2-syllable คำสร้อย, e.g. "...นา", "...เอย",
appended as EXTRA วรรค at the END of the `lines` array, after the 8 core วรรค.)

## Self-verify loop
Write the JSON to the output path, then run from `thaiwrite/prosody/`:
```
PYTHONIOENCODING=utf-8 python composed/verify_composed.py <your json path>
```
It prints, per poem, the score and, for any failing layer, the EXACT failing วรรค /
syllable / position. **Fix and re-run until every poem scores 1.0**, or until the
failure is clearly the poem's own (a real prosodic error in the source) rather than a
segmentation error. Distinguish the two:
- A **count** failure after correct segmentation means the source line has the wrong
  number of syllables (a real error): leave it and note it.
- A **tone / ครุ-ลหุ / rhyme** failure means the source does not satisfy that rule (a
  real error): note it.
- If re-segmenting (a legitimate alternate reading) fixes it, do that.

## Summary to report
- How many of N stanzas reach 1.0 (gold) vs fail.
- For each FAIL: the stanza number, its score, and the one-line reason from the grader
  (e.g. "วรรค2 has 6 พยางค์, need 5" or "tone: (วรรค6,พยางค์2) need โท got เอก").
- Keep ALL stanzas in the JSON (passing and failing); failing ones are flagged by their
  score when re-graded.
