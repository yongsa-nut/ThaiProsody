# Syllable-count audit: Claude Opus 4.7 vs the pythainlp tokenizer

- Words counted by Claude Opus 4.7 (10 parallel sessions): **120** (91 also have a hand-probe gold).

## Reliability check (where we have a known answer)

- **Opus vs hand-gold:** 90/91 = **98.9%** agreement.

- **dict tokenizer vs hand-gold:** 74/91 = **81.3%**.

## Tokenizer accuracy on the full Opus-gold set

- **dict vs Opus:** 100/120 = **83.3%** [95% CI 77-90].

- **han_solo vs Opus:** 100/120 = **83.3%**.


## Tokenizer disagreements (n=20): dict ≠ Opus

| word | Opus | dict | dir | Opus syllables | note |
|---|---|---|---|---|---|
| มนุษย์ | 2 | 1 | under | มะ.นุด |  |
| สังเกต | 3 | 2 | under | สัง.เก.ต | สัง.เกด in normal speech (2), but in verse scanning the final ◌ต is read as a syllable เกด/กด; commonly scanned สัง-เก-ต = 3 |
| ฤดู | 2 | 1 | under | รึ.ดู |  |
| โทรศัพท์ | 3 | 2 | under | โท.ระ.สับ |  |
| ธรรมชาติ | 3 | 2 | under | ทำ.มะ.ชาด |  |
| อร่อย | 2 | 1 | under | อะ.หฺร่อย |  |
| ตลาด | 2 | 1 | under | ตะ.หฺลาด |  |
| ขนม | 2 | 1 | under | ขะ.หฺนม |  |
| มหา | 2 | 1 | under | มะ.หา |  |
| กรุณา | 3 | 2 | under | กะ.รุ.นา |  |
| ฉันทลักษณ์ | 3 | 2 | under | ฉัน.ทะ.ลัก |  |
| สมุทร | 2 | 1 | under | สะ.หฺมุด |  |
| สบาย | 2 | 1 | under | สะ.บาย |  |
| สวรรค์ | 2 | 1 | under | สะ.หฺวัน |  |
| ถนน | 2 | 1 | under | ถะ.หฺนน |  |
| ความเงียบสงบ | 4 | 3 | under | ความ.เงียบ.สะ.หฺงบ |  |
| จมูก | 2 | 1 | under | จะ.หฺมูก |  |
| ฉลาด | 2 | 1 | under | ฉะ.หฺลาด |  |
| สนาม | 2 | 1 | under | สะ.หฺนาม |  |
| วิทยา | 3 | 1 | under | วิด.ทะ.ยา |  |

## Opus vs hand-gold disagreements (n=1)

| word | Opus | hand-gold | dict | Opus syllables | Opus note |
|---|---|---|---|---|---|
| สังเกต | 3 | 2 | 2 | สัง.เก.ต | สัง.เกด in normal speech (2), but in verse scanning the final ◌ต is read as a syllable เกด/กด; commonly scanned สัง-เก-ต = 3 |

## Opus-flagged uncertain (n=9)

- **สังเกต** = 3 (สัง.เก.ต): สัง.เกด in normal speech (2), but in verse scanning the final ◌ต is read as a syllable เกด/กด; commonly scanned สัง-เก-ต = 3
- **เปรมปรีดิ์** = 2 (เปฺรม.ปฺรี): ◌์ silences final ดิ์; ปรี is true cluster
- **วันสงกรานต์** = 3 (วัน.สง.กราน): ต์ silenced; กราน is one syllable (ควบแท้ กร)
- **อินทรวิเชียร** = 4 (อิน.ทอ.ระ.วิ.เชียร): in verse อินทร- read อิน.ทะ.ระ; full scan อิน.ทะ.ระ.วิ.เชียร = 5; common reading อิน.ทอ.ระ.วิ.เชียร also 5
- **ไหล** = 1 (ไหล): silent leading ห before ล
- **เล่ห์** = 1 (เล่ห์): ห์ silenced
- **สัตว์** = 1 (สัด): ว์ silenced
- **ครอบครัว** = 2 (ครอบ.ครัว): each ควบแท้ คร = one syllable
- **ครับ** = 1 (ครับ): ควบแท้ คร = one syllable
