"""Score the LLM (Claude Opus 4.7) syllable counts against the pythainlp tokenizer.

Reads the per-batch Opus outputs (audit_words/counts_*.json) and the known-gold
probe (words_meta.json), compares each word's Opus count to the grader's
tokenizer (dict, with han_solo as a cross-check), and lists every disagreement:

  * tokenizer vs Opus: the tokenizer's errors (Opus count taken as gold)
  * Opus vs hand-probe gold: where Opus and the hand counts differ

Writes audit_words/REPORT.md and audit_words/report.html.

  python audit_words_score.py
"""
from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))        # repo root
from pythainlp.tokenize import syllable_tokenize                        # noqa: E402

AW = HERE / "audit_words"


def tok(word: str, engine: str) -> int:
    return len(syllable_tokenize(word, engine=engine)) if word.strip() else 0


def boot_ci(hits, n_boot=5000, seed=20260525):
    if not hits:
        return (0.0, 0.0)
    rng = random.Random(seed); n = len(hits); ms = []
    for _ in range(n_boot):
        ms.append(sum(hits[rng.randrange(n)] for _ in range(n)) / n)
    ms.sort()
    return (ms[int(.025 * n_boot)], ms[int(.975 * n_boot)])


def load() -> list[dict]:
    meta = json.loads((AW / "words_meta.json").read_text(encoding="utf-8"))
    opus: dict[str, dict] = {}
    for f in sorted(glob.glob(str(AW / "counts_*.json"))):
        for o in json.loads(Path(f).read_text(encoding="utf-8")):
            opus[o["word"]] = o
    rows = []
    for w, m in meta.items():
        o = opus.get(w)
        if not o:
            continue
        rows.append({
            "word": w,
            "opus": o["count"],
            "syll": o.get("syllables", ""),
            "note": o.get("note", ""),
            "dict": tok(w, "dict"),
            "han_solo": tok(w, "han_solo"),
            "gold": m.get("probe_gold"),
        })
    return rows


def main() -> None:
    rows = load()
    n = len(rows)
    probe = [r for r in rows if r["gold"] is not None]

    # agreements
    opus_vs_gold = [1 if r["opus"] == r["gold"] else 0 for r in probe]
    dict_vs_gold = [1 if r["dict"] == r["gold"] else 0 for r in probe]
    dict_vs_opus = [1 if r["dict"] == r["opus"] else 0 for r in rows]
    hs_vs_opus = [1 if r["han_solo"] == r["opus"] else 0 for r in rows]

    def pct(h): return 100 * sum(h) / len(h) if h else 0.0
    lo, hi = boot_ci(dict_vs_opus)

    lines = []
    lines.append("# Syllable-count audit: Claude Opus 4.7 vs the pythainlp tokenizer\n")
    lines.append(f"- Words counted by Claude Opus 4.7 (10 parallel sessions): **{n}** "
                 f"({len(probe)} also have a hand-probe gold).\n")
    lines.append("## Reliability check (where we have a known answer)\n")
    lines.append(f"- **Opus vs hand-gold:** {sum(opus_vs_gold)}/{len(probe)} = "
                 f"**{pct(opus_vs_gold):.1f}%** agreement.\n")
    lines.append(f"- **dict tokenizer vs hand-gold:** {sum(dict_vs_gold)}/{len(probe)} = "
                 f"**{pct(dict_vs_gold):.1f}%**.\n")
    lines.append("## Tokenizer accuracy on the full Opus-gold set\n")
    lines.append(f"- **dict vs Opus:** {sum(dict_vs_opus)}/{n} = **{pct(dict_vs_opus):.1f}%** "
                 f"[95% CI {lo*100:.0f}-{hi*100:.0f}].\n")
    lines.append(f"- **han_solo vs Opus:** {sum(hs_vs_opus)}/{n} = **{pct(hs_vs_opus):.1f}%**.\n")

    tok_err = [r for r in rows if r["dict"] != r["opus"]]
    lines.append(f"\n## Tokenizer disagreements (n={len(tok_err)}): dict ≠ Opus\n")
    lines.append("| word | Opus | dict | dir | Opus syllables | note |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(tok_err, key=lambda r: r["opus"] - r["dict"]):
        d = "under" if r["dict"] < r["opus"] else "over"
        lines.append(f"| {r['word']} | {r['opus']} | {r['dict']} | {d} | {r['syll']} | {r['note']} |")

    gold_diff = [r for r in probe if r["opus"] != r["gold"]]
    lines.append(f"\n## Opus vs hand-gold disagreements (n={len(gold_diff)})\n")
    lines.append("| word | Opus | hand-gold | dict | Opus syllables | Opus note |")
    lines.append("|---|---|---|---|---|---|")
    for r in gold_diff:
        lines.append(f"| {r['word']} | {r['opus']} | {r['gold']} | {r['dict']} | {r['syll']} | {r['note']} |")

    uncertain = [r for r in rows if r["note"]]
    lines.append(f"\n## Opus-flagged uncertain (n={len(uncertain)})\n")
    for r in uncertain:
        lines.append(f"- **{r['word']}** = {r['opus']} ({r['syll']}) — {r['note']}")

    report = "\n".join(lines) + "\n"
    (AW / "REPORT.md").write_text(report, encoding="utf-8")

    # ---- HTML highlight ----
    def hrow(r, cols):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>"
    html = ["<!doctype html><meta charset=utf-8><title>Syllable-count audit</title>",
            "<style>body{font-family:'Sarabun',Tahoma,sans-serif;max-width:1000px;margin:30px auto;"
            "color:#1a1d24;line-height:1.5}h1{font-size:1.4rem}h2{font-size:1.05rem;margin-top:1.8em}"
            "table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:.95rem}"
            "td,th{border:1px solid #e3e6ea;padding:6px 10px;text-align:left}"
            "th{background:#f3eefe}.u{background:#fdecea}.o{background:#fff7e6}"
            ".big{font-size:1.2rem;font-weight:700}.k{color:#6b7280}</style>"]
    html.append(f"<h1>Syllable-count audit: Claude Opus 4.7 vs tokenizer</h1>")
    html.append(f"<p><b>{n}</b> words counted by Claude Opus 4.7 (10 parallel sessions) "
                f"({len(probe)} with hand-gold).</p>")
    html.append("<p class=big>dict tokenizer vs Opus: "
                f"{pct(dict_vs_opus):.1f}% <span class=k>[CI {lo*100:.0f}-{hi*100:.0f}]</span> "
                f"&nbsp;·&nbsp; Opus vs hand-gold: {pct(opus_vs_gold):.1f}%</p>")
    html.append(f"<h2>Tokenizer disagreements ({len(tok_err)}) — highlighted</h2>")
    html.append("<table><tr><th>word</th><th>Opus</th><th>dict</th><th>dir</th>"
                "<th>Opus syllables</th><th>note</th></tr>")
    for r in sorted(tok_err, key=lambda r: r["opus"] - r["dict"]):
        d = "under" if r["dict"] < r["opus"] else "over"
        cls = "u" if d == "under" else "o"
        html.append(f"<tr class={cls}>" + "".join(
            f"<td>{c}</td>" for c in [r['word'], r['opus'], r['dict'], d, r['syll'], r['note']]) + "</tr>")
    html.append("</table>")
    if gold_diff:
        html.append(f"<h2>Opus vs hand-gold disagreements ({len(gold_diff)})</h2>")
        html.append("<table><tr><th>word</th><th>Opus</th><th>hand-gold</th><th>dict</th>"
                    "<th>syllables</th><th>note</th></tr>")
        for r in gold_diff:
            html.append(hrow(r, [r['word'], r['opus'], r['gold'], r['dict'], r['syll'], r['note']]))
        html.append("</table>")
    (AW / "report.html").write_text("\n".join(html), encoding="utf-8")

    # console
    print(report)
    print(f"→ wrote {(AW/'REPORT.md').name} + {(AW/'report.html').name}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
