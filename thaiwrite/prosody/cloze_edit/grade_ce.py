"""Grade the cloze/edit runs.

Per output (after the composition pipeline's cleaning: reasoning traces, code fences,
markdown, stanza labels; plus trailing parenthetical commentary):

  meter: the rule-based meter score of the returned stanza (grader_meter.score_meter,
           same layer weights as composition), best of the two layouts (one วรรค per
           line as written, or whitespace-split), non-delivery = 0. This is exactly how
           eval/grade_runs.py scores cloze and edit (it applies the composition grader
           to the returned poem), except for the segmentation (below).
  segmentation: gold-anchored. Every syllable the output shares with the prompt
           (matched by position against the prompt's วรรค) keeps the source stanza's
           hand-verified segmentation; only NEW material (a cloze fill, an edited
           stretch, or an unmatched line) is segmented with the grader's dictionary
           tokenizer (pythainlp `syllable_tokenize(engine="dict")`). Thus a faithful
           restoration of the source scores exactly 1.0. The pure-tokenizer score
           (grade_runs' own segmentation) is reported alongside as `meter_tok`.
           A โคลง คำสร้อย written after วรรค 2 or 6 is moved to the optional slots.
  faithful: cloze: the output reproduces every given (non-blank) syllable of every
           วรรค verbatim and fills every blank; edit: every วรรค except the faulty one
           is reproduced verbatim, and the faulty วรรค is changed. (Whitespace ignored.)
  solved: faithful AND meter = 1.0 (all rules of the stanza satisfied).

CIs: cluster bootstrap over source stanzas (items built from the same stanza are
resampled together), 95% percentile, 2000 resamples.

Outputs go to --out (default cloze_edit/graded_new/), so the archived grading in
cloze_edit/graded/ is not overwritten by default:
  <out>/<key>__run1.jsonl (per-item records) and <out>/summary.json.

  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/grade_ce.py
  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/grade_ce.py --out thaiwrite/prosody/cloze_edit/graded_new
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from thaiwrite.prosody import meter_specs as M                           # noqa: E402
from thaiwrite.prosody.eval.promptlib import clean_poem, to_vrk_lines    # noqa: E402
from thaiwrite.prosody.grader_meter import score_meter, split_syllables  # noqa: E402

DATA, RUNS, GRADED = HERE / "data", HERE / "runs", HERE / "graded_new"   # graded/ = archived
FORMS = ["khlong_si_suphap", "klon_paet", "kap_yani_11", "intharawichian_chan_11"]
FORM_SHORT = {"khlong_si_suphap": "Khlong", "klon_paet": "Klon", "kap_yani_11": "Kap",
              "intharawichian_chan_11": "Chan"}
N_BOOT, SEED = 2000, 20260929
_NO_START = set("ัิีึืุู็่้๊๋์ํะาำๅ")       # cannot begin a syllable
_LEAD_V = set("เแโใไ")


# ---------------------------------------------------------------- cleaning
def ce_clean(text: str) -> str:
    if not text:
        return ""
    low = text.lower()
    if "</thought>" in low:                  # Gemma-4 thinking channel emitted as text
        text = text[low.rindex("</thought>") + len("</thought>"):]
    elif "<thought>" in low:
        return ""                             # truncated thinking, no answer
    poem = clean_poem(text)
    keep = []
    for ln in poem.splitlines():
        s = ln.strip()
        if s.startswith(("(", "（", "[")):     # parenthetical commentary after the poem
            continue
        keep.append(ln)
    return "\n".join(keep).strip()


def _lines(poem: str) -> list[str]:
    return [ln.strip() for ln in poem.splitlines() if ln.strip()]


# ---------------------------------------------------------------- segmentation
def _template(ref_line):
    pat = "".join(re.escape(u[0]) if u else "(.*?)" for u in ref_line)
    return re.compile(pat, re.S)


def seg_line(s: str, ref_line, gold_blanks: dict | None = None) -> tuple[list[str], list[str] | None]:
    """Gold-anchored syllables for one output วรรค. Returns (syllables, fills) where
    fills is the list of blank captures when the line matches its template. A fill
    identical to the source's own word keeps the source segmentation; any other fill
    is segmented by the dictionary tokenizer."""
    s = re.sub(r"\s+", "", s)
    if ref_line is None:
        return split_syllables(s), None
    m = _template(ref_line).fullmatch(s)
    if m:
        toks, gi = [], 0
        for ui, u in enumerate(ref_line):
            if u:
                toks += u[1]
            else:
                gi += 1
                g = (gold_blanks or {}).get(ui)
                fill = m.group(gi)
                toks += list(g[1]) if g and fill == g[0] else split_syllables(fill)
        return toks, list(m.groups())
    # fallback: keep matched prefix/suffix units, tokenize the middle
    pre, pos, i = [], 0, 0
    while i < len(ref_line) and ref_line[i] and s.startswith(ref_line[i][0], pos):
        nxt = pos + len(ref_line[i][0])
        if nxt < len(s) and s[nxt] in _NO_START:
            break
        pre += ref_line[i][1]
        pos, i = nxt, i + 1
    suf, end, j = [], len(s), len(ref_line) - 1
    while j >= i and ref_line[j] and s[:end].endswith(ref_line[j][0]) \
            and end - len(ref_line[j][0]) >= pos:
        st = end - len(ref_line[j][0])
        if st > 0 and (s[st - 1] in _LEAD_V):
            break
        suf = ref_line[j][1] + suf
        end, j = st, j - 1
    return pre + split_syllables(s[pos:end]) + suf, None


def khlong_to_grader(lines: list[list[str]]) -> list[list[str]]:
    """Move a 2-syllable คำสร้อย written after วรรค 2 or 6 to the optional end slots."""
    if len(lines) <= 8:
        return lines
    core, soi, n = [], [], len(lines)
    for idx, ln in enumerate(lines):
        if len(core) in (2, 6) and len(ln) == 2 and len(soi) < 2 and n - idx > 8 - len(core):
            soi.append(ln)
        else:
            core.append(ln)
    return core + soi


def _order(lines, meter_id):
    return khlong_to_grader(lines) if meter_id == "khlong_si_suphap" else lines


# ---------------------------------------------------------------- grading
def locate_stanza(flat: list[str], ref, given: list[str], skip: int = -1) -> tuple[list[str], bool]:
    """If the output has more วรรค than the stanza (e.g. a verse preface explaining the
    fix), keep the contiguous window of len(ref) วรรค that best matches the prompt's
    stanza (template match or identical line; the faulty วรรค of an edit item is not
    counted, so an echoed input and its correction tie); ties go to the later window."""
    n = len(ref)
    if len(flat) <= n:
        return flat, False
    def hits(win):
        h = 0
        for k, s in enumerate(win):
            if k != skip and (_template(ref[k]).fullmatch(s) or s == given[k]):
                h += 1
        return h
    best_i = max(range(len(flat) - n + 1), key=lambda i: (hits(flat[i:i + n]), i))
    return flat[best_i:best_i + n], True


def grade_one(item: dict, poem: str) -> dict:
    ref = item["reference"]
    mid = item["meter_id"]
    src = item["source_text"].splitlines()
    inp = item["input_poem"].splitlines() if item["task_mode"] == "edit" else None
    gold_blanks: dict[int, dict] = defaultdict(dict)      # display line -> unit -> gold
    for b in item.get("blank_positions", []):
        gold_blanks[item["display_order"].index(b["line"])][b["unit"]] = (
            b["gold_surface"], b["gold_syllables"])
    cands = []
    for name, lines in (("raw", _lines(poem)), ("normalized", _lines(to_vrk_lines(poem)))):
        flat = [re.sub(r"\s+", "", s) for s in lines]
        flat, windowed = locate_stanza(flat, ref, inp or src,
                                       item["fault"]["display_line"] if inp else -1)
        seg, fills = [], []
        for k, s in enumerate(flat):
            toks, f = seg_line(s, ref[k] if k < len(ref) else None, gold_blanks.get(k))
            seg.append(toks)
            fills.append(f)
        anchored = score_meter([tuple(v) for v in _order(seg, mid)], mid)
        tok = score_meter([tuple(v) for v in _order([split_syllables(s) for s in flat], mid)], mid)
        if item["task_mode"] == "cloze":
            faithful = len(flat) == len(ref) and all(
                f is not None and all(x and "_" not in x for x in f) for f in fills)
        else:
            fl = item["fault"]["display_line"]
            faithful = (len(flat) == len(inp)
                        and all(flat[k] == inp[k] for k in range(len(inp)) if k != fl)
                        and flat[fl] != inp[fl])
        cands.append({"layout": name, "windowed": windowed,
                      "meter": anchored["score"], "meter_tok": tok["score"],
                      "faithful": bool(faithful), "exact": flat == src,
                      "unchanged": inp is not None and flat == inp,
                      "solved": bool(faithful) and anchored["score"] >= 0.9999,
                      "layers": anchored["layers"], "rates": anchored["layer_rates"],
                      "fills": fills if item["task_mode"] == "cloze" else None,
                      "n_lines": len(flat)})
    best = max(cands, key=lambda c: (c["solved"], c["faithful"], c["meter"]))
    best = dict(best)
    best["meter"] = max(c["meter"] for c in cands)            # bestof2, as in grade_runs
    best["meter_tok"] = max(c["meter_tok"] for c in cands)
    best["valid"] = best["meter"] >= 0.9999                   # every rule of the stanza holds
    # sensitivity: pure dictionary-tokenizer segmentation (grade_runs' own)
    best["solved_tok"] = any(c["faithful"] and c["meter_tok"] >= 0.9999 for c in cands)
    best["position"] = position_checks(item, best["layers"])
    return best


def position_checks(item: dict, layers: dict) -> list[dict]:
    """Did the output satisfy the rule(s) at the targeted position(s)?"""
    out = []
    if item["task_mode"] == "cloze":
        targets = [(b["line"], b["syllable"], b["constraint"]) for b in item["blank_positions"]]
    else:
        f = item["fault"]
        targets = [(f["line"], f["syllable"], f["kind"])]
    for li, si, label in targets:
        ok_count = next((d[-1] for d in layers["count"]["details"] if d[0] == li), False)
        ok = ok_count
        if label.startswith("tone"):
            ok = ok and any(d[0] == li and d[1] == si and d[-1] for d in layers["tone"]["details"])
        elif label in ("karu", "lahu", "weight"):
            ok = ok and any(d[0] == li and d[1] == si and d[-1] for d in layers["kru_lahu"]["details"])
        elif label == "rhyme":
            spec = M.METER_SPECS_BY_ID[item["meter_id"]]
            idxs = [k for k, rl in enumerate(spec.rhyme_links)
                    if rl.source.line_index == li and
                    (rl.source.syllable_index == si or
                     (rl.source.syllable_index < 0 and si == len(item["source_lines"][li]) + rl.source.syllable_index))]
            ok = ok and all(layers["rhyme"]["details"][k][-1] for k in idxs)
        out.append({"label": label, "ok": bool(ok)})
    return out


def load_run(path: Path) -> list[dict]:
    best: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        good = (not r.get("error")) and bool(r.get("response"))
        prev = best.get(r["item_id"])
        if prev is None or good or not ((not prev.get("error")) and bool(prev.get("response"))):
            best[r["item_id"]] = r
    return list(best.values())


def grade_run(key: str, items: dict) -> list[dict]:
    out = []
    for r in load_run(RUNS / f"{key}__run1.jsonl"):
        it = items.get(r["item_id"])
        if it is None:
            continue
        g = {"model": key, "item_id": it["id"], "task_mode": it["task_mode"],
             "meter_id": it["meter_id"], "source_id": it["source_id"],
             "provenance": it["provenance"], "in_main": it["in_main"],
             "finish_reason": r.get("finish_reason"), "raw_response": r.get("response")}
        if r.get("error"):
            g.update(status="infra_error", error=str(r["error"])[:200])
        else:
            poem = ce_clean(r.get("response") or "")
            g["poem"] = poem
            if not poem:
                g.update(status="model_fail", meter=0.0, meter_tok=0.0, faithful=False,
                         solved=False, valid=False, solved_tok=False, exact=False, position=None)
            else:
                res = grade_one(it, poem)
                g.update(status="graded", **{k: v for k, v in res.items() if k != "layers"})
        out.append(g)
    return out


# ---------------------------------------------------------------- stats
def cluster_ci(recs: list[dict], field: str, rng: random.Random) -> tuple[float, list]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        by[r["source_id"]].append(float(r[field]))
    keys = list(by)
    tot = sum(sum(v) for v in by.values())
    n = sum(len(v) for v in by.values())
    if not n:
        return None, [None, None]
    mean = tot / n
    if len(keys) < 2:
        return round(mean, 4), [None, None]
    boots = []
    for _ in range(N_BOOT):
        s = c = 0
        for _k in range(len(keys)):
            v = by[keys[rng.randrange(len(keys))]]
            s += sum(v)
            c += len(v)
        boots.append(s / c)
    boots.sort()
    return round(mean, 4), [round(boots[int(0.025 * N_BOOT)], 4), round(boots[int(0.975 * N_BOOT) - 1], 4)]


def summarize(recs: list[dict], rng) -> dict:
    ev = [r for r in recs if r["status"] in ("graded", "model_fail")]
    if not ev:
        return None
    meter, meter_ci = cluster_ci(ev, "meter", rng)
    solved, solved_ci = cluster_ci(ev, "solved", rng)
    valid, valid_ci = cluster_ci(ev, "valid", rng)
    return {"n": len(ev), "n_stanzas": len({r["source_id"] for r in ev}),
            "n_infra_excluded": sum(1 for r in recs if r["status"] == "infra_error"),
            "delivery": round(sum(r["status"] == "graded" for r in ev) / len(ev), 4),
            "meter": meter, "meter_ci": meter_ci,
            "solved": solved, "solved_ci": solved_ci,
            "valid": valid, "valid_ci": valid_ci,
            "windowed": round(sum(bool(r.get("windowed")) for r in ev) / len(ev), 4),
            "faithful": round(sum(bool(r["faithful"]) for r in ev) / len(ev), 4),
            "exact": round(sum(bool(r["exact"]) for r in ev) / len(ev), 4),
            "meter_tok": round(sum(r["meter_tok"] for r in ev) / len(ev), 4),
            "solved_tok": round(sum(bool(r["solved_tok"]) for r in ev) / len(ev), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade the cloze/edit runs in cloze_edit/runs/.")
    ap.add_argument("--out", default=str(GRADED),
                    help="output directory (default: cloze_edit/graded_new/; "
                         "cloze_edit/graded/ holds the archived results)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    items = {}
    for mode in ("cloze", "edit"):
        for line in (DATA / f"items_{mode}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                items[it["id"]] = it
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.name.split("__run")[0] for p in RUNS.glob("*__run1.jsonl"))
    rng = random.Random(SEED)
    results = {"models": {}, "provenance": {}, "positions": {}}
    for key in keys:
        recs = grade_run(key, items)
        with (out_dir / f"{key}__run1.jsonl").open("w", encoding="utf-8") as f:
            for g in recs:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        res = {}
        for mode in ("cloze", "edit"):
            main = [r for r in recs if r["task_mode"] == mode and r["in_main"]]
            res[mode] = {"all": summarize(main, rng)}
            for form in FORMS:
                s = summarize([r for r in main if r["meter_id"] == form], rng)
                if s:
                    res[mode][form] = s
        results["models"][key] = res
        prov = {}
        for mode in ("cloze", "edit"):
            for form in FORMS:
                for p in ("student", "model"):
                    s = summarize([r for r in recs if r["task_mode"] == mode and r["meter_id"] == form
                                   and r["provenance"] == p], rng)
                    if s:
                        prov[f"{mode}|{form}|{p}"] = s
        results["provenance"][key] = prov
        pos = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["in_main"] and r["status"] in ("graded", "model_fail"):
                for pc in (r.get("position") or [{"label": "nondelivery", "ok": False}]):
                    k = f"{r['task_mode']}|{r['meter_id']}|{pc['label']}"
                    pos[k][0] += pc["ok"]
                    pos[k][1] += 1
        results["positions"][key] = {k: v for k, v in pos.items()}
        msg = f"{key:22}"
        for mode in ("cloze", "edit"):
            a = res[mode]["all"]
            if a:
                msg += (f" | {mode} n={a['n']} dlv={a['delivery']} meter={a['meter']} "
                        f"valid={a['valid']} solved={a['solved']} faithful={a['faithful']} "
                        f"win={a['windowed']}")
        print(msg)
    (out_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
