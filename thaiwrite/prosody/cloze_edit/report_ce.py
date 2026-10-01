"""Collect the cloze/edit results into results.json + results.md + a LaTeX table.

Reads the graded cloze/edit outputs (--graded, default cloze_edit/graded/, the archived
grade_ce.py results; pass cloze_edit/graded_new/ to report a fresh regrade),
data/pool.json, data/items_*.jsonl and the API spend (cost_ce.py). Writes to --out
(default: thaiwrite/prosody/cloze_edit/report/).

The composition columns (meter and valid-stanza share per model) are the paper's
composition results, which reproduce/make_paper_tables.py rebuilds from eval/graded
and the syllable count-of-record cache; they are listed in MODELS below.

  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/report_ce.py
  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/report_ce.py --graded thaiwrite/prosody/cloze_edit/graded_new --out /tmp/ce_report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from thaiwrite.prosody.cloze_edit import cost_ce                        # noqa: E402
from thaiwrite.prosody.cloze_edit.grade_ce import cluster_ci            # noqa: E402

REPO = HERE.parents[2]
DEFAULT_OUT = HERE / "report"
DEFAULT_GRADED = HERE / "graded"                  # archived grade_ce.py output
FORMS = ["khlong_si_suphap", "klon_paet", "kap_yani_11", "intharawichian_chan_11"]
SHORT = {"khlong_si_suphap": "Khlong", "klon_paet": "Klon", "kap_yani_11": "Kap",
         "intharawichian_chan_11": "Chan"}
#: display order + labels (composition-leaderboard order). Composition meter and
#: valid-stanza share are the paper's composition results (rebuilt by
#: reproduce/make_paper_tables.py from eval/graded + the count-of-record cache).
#: (key, label, compose meter, compose valid-stanza %, in the appendix table)
MODELS = [
    ("gpt55-med", "GPT-5.5 (medium)", 0.895, 58.1, True),
    ("gemini-3.5-flash", "Gemini 3.5 Flash", 0.819, 37.5, True),
    ("gemma-4-31b", "Gemma 4 31B", 0.728, 25.0, True),
    ("deepseek-v4-flash", "DeepSeek V4 Flash", 0.415, 3.8, True),
    ("typhoon-s", r"Typhoon-S 8B \textsc{th}", 0.258, 0.0, True),
    ("openthaigpt", r"OpenThaiGPT 8B \textsc{th}", 0.246, 0.0, True),
    ("qwen3-8b", "Qwen3 8B", 0.233, 0.6, True),
    ("typhoon25", r"Typhoon 2.5 30B \textsc{th}", 0.219, 0.0, True),
    # supplementary configurations (results.md only)
    ("deepseek-v4-pro", "DeepSeek V4 Pro (DeepSeek API)", 0.494, 8.9, False),
    ("gemma-4-31b-think", "Gemma 4 31B, thinking (Google AI API)", 0.728, 25.0, False),
]
FRONTIER = {"gpt55-med", "gemini-3.5-flash", "gemma-4-31b", "deepseek-v4-flash"}
THAI = {"openthaigpt", "typhoon-s", "typhoon25"}


def pooled(recs, field, rng):
    m, ci = cluster_ci(recs, field, rng)
    return {"n": len(recs), "n_stanzas": len({r["source_id"] for r in recs}), field: m,
            f"{field}_ci": ci}


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect cloze/edit results into results.{json,md} "
                                             "and table_rows.tex.")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory (default: thaiwrite/prosody/cloze_edit/report/)")
    ap.add_argument("--graded", default=str(DEFAULT_GRADED),
                    help="directory with grade_ce.py output (default: cloze_edit/graded/, archived)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gdir = Path(args.graded)
    S = json.loads((gdir / "summary.json").read_text(encoding="utf-8"))
    pool = json.loads((HERE / "data" / "pool.json").read_text(encoding="utf-8"))
    items = [json.loads(x) for m in ("cloze", "edit")
             for x in (HERE / "data" / f"items_{m}.jsonl").read_text(encoding="utf-8").splitlines()
             if x.strip()]
    graded = {k: [json.loads(x) for x in (gdir / f"{k}__run1.jsonl")
                  .read_text(encoding="utf-8").splitlines() if x.strip()]
              for k, *_ in MODELS if (gdir / f"{k}__run1.jsonl").exists()}
    rng = random.Random(7)

    res = {"sources": {
        "pool": "thaiwrite/prosody/cloze_edit/data/pool.json (+ pool_report.md)",
        "items": "thaiwrite/prosody/cloze_edit/data/items_{cloze,edit}.jsonl",
        "raw_runs": "thaiwrite/prosody/cloze_edit/runs/<model>__run1.jsonl",
        "graded": "thaiwrite/prosody/cloze_edit/graded/<model>__run1.jsonl + summary.json",
        "compose": "paper composition results (reproduce/make_paper_tables.py): "
                   "meter and valid-stanza %"}}
    # ---- pool / item counts
    pc = Counter((r["meter_id"], r["provenance"], r["in_main"]) for r in pool)
    res["pool"] = {f"{SHORT[f]}|{p}|{'main' if m else 'control'}": n
                   for (f, p, m), n in sorted(pc.items())}
    ic = Counter((it["task_mode"], it["meter_id"], it["provenance"], it["in_main"]) for it in items)
    res["items"] = {f"{mo}|{SHORT[f]}|{p}|{'main' if m else 'control'}": n
                    for (mo, f, p, m), n in sorted(ic.items())}
    # ---- per model
    res["models"] = {}
    for key, disp, cmeter, cvalid, in_table in MODELS:
        if key not in S["models"]:
            continue
        res["models"][key] = {"display": disp, "in_table": in_table,
                              "compose_meter": cmeter, "compose_valid_pct": cvalid,
                              **S["models"][key],
                              "provenance": S["provenance"].get(key, {}),
                              "positions": S["positions"].get(key, {})}
    # ---- provenance, pooled over model tiers (all items incl. control stanzas)
    prov = {}
    for tier, keys in (("frontier", FRONTIER), ("thai", THAI)):
        for mode in ("cloze", "edit"):
            for f in FORMS:
                for p in ("student", "model"):
                    recs = [r for k in keys if k in graded for r in graded[k]
                            if r["task_mode"] == mode and r["meter_id"] == f and r["provenance"] == p
                            and r["status"] in ("graded", "model_fail")]
                    if recs:
                        prov[f"{tier}|{mode}|{SHORT[f]}|{p}"] = {
                            **pooled(recs, "solved", rng),
                            "valid": round(sum(r["valid"] for r in recs) / len(recs), 4),
                            "faithful": round(sum(r["faithful"] for r in recs) / len(recs), 4)}
    res["provenance_pooled"] = prov
    # ---- positions pooled over tiers
    posp = {}
    for tier, keys in (("frontier", FRONTIER), ("thai", THAI)):
        acc = defaultdict(lambda: [0, 0])
        for k in keys:
            for kk, (a, b) in S["positions"].get(k, {}).items():
                acc[kk][0] += a
                acc[kk][1] += b
        posp[tier] = {kk: {"ok": a, "n": b, "rate": round(a / b, 4)} for kk, (a, b) in sorted(acc.items())}
    res["positions_pooled"] = posp
    res["spend_usd"] = cost_ce.main()
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res, out)
    write_tex_table(res, out)
    print(f"wrote {out}")


def _f(x, pct=True, nd=0):
    if x is None:
        return "—"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.3f}"


def write_md(res, out: Path) -> None:
    L = ["# Cloze and edit results", "",
         "All numbers from `thaiwrite/prosody/cloze_edit/` in the release repository "
         "(scripts: `build_pool.py`, `make_items.py`, `run_ce.py`, `grade_ce.py`, `report_ce.py`, "
         "`cost_ce.py`). Main-pool items only unless stated. Rates in %, CIs = 95% cluster "
         "bootstrap over source stanzas.", "",
         "## Source pool (stanzas)", "", "| form | provenance | set | stanzas |", "|---|---|---|--:|"]
    for k, n in res["pool"].items():
        f, p, s = k.split("|")
        L.append(f"| {f} | {p} | {s} | {n} |")
    L += ["", "## Items", "", "| mode | form | provenance | set | items |", "|---|---|---|---|--:|"]
    for k, n in res["items"].items():
        mo, f, p, s = k.split("|")
        L.append(f"| {mo} | {f} | {p} | {s} | {n} |")
    for mode in ("cloze", "edit"):
        forms = FORMS if mode == "cloze" else ["khlong_si_suphap", "intharawichian_chan_11"]
        L += ["", f"## {mode.capitalize()}: per model (all forms pooled, then per form)", "",
              "| model | n | stanzas | deliv | meter [CI] | meter (tokenizer only) | valid [CI] | solved [CI] | solved (tokenizer only) | faithful | exact |",
              "|---|--:|--:|--:|---|--:|---|---|--:|--:|--:|"]
        for key, m in res["models"].items():
            a = m[mode]["all"]
            if not a:
                continue
            L.append(f"| {key} | {a['n']} | {a['n_stanzas']} | {_f(a['delivery'])} | "
                     f"{_f(a['meter'], False)} [{_f(a['meter_ci'][0], False)}, {_f(a['meter_ci'][1], False)}] | "
                     f"{_f(a['meter_tok'], False)} | {_f(a['valid'])} [{_f(a['valid_ci'][0])}, {_f(a['valid_ci'][1])}] | "
                     f"{_f(a['solved'])} [{_f(a['solved_ci'][0])}, {_f(a['solved_ci'][1])}] | "
                     f"{_f(a['solved_tok'])} | {_f(a['faithful'])} | {_f(a['exact'])} |")
        L += ["", "| model | " + " | ".join(f"{SHORT[f]} n | {SHORT[f]} valid | {SHORT[f]} solved [CI]" for f in forms) + " |",
              "|---|" + "--:|--:|---|" * len(forms)]
        for key, m in res["models"].items():
            cells = []
            for f in forms:
                c = m[mode].get(f)
                cells.append("— | — | —" if not c else
                             f"{c['n']} | {_f(c['valid'])} | {_f(c['solved'])} [{_f(c['solved_ci'][0])}, {_f(c['solved_ci'][1])}]")
            L.append(f"| {key} | " + " | ".join(cells) + " |")
    L += ["", "## Composition reference", "",
          "Paper composition results (rebuilt by `reproduce/make_paper_tables.py`; syllable "
          "counts from the count-of-record, `count_api.py`): meter and valid-stanza share (all "
          "rules satisfied, non-delivery = invalid).", "",
          "| model | in appendix table | compose meter | compose valid (%) |", "|---|---|--:|--:|"]
    for key, m in res["models"].items():
        L.append(f"| {key} | {'yes' if m['in_table'] else 'no (supplementary)'} | "
                 f"{m['compose_meter']:.3f} | {m['compose_valid_pct']:.1f} |")
    L += ["", "## Provenance check (student vs model-composed stanzas; main + control items)", "",
          "| tier | mode | form | provenance | items | stanzas | valid | solved [CI] | faithful |",
          "|---|---|---|---|--:|--:|--:|---|--:|"]
    for k, v in res["provenance_pooled"].items():
        t, mo, f, p = k.split("|")
        L.append(f"| {t} | {mo} | {f} | {p} | {v['n']} | {v['n_stanzas']} | {_f(v['valid'])} | "
                 f"{_f(v['solved'])} [{_f(v['solved_ci'][0])}, {_f(v['solved_ci'][1])}] | {_f(v['faithful'])} |")
    L += ["", "Per model:", "", "| model | cell | items | valid | solved | faithful |", "|---|---|--:|--:|--:|--:|"]
    for key, m in res["models"].items():
        for k, v in m["provenance"].items():
            L.append(f"| {key} | {k} | {v['n']} | {_f(v['valid'])} | {_f(v['solved'])} | {_f(v['faithful'])} |")
    L += ["", "## Targeted positions (was the rule at the blanked / faulted position satisfied?)", "",
          "| tier | mode | form | constraint | ok / n | rate |", "|---|---|---|---|--:|--:|"]
    for t, d in res["positions_pooled"].items():
        for k, v in d.items():
            mo, f, lab = k.split("|")
            L.append(f"| {t} | {mo} | {SHORT[f]} | {lab} | {v['ok']} / {v['n']} | {_f(v['rate'])} |")
    L += ["", "## API spend (USD)", "", "| model | calls | input tok | output tok | USD |", "|---|--:|--:|--:|--:|"]
    for k, v in res["spend_usd"].items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | {v['calls']} | {v['input_tokens']:,} | {v['output_tokens']:,} | {v['usd']:.2f} |")
    L.append(f"| total | | | | {res['spend_usd']['_total_usd']:.2f} |")
    L += ["", "## Notes on the runs", "",
          "- Grader: all runs graded with grader_meter.py as released (ไ/ใ rhyme fallback, "
          "open สระออ = ครุ, บ/บ่ = ลหุ). Every pool stanza still scores 1.0 and every item passes "
          "the sanity checks (source solves, edit input is invalid) under it; no excluded student "
          "stanza newly qualifies.",
          "- Table models, all with their composition registry settings: gpt55-med (Responses API), "
          "gemini-3.5-flash (Google OpenAI-compat), typhoon25 (OpenTyphoon), typhoon-s and "
          "openthaigpt (ThaiLLM gateway), and via OpenRouter gemma-4-31b (non-thinking), qwen3-8b "
          "(native thinking) and deepseek-v4-flash (native thinking). OpenRouter serves "
          "deepseek-v4-flash without reasoning unless requested, so reasoning is requested explicitly "
          "({'enabled': true}) to match the composition run (median ~9k output tokens there).",
          "- Supplementary (not in the table): deepseek-v4-pro via the official DeepSeek API "
          "(27 Khlong editing items, ce-khlong-edit-022..048, are infrastructure errors because "
          "the API account balance was exhausted; spend = billed amount) and gemma-4-31b-think "
          "via the Google AI API "
          "(thinking cannot be disabled there; 16k budget).",
          "- Gemma-think spend is an upper bound at OpenRouter's Gemma price.",
          "- deepseek-v4-flash: one item (ce-khlong-cloze-001) was answered twice (probe + full "
          "run); grading keeps the later successful response. Token run-outs were not retried "
          "(one attempt per item, as in composition).",
          "- Tokenizer caveat: the dictionary tokenizer mis-segments 18 of the 256 gold fills "
          "(7%, mostly respelled or Pali-Sanskrit syllables such as รุณ, ขณะ). Exact restorations "
          "keep the gold segmentation; novel fills are tokenized and may be under-credited.",
          "- Pool caveat: several model-composed Chan stanzas use rare readings/words "
          "(ชล read ชะ-ละ, นภ read นะ-ภะ, วนี, filler ผิ) that models often 'correct', lowering "
          "Chan fidelity; this is part of the reported Chan numbers."]
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_tex_table(res, out: Path) -> None:
    rows = []
    for key, m in res["models"].items():
        if not m["in_table"]:
            continue
        c, e = m["cloze"], m["edit"]
        def cell(d, f, fld="solved"):
            x = d.get(f)
            return "--" if not x else f"{round(100 * x[fld]):d}"
        ca, ea = c["all"], e["all"]
        if not ca or not ea:
            continue
        rows.append(
            f"{m['display']:<32} & {m['compose_meter']:.3f} & {m['compose_valid_pct']:.1f} & "
            + " & ".join(cell(c, f) for f in FORMS)
            + f" & {round(100 * ca['solved']):d} \\scriptsize[{round(100 * ca['solved_ci'][0])},{round(100 * ca['solved_ci'][1])}] & "
            + " & ".join(cell(e, f) for f in ("khlong_si_suphap", "intharawichian_chan_11"))
            + f" & {round(100 * ea['solved']):d} \\scriptsize[{round(100 * ea['solved_ci'][0])},{round(100 * ea['solved_ci'][1])}]"
            + f" & {round(100 * ea['valid']):d} \\\\")
    (out / "table_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
