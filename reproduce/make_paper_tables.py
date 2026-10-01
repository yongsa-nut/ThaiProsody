"""Reproduce every table of the ThaiProsody paper from the released files, with no API calls.

    PYTHONIOENCODING=utf-8 python reproduce/make_paper_tables.py
    python reproduce/make_paper_tables.py --cache CACHE1.json [CACHE2.json ...] --judge DIR1 [DIR2 ...]
    python reproduce/make_paper_tables.py --graded-dir thaiwrite/prosody/eval/graded NEW_DIR --out OUT_DIR

Inputs (all inside this repository; nothing is written to them):

  thaiwrite/prosody/eval/graded/<config>__run1.jsonl  (--graded-dir, one or more directories)
        graded composition outputs, one file per configuration (native, `.nothink` =
        thinking disabled, `.rules` = rules in the prompt). Records that are not
        composition items are ignored. A configuration present in several directories is
        read from the first one listed.
  thaiwrite/prosody/eval/graded/replication/<config>__run{1,2,3}.jsonl
        three independent generation passes for the replication table.
  --cache   syllable count-of-record caches (piece -> {"count": int, ...}), produced by
            thaiwrite/prosody/count_api.py. Several caches may be given; for a piece present
            in more than one, the first cache listed wins; a missing file is skipped. Default:
            thaiwrite/prosody/vrk_count_cache_opus55_main.json (composition runs, native and
            rules in the prompt) and vrk_count_cache_opus55_bcd.json (thinking-disabled runs,
            replication runs, cloze and edit outputs).
  --judge   content-judge directories with <config>.jsonl rows {item_id, topicality,
            coherence[, poem][, unjudgeable]} (and optionally inputs/<config>.jsonl with the
            judged poem text), produced by thaiwrite/prosody/judge_api.py. A row marked
            unjudgeable (the judge declined the poem) counts as neither on-topic nor coherent. For each item the first
            directory listed that holds a judgment of the same poem text wins. Default:
            thaiwrite/prosody/eval/judged_api_all and eval/judged_api (Claude Opus 5.5, the
            judge of the paper). eval/judged_archive holds a second judge (Claude Opus 4.7, same
            rubric) and is used only when listed. tables.md lists which directory judged each
            configuration.
  thaiwrite/prosody/calibration/
        calibration items and rhyme pairs (calibration_items_v2.json), the expert's labels
        (labels/labels_expertA.json), and judge scores on the calibration poems
        (judge_cache_opus.json = Claude Opus 4.7, judge_cache_gpt55.json = GPT-5.5,
        judge_api_calib_<model>.json = output of `judge_api.py calib`), for the
        rhyme-checker and judge-agreement tables.

Scoring (as in the paper): composition items only (160 = 4 forms x 40 topics). Meter score =
weighted mean of the dimensions a form binds (grader_meter.LAYER_WEIGHTS) with the tone, weight
and rhyme layers as stored by the grader (best-of-2 layout) and the count layer recomputed from
the count-of-record on the stored layout: a line is counted as the sum of the cached counts of
its whitespace-separated pieces. Non-delivery (empty output, token runout) scores 0.
Infrastructure errors are excluded from n. 95% CIs: item bootstrap, 1000 resamples, seed
20260524. A piece missing from every cache falls back to the dictionary tokenizer and is
reported (tables.md, uncounted_pieces.json). Valid stanza = every bound check passes.
Top./Coh. = share of the evaluated items judged on-topic (topicality >= 3) and coherent
(coherence >= 2); a non-delivery counts as neither; a row is left blank (--) while any
delivered poem of the configuration has no judgment.

Outputs, in --out (default reproduce/out/): tables.md, tables.tex, numbers.json,
uncounted_pieces.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
PROS = ROOT / "thaiwrite" / "prosody"
GRADED = PROS / "eval" / "graded"
DATA = PROS / "data"
CAL = PROS / "calibration"
LABELS_A = CAL / "labels" / "labels_expertA.json"
DEFAULT_CACHES = [PROS / "vrk_count_cache_opus55_main.json", PROS / "vrk_count_cache_opus55_bcd.json"]
DEFAULT_JUDGE = [PROS / "eval" / "judged_api_all", PROS / "eval" / "judged_api"]
DEFAULT_OUT = ROOT / "reproduce" / "out"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CAL))
from thaiwrite.prosody.grader_meter import LAYER_WEIGHTS, rhymes, split_syllables  # noqa: E402
from thaiwrite.prosody import meter_specs as M                                      # noqa: E402
from thaiwrite.prosody.eval.promptlib import to_vrk_lines                           # noqa: E402
from thaiwrite.prosody.data.topics import TOPICS                                    # noqa: E402
from score_calibration import krippendorff_alpha                                    # noqa: E402

SEED = 20260524
N_BOOT = 1000
FORMS = ["khlong_si_suphap", "klon_paet", "kap_yani_11", "intharawichian_chan_11"]
FORM_EN = {"khlong_si_suphap": "Khlong", "klon_paet": "Klon", "kap_yani_11": "Kap",
           "intharawichian_chan_11": "Chan"}
DIM_EN = {"count": "count", "tone": "tone", "kru_lahu": "weight", "rhyme": "rhyme"}
CAT_EN = {"ธรรมชาติ": "nature", "นามธรรม": "abstract", "อารมณ์": "emotion",
          "ชีวิต": "everyday life", "ฤดูกาล": "seasons", "เทศกาล": "festivals"}
CAT_ORDER = ["nature", "abstract", "emotion", "everyday life", "seasons", "festivals"]
DISPLAY = {
    "gpt55-high": "GPT-5.5 (high)", "gpt55-med": "GPT-5.5 (medium)", "gpt55-low": "GPT-5.5 (low)",
    "gemini-3.5-flash": "Gemini 3.5 Flash", "gemma-4-31b": "Gemma 4 31B",
    "opus-4.7": "Claude Opus 4.7", "gpt54-mini": "GPT-5.4-mini",
    "deepseek-v4-pro": "DeepSeek V4 Pro", "deepseek-v4-flash": "DeepSeek V4 Flash",
    "sonnet-4.6": "Claude Sonnet 4.6", "minimax-m2.7": "MiniMax M2.7",
    "openthaigpt": "OpenThaiGPT 8B", "typhoon-s": "Typhoon-S 8B", "typhoon25": "Typhoon 2.5 30B",
    "thalle": "THaLLE 8B", "kimi-k2.6": "Kimi K2.6", "qwen3.6-35b-a3b": "Qwen3.6 35B-A3B",
    "glm-5.1": "GLM 5.1", "qwen3-8b": "Qwen3-8B",
}
THAI_TRAINED = {"openthaigpt", "typhoon-s", "typhoon25", "thalle"}
# Strong composers (per-dimension error table and topic categories).
STRONG = ["gpt55-high", "gpt55-med", "gpt55-low", "gemini-3.5-flash", "opus-4.7",
          "gemma-4-31b", "gpt54-mini"]
REPLICATION = ["gpt55-med", "gemini-3.5-flash", "gemma-4-31b", "gpt54-mini",
               "deepseek-v4-flash", "openthaigpt", "typhoon-s", "qwen3-8b"]
TEX_MIN_N = 5          # partial configurations with fewer evaluated items are not tabulated

CACHE: dict = {}


# ============================================================================ helpers
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    """Percentile item-bootstrap 95% CI of the mean (same routine as eval/grade_runs.py)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return [None, None]
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))

    def pct_(p):
        return means[min(n_boot - 1, max(0, round(p / 100 * (n_boot - 1))))]
    return [pct_(2.5), pct_(97.5)]


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 2:
        return None
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** .5
    vb = sum((y - mb) ** 2 for y in rb) ** .5
    return cov / (va * vb) if va and vb else None


def cohen_kappa(pairs):
    n = len(pairs)
    if not n:
        return None, None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, ((po - pe) / (1 - pe) if pe != 1 else None)


def weighted_kappa(pairs, cats=(1, 2, 3, 4, 5), weight="quadratic"):
    n = len(pairs)
    if not n:
        return None
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = [[0.0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1
    ra = [sum(obs[i]) for i in range(k)]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        d = abs(i - j) / (k - 1)
        return d * d if weight == "quadratic" else d
    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else None


def flips(ks, x, y):
    """Pairs of configurations whose order differs between score dicts x and y."""
    return [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:] if (x[a] - x[b]) * (y[a] - y[b]) < 0]


def hu(x, nd=3):
    """Round half-up from full precision."""
    if x is None:
        return "--"
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def dot3(x):
    s = hu(x, 3)
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)


def pct(x, nd=0):
    return "--" if x is None else hu(100 * x, nd)


def sgn(x, nd=3):
    if x is None:
        return "--"
    if float(hu(x, nd)) == 0:
        return hu(0.0, nd)
    s = hu(x, nd)
    return s if s.startswith("-") else "+" + s


def ci_s(ci):
    return "--" if not ci or ci[0] is None else f"[{dot3(ci[0])},{dot3(ci[1])}]"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def resolve(p: str) -> Path:
    """A CLI path: as given (absolute or relative to the working directory), else relative to
    the repository root, else relative to thaiwrite/prosody/."""
    q = Path(p)
    if q.is_absolute() or q.exists():
        return q
    for base in (ROOT, PROS):
        if (base / q).exists():
            return base / q
    return q


# ============================================================================ data
def load_compose_items():
    items = {}
    for s in ("public", "private"):
        for line in (DATA / f"prompts_compose_{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                items[it["id"]] = it
    return items


COMPOSE = load_compose_items()
COMPOSE_IDS = list(COMPOSE)
TOPIC_CAT = {t: CAT_EN[c] for c, t in TOPICS}


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def status_of(r):
    st = r.get("eval_status")
    if st:
        return st
    if r.get("meter_score") is not None:
        return "graded"
    return "model_fail" if r.get("gen_error") == "empty_output" else "infra_error"


def config_variant(key):
    for v in ("nothink", "rules"):
        if key.endswith("." + v):
            return key[: -len(v) - 1], v
    return key, "native"


def display(key):
    base, var = config_variant(key)
    d = DISPLAY.get(base, base)
    return d if var == "native" else f"{d} [{'thinking disabled' if var == 'nothink' else 'rules in prompt'}]"


def tex_name(k):
    base, _ = config_variant(k)
    return DISPLAY.get(base, base) + (r" \textsc{th}" if base in THAI_TRAINED else "")


# ------------------------------------------------------------------ per-record analysis
def cached_count(piece):
    c = CACHE.get(piece)
    if c is None:
        return None
    n = c.get("count")
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def graded_units(poem, layout):
    """The lines the grader scored, in the layout it chose."""
    if layout == "raw":
        return [ln.strip() for ln in poem.splitlines() if ln.strip()]
    return [v.strip() for v in to_vrk_lines(poem).splitlines() if v.strip()]


def count_pt(counts, n_units, meter_id):
    """(passed, total) of the count layer."""
    spec = M.METER_SPECS_BY_ID[meter_id]
    passed = total = 0
    for i, ls in enumerate(spec.lines):
        if ls.optional and i >= n_units:
            continue
        total += 1
        n = counts[i] if i < len(counts) else 0
        passed += ls.min_count <= n <= ls.max_count
    return passed, total


def combine(rates, meter_id):
    num = den = 0.0
    for name, w in LAYER_WEIGHTS[meter_id].items():
        r = rates.get(name)
        if r is None:
            continue
        num += w * r
        den += w
    return num / den if den else 0.0


def _full(pt):
    return pt[1] > 0 and pt[0] == pt[1]


def analyze(r):
    """Scores of one graded record: `cor` (count-of-record count) and `tok` (the grader's own
    dictionary-tokenizer count, as stored)."""
    it = COMPOSE[r["item_id"]]
    mid = it["meter_id"]
    a = {"item_id": r["item_id"], "status": status_of(r), "meter_id": mid,
         "category": TOPIC_CAT.get(it["topic"], "?")}
    if a["status"] != "graded":
        return a
    nslots = len(M.METER_SPECS_BY_ID[mid].lines)
    rates = dict(r.get("layer_rates") or {})
    layers = r.get("layers") or {}
    poem = r.get("poem") or ""
    units = graded_units(poem, r.get("layout"))
    counts, miss = [], []
    for u in units[:nslots]:
        tot = 0
        for p in re.split(r"\s+", u):
            if not p:
                continue
            c = cached_count(p)
            if c is None:
                miss.append(p)
                c = len(split_syllables(p))
            tot += c
        counts.append(tot)
    pf, tf = count_pt(counts, len(units), mid)
    bound = list(LAYER_WEIGHTS[mid])
    lc = layers.get("count") or {"passed": 0, "total": 0}
    checks = {"count": (pf, tf), "count_tok": (lc["passed"], lc["total"])}
    for d in ("tone", "rhyme", "kru_lahu"):
        if d in bound:
            L = layers.get(d) or {"passed": 0, "total": 0}
            checks[d] = (L["passed"], L["total"])
    other_full = all(_full(checks[d]) for d in bound if d != "count")
    rc = dict(rates)
    rc["count"] = pf / tf if tf else 0.0

    def eq(count_key):
        rs = [checks[count_key][0] / checks[count_key][1] if checks[count_key][1] else 0.0]
        rs += [checks[d][0] / checks[d][1] if checks[d][1] else 0.0 for d in bound if d != "count"]
        return sum(rs) / len(rs)
    a.update({"bound": bound, "checks": checks,
              "cor": combine(rc, mid), "tok": r.get("meter_score") or 0.0,
              "valid": _full((pf, tf)) and other_full,
              "valid_tok": _full(checks["count_tok"]) and other_full,
              "eq": eq("count"), "eq_tok": eq("count_tok"), "miss": miss,
              "rhyme_links": [bool(d[3]) for d in (layers.get("rhyme") or {}).get("details", [])]})
    return a


def load_config(path: Path):
    recs = [r for r in read_jsonl(path) if r.get("item_id") in COMPOSE]
    seen, A, raw = set(), [], {}
    for r in recs:
        if r["item_id"] in seen:
            continue
        seen.add(r["item_id"])
        raw[r["item_id"]] = r
        A.append(analyze(r))
    return {"key": path.name.split("__run")[0], "path": path, "records": A, "raw": raw}


# ------------------------------------------------------------------ judge
JUDGE_DIRS: list = []


def _poem_rows(path: Path):
    out = {}
    if path.exists():
        for o in read_jsonl(path):
            out[str(o["item_id"])] = o
    return out


def load_judge(key):
    """item_id -> list of candidate judgments (dir order), each {'top','coh','poem' or None}."""
    cand = defaultdict(list)
    for d in JUDGE_DIRS:
        rows = _poem_rows(d / f"{key}.jsonl")
        if not rows:
            continue
        inputs = _poem_rows(d / "inputs" / f"{key}.jsonl")
        for iid, o in rows.items():
            poem = o.get("poem")
            if poem is None and iid in inputs:
                poem = inputs[iid].get("poem")
            if o.get("unjudgeable"):
                # the judge declined the poem: it counts as neither on-topic nor coherent
                cand[iid].append({"top": 0, "coh": 0, "poem": poem, "dir": str(d), "unjudgeable": True})
            else:
                cand[iid].append({"top": o.get("topicality"), "coh": o.get("coherence"),
                                  "poem": poem, "dir": str(d)})
    return cand


def judge_rates(cfg, ids=None):
    cand = load_judge(cfg["key"])
    top, coh, missing, stale, used, unjudgeable = [], [], [], [], Counter(), []
    for a in cfg["records"]:
        if ids is not None and a["item_id"] not in ids:
            continue
        if a["status"] == "infra_error":
            continue
        if a["status"] != "graded":
            top.append(0)
            coh.append(0)
            continue
        graded_poem = (cfg["raw"][a["item_id"]].get("poem") or "").strip()
        pick, was_stale = None, False
        for c in cand.get(a["item_id"], []):
            if not isinstance(c["top"], (int, float)) or not isinstance(c["coh"], (int, float)):
                continue
            if c["poem"] is not None and c["poem"].strip() != graded_poem:
                was_stale = True
                continue
            pick = c
            break
        if pick is None:
            (stale if was_stale else missing).append(a["item_id"])
            continue
        used[pick["dir"]] += 1
        if pick.get("unjudgeable"):
            unjudgeable.append(a["item_id"])
        top.append(1 if pick["top"] >= 3 else 0)
        coh.append(1 if pick["coh"] >= 2 else 0)
    unj = missing + stale
    n = len(top) + len(unj)
    ok = not unj and n > 0
    return {"n": n, "unjudged": len(unj), "missing": missing, "stale": stale, "unjudgeable": unjudgeable,
            "sources": dict(used),
            "top": sum(top) / n if ok else None, "coh": sum(coh) / n if ok else None,
            "top_judged_subset": sum(top) / len(top) if top else None,
            "coh_judged_subset": sum(coh) / len(coh) if coh else None}


# ------------------------------------------------------------------ aggregation
def aggregate(cfg, basis="cor", valid_key="valid", ids=None):
    A = [a for a in cfg["records"] if ids is None or a["item_id"] in ids]
    graded = [a for a in A if a["status"] == "graded"]
    mf = [a for a in A if a["status"] not in ("graded", "infra_error")]
    infra = [a for a in A if a["status"] == "infra_error"]
    universe = set(ids) if ids is not None else set(COMPOSE)
    absent = sorted(universe - {a["item_id"] for a in A})
    scores = [a[basis] if a["status"] == "graded" else 0.0 for a in A if a["status"] != "infra_error"]
    n_eval = len(graded) + len(mf)
    valid = sum(1 for a in graded if a[valid_key])
    miss = [p for a in graded for p in a["miss"]]
    return {"n_eval": n_eval, "n_graded": len(graded), "n_model_fail": len(mf),
            "n_infra": len(infra), "n_absent": len(absent),
            "delivery": len(graded) / n_eval if n_eval else None,
            "meter": mean(scores), "ci": bootstrap_ci(scores),
            "valid": valid, "valid_rate": valid / n_eval if n_eval else None,
            "complete": n_eval == len(universe) and not infra and not absent,
            "uncounted_pieces": len(set(miss))}


def dim_stats(A, count_key="count"):
    """Per bound dimension over delivered poems: mean per-poem pass rate and pooled error rate."""
    out = {}
    for d in ("count", "tone", "kru_lahu", "rhyme"):
        key = count_key if d == "count" else d
        rates, p, t = [], 0, 0
        for a in A:
            if a["status"] != "graded" or d not in a["bound"]:
                continue
            pp, tt = a["checks"][key]
            if not tt:
                continue
            rates.append(pp / tt)
            p += pp
            t += tt
        if rates:
            out[d] = {"poem_mean": sum(rates) / len(rates), "check_pass": p / t, "poems": len(rates)}
    return out


def mean_basis(cfg, basis, cond=lambda a: True):
    xs = [a[basis] if a["status"] == "graded" else 0.0 for a in cfg["records"]
          if a["status"] != "infra_error" and cond(a)]
    return mean(xs)


def count_basis(cfgs_list):
    """'cor' if the count-of-record covers every piece these configurations need, else 'tok'."""
    return "cor" if all(not a["miss"] for c in cfgs_list for a in c["records"] if a["status"] == "graded") else "tok"


BASIS_NAME = {"cor": "count-of-record", "tok": "dictionary-tokenizer count"}


# ============================================================================ main
def main():
    global CACHE, JUDGE_DIRS  # noqa: PLW0603
    ap = argparse.ArgumentParser(description="Reproduce the ThaiProsody paper tables (no API calls).")
    ap.add_argument("--cache", nargs="+", default=[str(p) for p in DEFAULT_CACHES],
                    help="count-of-record caches; the first cache listed wins for a piece in several")
    ap.add_argument("--judge", nargs="*", default=[str(p) for p in DEFAULT_JUDGE],
                    help="content-judge directories; for each item the first directory listed wins")
    ap.add_argument("--graded-dir", nargs="+", default=[str(GRADED)],
                    help="directories of graded composition files (default thaiwrite/prosody/eval/graded); "
                         "a configuration found in several is read from the first directory listed")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (default reproduce/out)")
    args = ap.parse_args()

    caches_used = []
    for c in args.cache:
        p = resolve(c)
        if not p.exists():
            print(f"[warn] count cache not found, skipped: {p}")
            continue
        n0 = len(CACHE)
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            CACHE.setdefault(k, v)
        caches_used.append({"path": str(p), "added": len(CACHE) - n0})
    if not caches_used:
        sys.exit("No count-of-record cache found. Pass --cache <file.json> (see README).")
    JUDGE_DIRS = []
    for d in args.judge:
        p = resolve(d)
        if p.is_dir():
            JUDGE_DIRS.append(p)
        else:
            print(f"[warn] judge directory not found, skipped: {p}")
    graded_dirs = [resolve(g) for g in args.graded_dir]
    out_dir = Path(args.out)
    if any(out_dir.resolve() == g.resolve() for g in graded_dirs):
        sys.exit("--out must differ from every --graded-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for g in graded_dirs:
        if not g.is_dir():
            sys.exit(f"graded directory not found: {g}")
        for p in sorted(g.glob("*__run1.jsonl")):
            paths.setdefault(p.name.split("__run")[0], p)
    keys = sorted(paths)
    graded_dir = graded_dirs[0]
    print(f"{len(keys)} configurations in {[str(g) for g in graded_dirs]}; count cache {len(CACHE)} pieces; "
          f"judge dirs {[str(d) for d in JUDGE_DIRS] or 'none'}")
    cfgs = {k: load_config(paths[k]) for k in keys}
    native = [k for k in keys if config_variant(k)[1] == "native"]
    nothink = [k for k in keys if config_variant(k)[1] == "nothink"]
    rules = [k for k in keys if config_variant(k)[1] == "rules"]
    o = {"_meta": {"graded_dirs": [str(g) for g in graded_dirs], "caches": caches_used, "cache_pieces": len(CACHE),
                   "judge_dirs": [str(d) for d in JUDGE_DIRS], "n_compose_items": len(COMPOSE),
                   "configs": keys, "seed": SEED, "n_boot": N_BOOT}}

    # ------------------------------------------------------------ leaderboard
    board = {}
    for k in keys:
        ag = aggregate(cfgs[k])
        tk = aggregate(cfgs[k], basis="tok", valid_key="valid_tok")
        board[k] = {"display": display(k), "variant": config_variant(k)[1], **ag,
                    "meter_tok": tk["meter"], "ci_tok": tk["ci"], "valid_tok": tk["valid"],
                    "valid_rate_tok": tk["valid_rate"], "judge": judge_rates(cfgs[k])}
    o["leaderboard"] = board
    complete = [k for k in native if board[k]["complete"]]
    partial = [k for k in native if not board[k]["complete"]]
    near = [k for k in native if board[k]["n_eval"] >= 150]
    o["config_sets"] = {"native": native, "complete160": complete, "partial": partial,
                        "n>=150": near, "nothink": nothink, "rules": rules}

    # ------------------------------------------------------------ per-dimension errors (strong)
    strong = [k for k in STRONG if k in cfgs]
    err = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    links = defaultdict(lambda: [0, 0])
    for k in strong:
        for a in cfgs[k]["records"]:
            if a["status"] != "graded":
                continue
            for d in a["bound"]:
                pp, tt = a["checks"][d]
                err[a["meter_id"]][d][0] += tt - pp
                err[a["meter_id"]][d][1] += tt
            if a["meter_id"] == "intharawichian_chan_11":
                for i, ok in enumerate(a["rhyme_links"]):
                    links[i][0] += not ok
                    links[i][1] += 1
    o["dims_strong"] = {"configs": strong,
                        "error_pct": {FORM_EN[f]: {DIM_EN[d]: 100 * v[0] / v[1] for d, v in err[f].items() if v[1]}
                                      for f in FORMS},
                        "chan_rhyme_link_fail_pct": [100 * v[0] / v[1] for _, v in sorted(links.items())]}
    pm, allsc = defaultdict(list), []
    for k in native:
        for a in cfgs[k]["records"]:
            if a["status"] == "infra_error":
                continue
            if a["status"] == "graded":
                pm[a["meter_id"]].append(a["cor"])
                allsc.append(a["cor"])
            else:
                allsc.append(0.0)
    o["pooled_by_form"] = {"configs": native, "by_form_delivered": {FORM_EN[f]: mean(pm[f]) for f in FORMS},
                           "overall_fail_scored": mean(allsc)}

    # ------------------------------------------------------------ tokenizer vs count-of-record
    cor = {k: board[k]["meter"] for k in near}
    tok = {k: board[k]["meter_tok"] for k in near}
    d = {k: tok[k] - cor[k] for k in near}
    o["tokenizer_vs_count"] = {
        "configs": near, "rows": {k: {"cor": cor[k], "tok": tok[k], "delta": d[k]} for k in near},
        "max_abs_delta": max((abs(v) for v in d.values()), default=None),
        "argmax": max(d, key=lambda k: abs(d[k])) if d else None,
        "rho": spearman(list(cor.values()), list(tok.values())), "flips": flips(near, cor, tok)}

    # ------------------------------------------------------------ weights / valid-stanza ranking
    pub = [board[k]["meter"] for k in near]
    eqw = [mean_basis(cfgs[k], "eq") for k in near]
    o["weights"] = {"configs": near, "rho_equal_weight_vs_published": spearman(pub, eqw),
                    "rho_valid_stanza_vs_published": spearman(pub, [board[k]["valid_rate"] for k in near]),
                    "equal_weight_flips": flips(near, dict(zip(near, pub)), dict(zip(near, eqw)))}

    # ------------------------------------------------------------ thinking disabled
    nt_pairs = [(config_variant(k)[0], k) for k in nothink if config_variant(k)[0] in board]
    b_nt = count_basis([cfgs[k] for _, k in nt_pairs])
    o["thinking_disabled"] = {"basis_nothink": BASIS_NAME[b_nt], "rows": {
        base: {"native": {"delivery": board[base]["delivery"], "meter": board[base]["meter"]},
               "nothink": {"delivery": board[k]["delivery"],
                           "meter": board[k]["meter"] if b_nt == "cor" else board[k]["meter_tok"],
                           "meter_count_of_record": board[k]["meter"], "meter_tokenizer": board[k]["meter_tok"],
                           "uncounted_pieces": board[k]["uncounted_pieces"]}}
        for base, k in nt_pairs}}

    # ------------------------------------------------------------ rules in prompt
    rl_pairs = [(config_variant(k)[0], k) for k in rules if config_variant(k)[0] in board]
    b_rl = count_basis([cfgs[x] for pair in rl_pairs for x in pair])
    vk = "valid" if b_rl == "cor" else "valid_tok"
    ck = "count" if b_rl == "cor" else "count_tok"
    abl = {}
    for base, k in rl_pairs:
        z, r_ = aggregate(cfgs[base], basis=b_rl, valid_key=vk), aggregate(cfgs[k], basis=b_rl, valid_key=vk)
        abl[base] = {"zero": {kk: z[kk] for kk in ("n_eval", "delivery", "meter", "ci", "valid")},
                     "rules": {kk: r_[kk] for kk in ("n_eval", "delivery", "meter", "ci", "valid")},
                     "delta": r_["meter"] - z["meter"],
                     "dims_zero": dim_stats(cfgs[base]["records"], ck),
                     "dims_rules": dim_stats(cfgs[k]["records"], ck),
                     "uncounted_pieces_rules": board[k]["uncounted_pieces"]}
    o["rules_in_prompt"] = {"basis": BASIS_NAME[b_rl], "rows": abl}

    # ------------------------------------------------------------ topic categories
    topics_per_cat = Counter(CAT_EN[c] for c, _ in TOPICS)
    cat = {"topics_per_category": {c: topics_per_cat[c] for c in CAT_ORDER}}
    for name, ks in (("strong", strong), ("complete160", complete)):
        byc, byfc = {}, {}
        for c in CAT_ORDER:
            xs = [a["cor"] if a["status"] == "graded" else 0.0 for k in ks for a in cfgs[k]["records"]
                  if a["status"] != "infra_error" and a["category"] == c]
            byc[c] = {"mean": mean(xs), "n": len(xs), "ci": bootstrap_ci(xs)}
            for f in FORMS:
                ys = [a["cor"] if a["status"] == "graded" else 0.0 for k in ks for a in cfgs[k]["records"]
                      if a["status"] != "infra_error" and a["category"] == c and a["meter_id"] == f]
                byfc[f"{FORM_EN[f]}|{c}"] = {"mean": mean(ys), "n": len(ys)}
        byf = {}
        for f in FORMS:
            zs = [a["cor"] if a["status"] == "graded" else 0.0 for k in ks for a in cfgs[k]["records"]
                  if a["status"] != "infra_error" and a["meter_id"] == f]
            byf[FORM_EN[f]] = mean(zs)
        overall = mean([a["cor"] if a["status"] == "graded" else 0.0 for k in ks
                        for a in cfgs[k]["records"] if a["status"] != "infra_error"])
        rho = {c: spearman([board[k]["meter"] for k in ks],
                           [mean_basis(cfgs[k], "cor", lambda a, c=c: a["category"] == c) for k in ks])
               for c in CAT_ORDER}
        vals = [v["mean"] for v in byc.values() if v["mean"] is not None]
        cat[name] = {"configs": ks, "overall": overall, "by_category": byc, "by_form_category": byfc,
                     "by_form": byf, "rank_rho_vs_overall": rho,
                     "spread_max_minus_min": (max(vals) - min(vals)) if vals else None}
    o["topic_categories"] = cat

    # ------------------------------------------------------------ replication
    rep_dir = next((g / "replication" for g in graded_dirs if (g / "replication").is_dir()), graded_dir / "replication")
    rep_cfgs = {}
    for key in REPLICATION:
        runs = {}
        for i in (1, 2, 3):
            p = rep_dir / f"{key}__run{i}.jsonl"
            if p.exists():
                runs[i] = load_config(p)
        if len(runs) >= 2:
            rep_cfgs[key] = runs
    b_rep = count_basis([c for runs in rep_cfgs.values() for c in runs.values()])
    rep = {}
    for key, runs in rep_cfgs.items():
        per_run = {}
        for i, c in runs.items():
            byid = {a["item_id"]: a for a in c["records"]}
            sc = [byid[x][b_rep] if byid[x]["status"] == "graded" else 0.0
                  for x in COMPOSE_IDS if x in byid and byid[x]["status"] != "infra_error"]
            ng = sum(1 for x in COMPOSE_IDS if x in byid and byid[x]["status"] == "graded")
            lo, hi = bootstrap_ci(sc)
            per_run[i] = {"n": len(sc), "delivery": ng / len(sc) if sc else None, "mean": mean(sc),
                          "ci": [lo, hi], "ci_halfwidth": (hi - lo) / 2 if lo is not None else None}
        ms = [per_run[i]["mean"] for i in sorted(per_run)]
        sd = statistics.stdev(ms) if len(ms) > 1 else None
        hw = mean([per_run[i]["ci_halfwidth"] for i in per_run])
        rep[key] = {"runs": per_run, "n_runs": len(per_run), "mean_of_runs": mean(ms), "sd_runs": sd,
                    "mean_ci_halfwidth": hw, "sd_over_ci_halfwidth": sd / hw if sd is not None and hw else None}
    full = [k for k in rep if rep[k]["n_runs"] == 3]
    ranking = {}
    if len(full) >= 2:
        vec = {i: [rep[k]["runs"][i]["mean"] for k in full] for i in (1, 2, 3)}
        ranking = {"configs": full, "rho_run1_run2": spearman(vec[1], vec[2]),
                   "rho_run1_run3": spearman(vec[1], vec[3]), "rho_run2_run3": spearman(vec[2], vec[3]),
                   "rank_by_run": {k: [sorted(full, key=lambda x: -rep[x]["runs"][i]["mean"]).index(k) + 1
                                       for i in (1, 2, 3)] for k in full}}
    o["replication"] = {"basis": BASIS_NAME[b_rep], "configs": rep, "ranking": ranking}

    # ------------------------------------------------------------ rhyme checker vs expert
    labels = json.loads(LABELS_A.read_text(encoding="utf-8"))["labels"]
    cal_items = json.loads((CAL / "calibration_items_v2.json").read_text(encoding="utf-8"))
    tp = fp = fn = tn = 0
    errs = []
    for it in (x for x in cal_items if x.get("kind") == "rhyme_pair"):
        lab = labels.get(it["id"], {}).get("valid")
        if lab is None:
            continue
        g, lab = bool(rhymes(it["word_a"], it["word_b"])), bool(lab)
        tp += g and lab
        fp += g and not lab
        fn += (not g) and lab
        tn += (not g) and (not lab)
        if g != lab:
            errs.append({"type": "FP" if g else "FN", "a": it["word_a"], "b": it["word_b"]})
    nl = tp + fp + fn + tn
    o["rhyme_checker"] = {"n_labeled": nl, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / (tp + fn) if tp + fn else None,
                          "accuracy": (tp + tn) / nl if nl else None, "errors": errs}

    # ------------------------------------------------------------ judge agreement with the expert
    judges = {"Claude Opus 4.7": CAL / "judge_cache_opus.json", "GPT-5.5": CAL / "judge_cache_gpt55.json"}
    for p in sorted(CAL.glob("judge_api_calib_*.json")):
        name = p.stem.replace("judge_api_calib_", "")
        judges[{"claude-opus-5-5": "Claude Opus 5.5"}.get(name, name)] = p
    jc = {}
    for jn, p in judges.items():
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            jc[jn] = obj.get("scores", obj) if isinstance(obj, dict) and "scores" in obj else obj
    agree = {}
    for axis, cut in (("topicality", 3), ("coherence", 2), ("coherence", 3)):
        ax = {"cut": cut, "judges": {}}
        hs = []
        for jn, cache in jc.items():
            pr = [(labels[i][axis], cache[i][axis]) for i in cache
                  if isinstance(labels.get(i, {}).get(axis), (int, float))
                  and isinstance(cache[i].get(axis), (int, float))]
            hs = [h for h, _ in pr]
            po_, kb = cohen_kappa([(int(h >= cut), int(j >= cut)) for h, j in pr])
            ax["judges"][jn] = {"n": len(pr), "binary_agree": po_, "binary_kappa": kb,
                                "weighted_kappa_quadratic": weighted_kappa(pr),
                                "alpha_ordinal": krippendorff_alpha([[h, j] for h, j in pr], "ordinal"),
                                "dist": {s: sum(1 for _, j in pr if j == s) for s in range(1, 6)}}
        ax["expert_dist"] = {s: sum(1 for h in hs if h == s) for s in range(1, 6)}
        agree[f"{axis}>={cut}"] = ax
    o["judge_agreement"] = agree

    # ------------------------------------------------------------ count-of-record checks
    ctr = {"cache_pieces": len(CACHE),
           "selfcheck_flags": sum(1 for v in CACHE.values()
                                  if v.get("selfcheck_flag") is True or v.get("flag") == "selfcheck")}
    agree_dict = [len(split_syllables(k)) == cached_count(k) for k in CACHE if cached_count(k) is not None]
    ctr["dictionary_agrees"] = {"agree": sum(agree_dict), "n": len(agree_dict),
                                "rate": sum(agree_dict) / len(agree_dict) if agree_dict else None}
    try:
        from thaiwrite.prosody.data.audit_poems import POEMS as AUDIT
        gold = [("".join(v), len(v)) for p in AUDIT for v in p.lines]
        inc = [(t, g, cached_count(t)) for t, g in gold if cached_count(t) is not None]
        ctr["hand_segmented_lines"] = {"n_lines": len(gold), "in_cache": len(inc),
                                       "exact_in_cache": sum(1 for _, g, c in inc if g == c),
                                       "dictionary_exact": sum(1 for t, g in gold if len(split_syllables(t)) == g)}
    except Exception as exc:  # noqa: BLE001
        ctr["hand_segmented_lines"] = {"error": str(exc)[:200]}
    o["count_of_record"] = ctr

    unc = defaultdict(set)
    for k in keys:
        for a in cfgs[k]["records"]:
            for p in a.get("miss", []):
                unc[p].add(k)
    for key, runs in rep_cfgs.items():
        for i, c in runs.items():
            for a in c["records"]:
                for p in a.get("miss", []):
                    unc[p].add(f"{key} run{i}")
    o["uncounted"] = {"distinct": len(unc), "by_config": {k: board[k]["uncounted_pieces"] for k in keys}}
    (out_dir / "uncounted_pieces.json").write_text(json.dumps(
        {"description": "whitespace-separated line pieces not found in any count cache; they were counted "
                        "with the dictionary tokenizer. Count them with thaiwrite/prosody/count_api.py.",
         "pieces": {p: sorted(ks) for p, ks in sorted(unc.items())}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "numbers.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=list),
                                          encoding="utf-8")
    write_md(o, out_dir)
    write_tex(o, out_dir)
    print(f"wrote tables.md, tables.tex, numbers.json, uncounted_pieces.json in {out_dir}")
    if unc:
        print(f"[note] {len(unc)} distinct line pieces are not in the count cache(s) and were counted with "
              "the dictionary tokenizer (see uncounted_pieces.json).")
    unj = {k: board[k]["judge"]["unjudged"] for k in native if board[k]["judge"]["unjudged"]}
    if unj:
        print(f"[note] delivered poems without a content judgment: {unj}")


# ============================================================================ tables.md
def write_md(o, out_dir):
    B, S = o["leaderboard"], o["config_sets"]
    meta = o["_meta"]
    L = ["# ThaiProsody: paper tables", "",
         "Generated by `reproduce/make_paper_tables.py`. Count-of-record caches: "
         + ", ".join(f"`{Path(c['path']).name}` ({c['added']} pieces)" for c in meta["caches"])
         + ". Content-judge directories: "
         + (", ".join(f"`{Path(d).name}`" for d in meta["judge_dirs"]) or "none") + ".", ""]

    def lb_rows(ks, basis_note=False):
        rows = []
        for k in sorted(ks, key=lambda k: -B[k]["meter"]):
            b, j = B[k], B[k]["judge"]
            rows.append([b["display"], b["n_eval"], b["n_model_fail"], b["n_infra"], hu(b["delivery"], 2),
                         f"{hu(b['meter'], 3)} {ci_s(b['ci'])}", f"{b['valid']} ({pct(b['valid_rate'], 1)}%)",
                         pct(j["top"]), pct(j["coh"]), j["unjudged"], hu(b["meter_tok"], 3), b["uncounted_pieces"]])
        return rows
    hdr = ["Model", "n", "non-delivery", "infra", "dlv.", "Meter [95% CI]", "Valid", "Top. %", "Coh. %",
           "unjudged", "Meter (tokenizer)", "uncounted pieces"]
    src = defaultdict(Counter)
    for k in S["native"]:
        for d_, n_ in B[k]["judge"]["sources"].items():
            src[Path(d_).name][B[k]["display"]] += n_
    judge_note = ("Content judgments by directory: " + "; ".join(
        f"`{d_}`: " + ", ".join(f"{m} ({n_})" for m, n_ in c.items()) for d_, c in src.items()) + ".") if src else ""
    L += ["## Composition leaderboard (configurations evaluated on all 160 items)", "",
          "Meter under the count-of-record; Valid = stanzas satisfying every bound constraint (non-delivery "
          "invalid); Top./Coh. = % of items judged on-topic / coherent (non-delivery counts as neither); "
          "'unjudged' = delivered poems with no judgment (Top./Coh. are left blank while any remain); "
          "'Meter (tokenizer)' = the same score with the grader's dictionary-tokenizer syllable count.", "",
          md_table(hdr, lb_rows(S["complete160"])), "", judge_note, ""]
    if S["partial"]:
        L += ["## Configurations with fewer than 160 evaluated items", "", md_table(hdr, lb_rows(S["partial"])), ""]

    D = o["dims_strong"]
    rows = []
    for f in FORMS:
        e = D["error_pct"][FORM_EN[f]]
        rows.append([FORM_EN[f]] + [hu(e[d], 1) if d in e else "--" for d in ("count", "tone", "rhyme", "weight")])
    L += ["## Per-dimension error rate, strong composers (% of rule checks failed)", "",
          "Configurations: " + ", ".join(DISPLAY.get(k, k) for k in D["configs"]) + ".", "",
          md_table(["Form", "count", "tone", "rhyme", "weight"], rows), "",
          "Chan rhyme links failed (%, per link): " + ", ".join(hu(x, 1) for x in D["chan_rhyme_link_fail_pct"]) + ".", ""]
    P = o["pooled_by_form"]
    L += ["Mean meter by form over delivered poems of all native configurations: "
          + ", ".join(f"{f} {hu(v, 3)}" for f, v in P["by_form_delivered"].items())
          + f"; overall (non-delivery = 0) {hu(P['overall_fail_scored'], 3)}.", ""]

    T = o["tokenizer_vs_count"]
    rows = [[B[k]["display"], hu(v["cor"], 3), hu(v["tok"], 3), sgn(v["delta"])]
            for k, v in sorted(T["rows"].items(), key=lambda kv: -kv[1]["cor"])]
    L += ["## Count-of-record vs dictionary tokenizer (configurations with n >= 150)", "",
          md_table(["Model", "count-of-record", "tokenizer", "delta"], rows), "",
          f"Max |delta| = {hu(T['max_abs_delta'], 3)} ({T['argmax']}); Spearman rho = {hu(T['rho'], 3)}; "
          f"order flips: {T['flips'] or 'none'}.", ""]
    W = o["weights"]
    L += [f"Ranking robustness (n >= 150, {len(W['configs'])} configurations): equal weights vs published "
          f"rho = {hu(W['rho_equal_weight_vs_published'], 3)} (flips: {W['equal_weight_flips'] or 'none'}); "
          f"valid-stanza rate vs published rho = {hu(W['rho_valid_stanza_vs_published'], 3)}.", ""]

    N = o["thinking_disabled"]
    rows = [[DISPLAY.get(b, b), hu(v["native"]["delivery"], 2), hu(v["native"]["meter"], 3),
             hu(v["nothink"]["delivery"], 2), hu(v["nothink"]["meter"], 3),
             hu(v["nothink"]["meter_count_of_record"], 3), hu(v["nothink"]["meter_tokenizer"], 3),
             v["nothink"]["uncounted_pieces"]]
            for b, v in sorted(N["rows"].items(), key=lambda kv: -kv[1]["nothink"]["meter"])]
    L += ["## Thinking disabled", "",
          f"Native meter under the count-of-record; the no-think column uses the {N['basis_nothink']} "
          "(the count-of-record is used when it covers every no-think line).", "",
          md_table(["Model", "native dlv.", "native meter", "no-think dlv.", "no-think meter",
                    "no-think (count-of-record)", "no-think (tokenizer)", "uncounted pieces"], rows), ""]

    R = o["rules_in_prompt"]
    rows = []
    for k, v in sorted(R["rows"].items(), key=lambda kv: -kv[1]["zero"]["meter"]):
        dz, dr = v["dims_zero"], v["dims_rules"]
        cells = [f"{hu(dz[d]['poem_mean'], 2)} -> {hu(dr[d]['poem_mean'], 2)}" if d in dz and d in dr else "--"
                 for d in ("count", "tone", "kru_lahu", "rhyme")]
        rows.append([DISPLAY.get(k, k), f"{hu(v['zero']['meter'], 3)} -> {hu(v['rules']['meter'], 3)} ({sgn(v['delta'])})",
                     f"{v['zero']['valid']} -> {v['rules']['valid']}"] + cells)
    L += ["## Rules in the prompt", "",
          f"Zero-shot -> rules in the system prompt, 160 items each, {R['basis']} for the count dimension "
          "(the count-of-record is used when it covers every line of both conditions). Per-dimension "
          "columns = mean per-poem pass rate over delivered poems.", "",
          md_table(["Model", "Meter", "Valid (of 160)", "count", "tone", "weight", "rhyme"], rows), ""]

    C = o["topic_categories"]
    cs, cc = C["strong"], C["complete160"]
    rows = [[c, C["topics_per_category"][c]]
            + [hu(cs["by_form_category"][f"{FORM_EN[f]}|{c}"]["mean"], 3) for f in FORMS]
            + [f"{hu(cs['by_category'][c]['mean'], 3)} {ci_s(cs['by_category'][c]['ci'])}",
               f"{hu(cc['by_category'][c]['mean'], 3)} {ci_s(cc['by_category'][c]['ci'])}",
               hu(cc["rank_rho_vs_overall"][c], 3)]
            for c in CAT_ORDER]
    rows.append(["all", sum(C["topics_per_category"].values())]
                + [hu(cs["by_form"][FORM_EN[f]], 3) for f in FORMS] + [hu(cs["overall"], 3), hu(cc["overall"], 3), ""])
    L += ["## Topic categories", "",
          f"Strong composers = {len(cs['configs'])} configurations; all = the {len(cc['configs'])} configurations "
          "of the leaderboard. Non-delivery = 0.", "",
          md_table(["Category", "Topics", "Khlong", "Klon", "Kap", "Chan", "strong", "all", "rho (rank in category vs overall, all)"], rows),
          "", f"Max - min across category means (all): {hu(cc['spread_max_minus_min'], 3)}; "
              f"(strong): {hu(cs['spread_max_minus_min'], 3)}.", ""]

    Rp = o["replication"]
    rk = Rp["ranking"].get("rank_by_run", {})
    rows = []
    for k, v in sorted(Rp["configs"].items(), key=lambda kv: -kv[1]["mean_of_runs"]):
        rows.append([DISPLAY.get(k, k)] + [hu(v["runs"][i]["mean"], 3) if i in v["runs"] else "--" for i in (1, 2, 3)]
                    + [f"{hu(v['mean_of_runs'], 3)} +- {hu(v['sd_runs'], 3)}", hu(v["mean_ci_halfwidth"], 3),
                       pct(v["sd_over_ci_halfwidth"]), "/".join(str(x) for x in rk.get(k, [])) or "--"])
    R_ = Rp["ranking"]
    L += ["## Replication (three generation passes)", "",
          f"Mean meter per run (non-delivery = 0) under the {Rp['basis']} (the count-of-record is used when it "
          "covers every line of every run).", "",
          md_table(["Model", "Run 1", "Run 2", "Run 3", "Mean +- SD", "CI half-width", "SD / half-width %", "Rank"], rows),
          "", (f"Spearman rho between runs: 1-2 {hu(R_.get('rho_run1_run2'), 2)}, 1-3 {hu(R_.get('rho_run1_run3'), 2)}, "
               f"2-3 {hu(R_.get('rho_run2_run3'), 2)}." if R_ else ""), ""]

    rv = o["rhyme_checker"]
    L += ["## Rhyme checker vs expert labels", "",
          f"{rv['n_labeled']} labeled pairs: TP {rv['TP']}, FP {rv['FP']}, FN {rv['FN']}, TN {rv['TN']}; "
          f"precision {hu(rv['precision'], 3)}, recall {hu(rv['recall'], 3)}, accuracy {hu(rv['accuracy'], 3)}. "
          "Disagreements: " + ("; ".join(f"{e['type']} {e['a']}-{e['b']}" for e in rv["errors"]) or "none") + ".", ""]

    J = o["judge_agreement"]
    rows = []
    for ax, v in J.items():
        for jn, s in v["judges"].items():
            rows.append([ax, jn, s["n"], hu(s["binary_kappa"], 2), hu(s["binary_agree"], 3),
                         hu(s["weighted_kappa_quadratic"], 2), hu(s["alpha_ordinal"], 2),
                         "/".join(str(s["dist"][x]) for x in range(1, 6))])
        rows.append([ax, "expert", "", "", "", "", "", "/".join(str(v["expert_dist"][x]) for x in range(1, 6))])
    L += ["## Content-judge agreement with the expert", "",
          md_table(["Axis (binary cut)", "Judge", "n", "kappa (binary)", "agreement", "kappa_w (1-5)",
                    "alpha ordinal (1-5)", "scores 1/2/3/4/5"], rows), ""]

    K = o["count_of_record"]
    h = K.get("hand_segmented_lines", {})
    L += ["## Count-of-record checks", "",
          f"- Cache: {K['cache_pieces']} pieces; self-check flags {K['selfcheck_flags']}.",
          f"- Dictionary tokenizer agrees with the count-of-record on {K['dictionary_agrees']['agree']}/"
          f"{K['dictionary_agrees']['n']} = {pct(K['dictionary_agrees']['rate'], 1)}% of cached pieces.",
          f"- Hand-segmented audit lines: {h.get('n_lines')}; in cache {h.get('in_cache')}, exact "
          f"{h.get('exact_in_cache')}; dictionary tokenizer exact on {h.get('dictionary_exact')}.",
          f"- Line pieces not in any cache (counted with the tokenizer): {o['uncounted']['distinct']} distinct.", ""]
    (out_dir / "tables.md").write_text("\n".join(L), encoding="utf-8")


# ============================================================================ tables.tex
def write_tex(o, out_dir):
    B, S = o["leaderboard"], o["config_sets"]
    T = ["% Generated by reproduce/make_paper_tables.py. Requires booktabs. Values rounded half-up.", ""]

    def row(k):
        b, j = B[k], B[k]["judge"]
        return (f"{tex_name(k):<28}& {b['n_eval']} & {hu(b['delivery'], 2)} & ${hu(b['meter'], 3)}$ "
                f"\\scriptsize{ci_s(b['ci'])} & {pct(b['valid_rate'])} & {pct(j['top'])} & {pct(j['coh'])} \\\\")
    head = [r"\toprule", r"Model & $n$ & dlv. & Meter \scriptsize[95\% CI] & Valid & Top. & Coh. \\", r"\midrule"]
    T += [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{6pt}",
          r"\begin{tabular}{@{}lrclccc@{}}", *head,
          *(row(k) for k in sorted(S["complete160"], key=lambda k: -B[k]["meter"])),
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Composition leaderboard: configurations evaluated on all 160 items, sorted by meter score.",
          r"\emph{Meter} = mean over the 160 items (non-delivery${}={}0$) under the count-of-record, with",
          r"item-bootstrap 95\% CIs; dlv.\ = delivery rate; \emph{Valid} = \% of stanzas that satisfy every",
          r"constraint of their form (non-delivery counting as invalid). \emph{Top.}\ and \emph{Coh.}\ = \% of poems",
          r"judged on-topic and coherent, a non-delivery counting as neither. \textsc{th}~=~Thai-trained.}",
          r"\label{tab:leaderboard}", r"\end{table*}", ""]
    part = sorted((k for k in S["partial"] if B[k]["n_eval"] >= TEX_MIN_N), key=lambda k: -B[k]["meter"])
    if part:
        T += [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
              r"\begin{tabular}{@{}lrclccc@{}}", *head, *(row(k) for k in part), r"\bottomrule", r"\end{tabular}",
              r"\caption{Configurations with fewer than 160 evaluated items. Columns as in Table~\ref{tab:leaderboard}.}",
              r"\label{tab:partial}", r"\end{table}", ""]
    D = o["dims_strong"]["error_pct"]
    T += [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lcccc@{}}",
          r"\toprule", r"Form & count & tone & rhyme & weight \\", r"\midrule"]
    for f in FORMS:
        e = D[FORM_EN[f]]
        T.append(f"\\emph{{{FORM_EN[f]}}} & " + " & ".join(hu(e[d], 0) if d in e else "---"
                                                        for d in ("count", "tone", "rhyme", "weight")) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Per-dimension \emph{error rate} (\% of rule checks failed) for strong composers",
          r"(GPT-5.5$\times$3, Gemini, Opus, Gemma, GPT-5.4-mini).}", r"\label{tab:dims}", r"\end{table}", ""]
    Tk = o["tokenizer_vs_count"]
    T += [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lccc@{}}",
          r"\toprule", r"Model & Count-of-record & Tokenizer & $\Delta$ \\", r"\midrule",
          *(f"{tex_name(k):<28}& ${hu(v['cor'], 3)}$ & ${hu(v['tok'], 3)}$ & ${sgn(v['delta'])}$ \\\\"
            for k, v in sorted(Tk["rows"].items(), key=lambda kv: -kv[1]["cor"])),
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Meter score with the syllable count from the count-of-record versus the dictionary",
          r"tokenizer ($\Delta$ = tokenizer $-$ count-of-record), configurations with $n \ge 150$. "
          f"Max $|\\Delta| = {hu(Tk['max_abs_delta'], 3)}$; Spearman $\\rho = {hu(Tk['rho'], 3)}$ between the two rankings.}}",
          r"\label{tab:tokenizer}", r"\end{table}", ""]
    N = o["thinking_disabled"]
    T += [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lcccc@{}}",
          r"\toprule", r" & \multicolumn{2}{c}{native} & \multicolumn{2}{c}{no-think} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Model & dlv. & meter & dlv. & meter \\", r"\midrule"]
    for b, v in sorted(N["rows"].items(), key=lambda kv: -kv[1]["nothink"]["meter"]):
        T.append(f"{tex_name(b):<20}& {hu(v['native']['delivery'], 2)} & ${hu(v['native']['meter'], 3)}$ & "
                 f"{hu(v['nothink']['delivery'], 2)} & ${hu(v['nothink']['meter'], 3)}$ \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Thinking-disabled ablation for the open reasoning models on the 160 composition items",
          r"(non-delivery${}={}0$). \emph{native}: count-of-record; \emph{no-think}: "
          + N["basis_nothink"].replace("dictionary-tokenizer count", "dictionary-tokenizer syllable count") + ".}",
          r"\label{tab:ablation}", r"\end{table}", ""]
    R = o["rules_in_prompt"]
    T += [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lcccccc@{}}",
          r"\toprule", r"Model & Meter & Valid & count & tone & weight & rhyme \\", r"\midrule"]

    def arr(a, b, nd=2):
        f = dot3 if nd == 3 else (lambda x: hu(x, nd).replace("0.", ".", 1))
        return f"${f(a)}\\to{f(b)}$"
    for k, v in sorted(R["rows"].items(), key=lambda kv: -kv[1]["zero"]["meter"]):
        dz, dr = v["dims_zero"], v["dims_rules"]
        cells = [arr(dz[d]["poem_mean"], dr[d]["poem_mean"]) if d in dz and d in dr else "---"
                 for d in ("count", "tone", "kru_lahu", "rhyme")]
        T.append(f"{tex_name(k):<27} & {arr(v['zero']['meter'], v['rules']['meter'], 3)} & "
                 f"${v['zero']['valid']}\\to{v['rules']['valid']}$ & " + " & ".join(cells) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Rules-in-prompt ablation: zero-shot $\to$ the same prompt with the form's full specification",
          r"in the system prompt, 160 items each. \emph{Meter} with the " + R["basis"] + r" for the count dimension;",
          r"\emph{Valid} = stanzas satisfying every constraint (of 160); per-dimension columns = mean per-poem",
          r"pass rate over delivered poems (tone: \emph{Khlong} only; weight: \emph{Chan} only).}",
          r"\label{tab:rules}", r"\end{table*}", ""]
    C = o["topic_categories"]
    cs, cc = C["strong"], C["complete160"]
    T += [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lrccccccc@{}}",
          r"\toprule", r" & & \multicolumn{5}{c}{strong composers} & all \\", r"\cmidrule(lr){3-7}\cmidrule(lr){8-8}",
          r"Category & Topics & \emph{Khlong} & \emph{Klon} & \emph{Kap} & \emph{Chan} & all & all \\", r"\midrule"]
    for c in CAT_ORDER:
        cells = [dot3(cs["by_form_category"][f"{FORM_EN[f]}|{c}"]["mean"]) for f in FORMS]
        T.append(f"{c} & {C['topics_per_category'][c]} & " + " & ".join(cells)
                 + f" & {dot3(cs['by_category'][c]['mean'])} & {dot3(cc['by_category'][c]['mean'])} \\\\")
    T += [r"\midrule", f"all & {sum(C['topics_per_category'].values())} & "
          + " & ".join(dot3(cs["by_form"][FORM_EN[f]]) for f in FORMS) + f" & {dot3(cs['overall'])} & {dot3(cc['overall'])} \\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Mean meter score by topic category (each topic is prompted in all four forms).",
          f"Strong composers = {len(cs['configs'])} configurations; \\emph{{all}} = the {len(cc['configs'])} configurations of",
          r"Table~\ref{tab:leaderboard}. Non-delivery${}={}0$.}", r"\label{tab:categories}", r"\end{table}", ""]
    Rp = o["replication"]
    rk = Rp["ranking"].get("rank_by_run", {})
    T += [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lccccc@{}}",
          r"\toprule", r"Model & Run 1 & Run 2 & Run 3 & Mean $\pm$ SD & Rank \\", r"\midrule"]
    for k, v in sorted(Rp["configs"].items(), key=lambda kv: -kv[1]["mean_of_runs"]):
        cells = [f"{v['runs'][i]['mean']:.3f}" if i in v["runs"] else "--" for i in (1, 2, 3)]
        T.append(f"{tex_name(k):<24} & " + " & ".join(cells)
                 + f" & ${v['mean_of_runs']:.3f} \\pm {v['sd_runs']:.3f}$ & {'/'.join(str(x) for x in rk.get(k, [])) or '--'} \\\\")
    T += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Run-to-run replication: three generation passes over the 160 composition items with identical",
          r"decoding settings. Mean meter score per run (non-delivery${}={}0$) with the " + Rp["basis"]
          + r". Rank = position among the replicated configurations in runs 1/2/3.}",
          r"\label{tab:replication}", r"\end{table}", ""]
    J = o["judge_agreement"]
    names = list(next(iter(J.values()))["judges"]) if J else []
    T += [r"\begin{table}[h]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{@{}ll" + "c" * len(names) + r"@{}}", r"\toprule",
          r"Axis & Binary cut & " + " & ".join(names) + r" \\", r"\midrule"]
    lab = {"topicality>=3": ("Topicality", r"on-topic ($\ge 3$)"), "coherence>=2": ("Coherence", r"not word salad ($\ge 2$)"),
           "coherence>=3": ("Coherence", r"acceptable ($\ge 3$)")}
    for ax, v in J.items():
        a_, c_ = lab.get(ax, (ax, ""))
        T.append(f"{a_} & {c_} & " + " & ".join(hu(v["judges"][n]["binary_kappa"], 2) for n in names) + r" \\")
    n = next(iter(next(iter(J.values()))["judges"].values()))["n"] if J and names else 0
    T += [r"\bottomrule", r"\end{tabular}",
          f"\\caption{{Content-judge agreement with the expert (Cohen's $\\kappa$, {n} poems).}}",
          r"\label{tab:judge}", r"\end{table}", ""]
    (out_dir / "tables.tex").write_text("\n".join(T), encoding="utf-8")


if __name__ == "__main__":
    main()
