"""Bundle calibration items into self-contained per-expert HTML files.

Each expert receives ONE file (annotate_<id>.html) with the items and their id
baked in: no separate JSON, no file picker, no install. They open it, label,
click Export, and send back a single labels_<id>.json. Annotation instructions
are in docs/annotation-guidelines.md and the procedure in docs/annotation-protocol.md.

Run after the items file exists (build_calibration_v2.py produces it):
  python bundle_for_experts.py calibration_items_v2.json --experts expertA expertB
Outputs to ./dist/annotate_<id>.html (one per expert).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "annotate.html"

# Must match the placeholder lines in annotate.html exactly.
ITEMS_PLACEHOLDER = "const BAKED_ITEMS = null; /* BAKED_ITEMS_PLACEHOLDER */"
NAME_PLACEHOLDER = "const BAKED_NAME  = null; /* BAKED_NAME_PLACEHOLDER */"


def bundle(items_path: Path, experts: list[str], out_dir: Path) -> list[Path]:
    items = json.loads(items_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        sys.exit(f"items file must be a non-empty JSON array: {items_path}")

    tpl = TEMPLATE.read_text(encoding="utf-8")
    if ITEMS_PLACEHOLDER not in tpl or NAME_PLACEHOLDER not in tpl:
        sys.exit("annotate.html is missing the bake placeholders — did the template change?")

    # ensure </script> can't be broken out of by content
    items_js = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for ex in experts:
        html = tpl.replace(ITEMS_PLACEHOLDER, f"const BAKED_ITEMS = {items_js};")
        html = html.replace(NAME_PLACEHOLDER, f"const BAKED_NAME  = {json.dumps(ex)};")
        path = out_dir / f"annotate_{ex}.html"
        path.write_text(html, encoding="utf-8")
        written.append(path)
        print(f"  wrote {path.relative_to(HERE)}  ({len(items)} items · id={ex})")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("items", type=Path, help="path to calibration_items.json")
    ap.add_argument("--experts", nargs="+", default=["expertA", "expertB"])
    ap.add_argument("--out", type=Path, default=HERE / "dist")
    args = ap.parse_args()
    bundle(args.items, args.experts, args.out)
    print(f"\nSend each expert their one annotate_<id>.html; collect one labels_<id>.json back.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
