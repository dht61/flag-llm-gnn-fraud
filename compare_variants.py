"""Paired differences between variants (same data seed + init), averaged per cell and over cells.

    python compare_variants.py flag text          # main claim: LLM text vs raw text
    python compare_variants.py flag_star flag     # does fine-tuning help?
    python compare_variants.py --table5           # component gains on the ablation backbones
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "main"


def load():
    runs = defaultdict(dict)
    for f in ROOT.glob("*/*/*/s*_i*/metrics.json"):
        ds, bb, var, run = f.parts[-5], f.parts[-4], f.parts[-3], f.parts[-2]
        runs[(ds, bb, var)][run] = json.loads(f.read_text())
    return runs


def diff(runs, a, b, backbones=None):
    cells = {}
    for (ds, bb, var), ra in runs.items():
        if var != a or (backbones and bb not in backbones):
            continue
        rb = runs.get((ds, bb, b), {})
        common = sorted(set(ra) & set(rb))
        if not common:
            continue
        cells[(ds, bb)] = {k: st.mean(100 * (ra[r][k] - rb[r][k]) for r in common)
                           for k in ("test_f1_macro", "test_auc")} | {"n": len(common)}
    return cells


def report(name, cells, verbose=True):
    if not cells:
        print(name, "no paired runs")
        return
    f1 = [c["test_f1_macro"] for c in cells.values()]
    auc = [c["test_auc"] for c in cells.values()]
    print(f"{name}: {len(cells)} cells, F1 {st.mean(f1):+.2f} ({sum(v > 0 for v in f1)}/{len(f1)} > 0), "
          f"AUC {st.mean(auc):+.2f} ({sum(v > 0 for v in auc)}/{len(auc)} > 0)")
    if verbose:
        for (ds, bb), c in sorted(cells.items()):
            print(f"  {ds:9s} {bb:9s} F1 {c['test_f1_macro']:+6.2f}  AUC {c['test_auc']:+6.2f}  (n={c['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--table5", action="store_true")
    args = ap.parse_args()
    runs = load()
    if args.table5:
        bbs = {bb for (_, bb, var) in runs if var == "abl_sg"}
        print("ablation backbones:", sorted(bbs))
        report("LLM  (flag - abl_ss_sg)", diff(runs, "flag", "abl_ss_sg", bbs), False)
        report("skip (flag - abl_ss_llm)", diff(runs, "flag", "abl_ss_llm", bbs), False)
        report("SS   (abl_ss_sg - abl_sg)", diff(runs, "abl_ss_sg", "abl_sg", bbs), False)
    else:
        report(f"{args.a} - {args.b}", diff(runs, args.a, args.b))


if __name__ == "__main__":
    main()
