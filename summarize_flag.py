"""Table 4 / 5 summary (mean ± std over 5 data seeds x 5 inits), paper numbers in brackets.

    python summarize_flag.py --md results/main/summary.md
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "main"
BACKBONES = ["gcn", "gat", "geniepath", "caregnn", "bwgnn", "dgagnn", "pmp"]
VARIANTS = ["baseline", "text", "flag", "flag_star"]
ABLATIONS = ["abl_sg", "abl_ss_sg", "abl_ss_llm"]
# paper Table 4: (Reddit F1, Reddit AUC, Instagram F1, Instagram AUC)
PAPER = {
    ("gcn", "baseline"): (45.46, 50.32, 47.88, 52.61), ("gcn", "text"): (45.84, 57.82, 47.29, 55.74),
    ("gcn", "flag"): (48.19, 60.18, 48.05, 56.31), ("gcn", "flag_star"): (48.72, 60.88, 49.79, 55.45),
    ("gat", "baseline"): (46.66, 52.66, 49.21, 51.53), ("gat", "text"): (48.26, 59.32, 48.31, 54.69),
    ("gat", "flag"): (49.70, 60.61, 49.13, 54.97), ("gat", "flag_star"): (50.20, 60.57, 50.65, 55.98),
    ("geniepath", "baseline"): (45.46, 52.18, 47.31, 51.22), ("geniepath", "text"): (46.84, 56.91, 47.29, 52.45),
    ("geniepath", "flag"): (48.30, 59.43, 48.41, 55.59), ("geniepath", "flag_star"): (48.69, 59.74, 48.28, 56.24),
    ("caregnn", "baseline"): (45.46, 51.35, 47.29, 52.07), ("caregnn", "text"): (47.66, 56.72, 48.05, 54.92),
    ("caregnn", "flag"): (50.78, 58.43, 49.64, 55.79), ("caregnn", "flag_star"): (51.95, 58.74, 50.24, 56.40),
    ("bwgnn", "baseline"): (45.47, 53.82, 47.28, 51.52), ("bwgnn", "text"): (48.76, 57.56, 47.61, 54.10),
    ("bwgnn", "flag"): (50.93, 58.89, 48.55, 56.33), ("bwgnn", "flag_star"): (51.91, 59.20, 49.35, 57.19),
    ("dgagnn", "baseline"): (45.46, 50.10, 47.29, 50.86), ("dgagnn", "text"): (45.49, 59.59, 47.29, 56.28),
    ("dgagnn", "flag"): (48.77, 61.05, 48.53, 56.73), ("dgagnn", "flag_star"): (49.50, 61.61, 48.95, 57.20),
    ("pmp", "baseline"): (46.31, 50.16, 47.50, 50.63), ("pmp", "text"): (47.21, 59.79, 48.29, 56.05),
    ("pmp", "flag"): (48.91, 61.32, 48.48, 57.10), ("pmp", "flag_star"): (50.14, 61.80, 49.76, 57.67),
}


def load():
    runs = defaultdict(list)
    for f in ROOT.glob("*/*/*/s*_i*/metrics.json"):
        ds, bb, var = f.parts[-5], f.parts[-4], f.parts[-3]
        runs[(ds, bb, var)].append(json.loads(f.read_text()))
    return runs


def fmt(rows, key):
    if not rows:
        return "–"
    v = [100 * r[key] for r in rows]
    return f"{st.mean(v):.2f} ± {st.stdev(v) if len(v) > 1 else 0:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    a = ap.parse_args()
    runs = load()
    lines = ["**FLAG Table 4** (ours: F1@0.5 / AUC, mean ± std over n runs; paper in brackets)", "",
             "| Backbone | Variant | n | Reddit F1 | Reddit AUC | Instagram F1 | Instagram AUC | Reddit F1 (tuned thr) | Instagram F1 (tuned thr) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for bb in BACKBONES:
        for var in VARIANTS + ABLATIONS:
            r, i = runs.get(("reddit", bb, var), []), runs.get(("instagram", bb, var), [])
            if not r and not i:
                continue
            pp = PAPER.get((bb, var))
            ref = (lambda k: f" [{pp[k]}]") if pp else (lambda k: "")
            lines.append(f"| {bb} | {var} | {len(r)}/{len(i)} | {fmt(r, 'test_f1_macro')}{ref(0)} | {fmt(r, 'test_auc')}{ref(1)} | "
                         f"{fmt(i, 'test_f1_macro')}{ref(2)} | {fmt(i, 'test_auc')}{ref(3)} | "
                         f"{fmt(r, 'test_f1_macro_tuned')} | {fmt(i, 'test_f1_macro_tuned')} |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
