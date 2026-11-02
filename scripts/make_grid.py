"""Write grids/<name>.txt, one "dataset backbone variant seed" per line."""
import itertools
import sys

name, datasets, backbones, variants = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4].split(",")
lines = [f"{d} {b} {v} {s}" for d, b, v, s in itertools.product(datasets, backbones, variants, range(5))]
open(f"grids/{name}.txt", "w").write("\n".join(lines) + "\n")
print(name, len(lines), "tasks")
