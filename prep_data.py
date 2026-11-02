"""Build the imbalanced "fraud" versions of GLBench Reddit / Instagram, one per data seed.

Popular (Reddit) / commercial (Instagram) users are the minority class, subsampled. The paper says "about 1:10",
but its all-normal baseline rows (macro-F1 = q/(q+1)) imply 16.65% / 10.32% minority, so those are the default.
--ratio 1:10 gives the stated ratio instead. Split: 40% train, rest 1:2 val:test (BWGNN setting), stratified.

    python prep_data.py --seeds 0 1 2 3 4
"""
import argparse
import hashlib
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from flag.paths import PROC as OUT
from flag.paths import RAW

IMPLIED_MINORITY_SHARE = {"reddit": 1 - 0.4546 / 0.5454, "instagram": 1 - 0.4729 / 0.5271}  # from Table 4 rows


def load_glbench(name):
    # GLBench files are pickled PyG Data objects; only allow the classes they actually contain
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    return torch.load(RAW / f"{name}.pt", weights_only=True)


def edge_homophily_by_class(edge_index, y):
    """Mean fraction of same-class neighbours, per class (cf. paper Fig. 5)."""
    src, dst = edge_index
    same = (y[src] == y[dst]).float()
    deg = torch.zeros(len(y)).index_add_(0, src, torch.ones(len(src)))
    hom = torch.zeros(len(y)).index_add_(0, src, same) / deg.clamp(min=1)
    has = deg > 0
    return {int(c): float(hom[(y == c) & has].mean()) for c in (0, 1)}


def build(name, seed, minority_label, minority_share, train_ratio):
    d = load_glbench(name)
    y_all = d.y.view(-1).long()
    n_all = len(y_all)
    rng = np.random.default_rng([seed, 17])
    maj = np.where(y_all.numpy() != minority_label)[0]
    mino = np.where(y_all.numpy() == minority_label)[0]
    n_min = int(round(len(maj) * minority_share / (1 - minority_share)))
    assert n_min <= len(mino), (name, n_min, len(mino))
    keep = np.sort(np.concatenate([maj, rng.choice(mino, n_min, replace=False)]))

    remap = -np.ones(n_all, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    ei = d.edge_index.numpy()
    m = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    edge_index = torch.from_numpy(remap[ei[:, m]])
    y = (y_all[keep] == minority_label).long()  # 1 = minority ("fraud")
    x = d.x[keep].float()
    texts = [d.raw_texts[i] for i in keep]

    idx = np.arange(len(keep))
    tr, rest = train_test_split(idx, train_size=train_ratio, stratify=y.numpy(), random_state=seed)
    va, te = train_test_split(rest, test_size=2 / 3, stratify=y.numpy()[rest], random_state=seed)
    masks = {k: torch.zeros(len(keep), dtype=torch.bool) for k in ("train_mask", "val_mask", "test_mask")}
    masks["train_mask"][tr], masks["val_mask"][va], masks["test_mask"][te] = True, True, True

    stats = {
        "dataset": name, "seed": seed, "minority_label_in_glbench": minority_label,
        "n_glbench": n_all, "glbench_class_counts": np.bincount(y_all.numpy()).tolist(),
        "n_nodes": len(keep), "n_edges": int(edge_index.shape[1]), "n_minority": int(y.sum()),
        "minority_share": float(y.float().mean()), "train_ratio": train_ratio,
        "split_sizes": [len(tr), len(va), len(te)],
        "test_minority_share": float(y[masks["test_mask"]].float().mean()),
        "all_normal_test_macro_f1": None, "x_dim": int(x.shape[1]),
        "edge_homophily_by_class": edge_homophily_by_class(edge_index, y),
        "kept_ids_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
    }
    q = 1 - stats["test_minority_share"]
    stats["all_normal_test_macro_f1"] = q / (q + 1)  # should match the paper's collapsed baseline rows
    return {"x": x, "y": y, "edge_index": edge_index, "texts": texts, "glbench_ids": torch.from_numpy(keep),
            **masks}, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["reddit", "instagram"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ratio", default="implied", choices=["implied", "1:10"])
    ap.add_argument("--train_ratio", type=float, default=0.4)
    ap.add_argument("--minority_label", type=int, default=1,
                    help="GLBench label of popular (Reddit) / commercial (Instagram)")
    a = ap.parse_args()
    tag = f"{a.ratio.replace(':', 'to')}_tr{int(a.train_ratio * 100)}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for name in a.datasets:
        share = IMPLIED_MINORITY_SHARE[name] if a.ratio == "implied" else 1 / 11
        for s in a.seeds:
            data, st = build(name, s, a.minority_label, share, a.train_ratio)
            torch.save(data, out / f"{name}_s{s}.pt")
            all_stats.append(st)
            print(json.dumps(st), flush=True)
    (out / "stats.json").write_text(json.dumps(all_stats, indent=1))


if __name__ == "__main__":
    main()
