"""Train/eval one cell of Table 4/5: (dataset, data seed, init seed, backbone, variant) -> metrics.json.

Variants:
  baseline    word2vec features, random-10 subgraphs, plain backbone
  text        SBERT of the raw text, random-10, plain backbone
  flag        raw + LLM "discriminative" text, semantic top-10 (SS), skip-GNN, attention fusion
  flag_star   same as flag but with the fine-tuned LLM's texts (--disc_tag)
  abl_sg      raw text, random-10, skip-GNN          (Table 5)
  abl_ss_sg   raw text, SS, skip-GNN                 (Table 5)
  abl_ss_llm  raw + disc, SS, plain backbone, fusion (Table 5)

2 layers, hidden 64, Adam lr 0.01, 10 subgraphs per step, early stopping on val macro-F1.
F1 is reported at threshold 0.5 (that's what matches the paper's baseline rows), plus a val-tuned threshold.
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from flag.models import FLAGFusion, SingleView, batch_graphs
from flag.models_extra import dt_bin_encode
from flag.paths import CACHE, PROC, TXT
from flag.sampling import adjacency_lists, edge_homophily, induced_edges, sample_subgraphs

VARIANTS = {  # views, sampling, skip, fusion
    "baseline": (["w2v"], "rs", False, False),
    "text": (["raw"], "rs", False, False),
    "flag": (["raw", "disc"], "ss", True, True),
    "flag_star": (["raw", "disc"], "ss", True, True),
    "abl_sg": (["raw"], "rs", True, False),
    "abl_ss_sg": (["raw"], "ss", True, False),
    "abl_ss_llm": (["raw", "disc"], "ss", False, True),
}


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["reddit", "instagram"])
    ap.add_argument("--data_tag", default="implied_tr40")
    ap.add_argument("--seed", type=int, default=0, help="data seed (subsample + split)")
    ap.add_argument("--init", type=int, default=0, help="initialisation seed")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--disc_tag", default="", help="suffix of the disc embedding file, e.g. _ft1 for FLAG*")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--accum", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def best_threshold_f1(p, y):
    best = (-1.0, 0.5)
    for t in np.unique(np.round(p, 4)):
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > best[0]:
            best = (f, float(t))
    return best


def get_subgraphs(a, d, emb_raw, strategy):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{a.data_tag}_{a.dataset}_s{a.seed}_{strategy}_n{a.top_n}_d{a.delta}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    n = len(d["y"])
    sub = sample_subgraphs(d["edge_index"], n, strategy, emb=emb_raw, top_n=a.top_n, delta=a.delta, seed=a.seed)
    f.write_bytes(pickle.dumps(sub))
    return sub


def main():
    a = args()
    out = Path(a.out)
    if (out / "metrics.json").exists():
        print("[skip]", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    import dgl
    views, strategy, skip, fusion = VARIANTS[a.variant]

    d = torch.load(PROC / a.data_tag / f"{a.dataset}_s{a.seed}.pt", weights_only=False)  # written by prep_data.py
    gid = d["glbench_ids"].numpy()
    y = d["y"].numpy()
    files = {"w2v": f"{a.dataset}_w2v.npy", "raw": f"{a.dataset}_raw_mpnet.npy",
             "disc": f"{a.dataset}_disc{a.disc_tag.format(seed=a.seed)}_mpnet.npy"}  # e.g. _ft_s{seed} for FLAG*
    feats = [torch.from_numpy(np.load(TXT / files[v])[gid]).float() for v in views]
    tr, va, te = (np.where(d[k].numpy())[0] for k in ("train_mask", "val_mask", "test_mask"))
    if a.backbone == "dgagnn":  # decision-tree binning, fit on train nodes
        feats = [dt_bin_encode(f, y, tr) for f in feats]
    emb_raw = np.load(TXT / files["raw"])[gid] if strategy == "ss" else None

    sub = get_subgraphs(a, d, emb_raw, strategy)
    adj = adjacency_lists(d["edge_index"], len(y))
    adj_set = [set(x.tolist()) for x in adj]
    graphs = []
    for nodes in sub:
        s, t = induced_edges(nodes, adj_set)
        g = dgl.graph((torch.from_numpy(s), torch.from_numpy(t)), num_nodes=len(nodes))
        graphs.append(dgl.add_self_loop(g))

    torch.manual_seed(a.init)
    np.random.seed(a.init)
    Model = FLAGFusion if fusion else SingleView
    model = Model(a.backbone, feats[0].shape[1], a.hid, skip)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng([a.seed, a.init])
    gnn = model.enc.gnn
    is_train = torch.from_numpy(d["train_mask"].numpy())
    y_t = torch.from_numpy(y).long()

    def decorate(g, nodes, tg):
        # extra fields for PMP / CARE-GNN / DGA-GNN; only train labels are visible, targets are masked
        lu = torch.where(is_train[nodes], y_t[nodes], torch.full_like(nodes, 2))
        lu[tg] = 2
        g.ndata["label_unk"] = lu
        g.ndata["label_trn"] = torch.where(is_train[nodes], y_t[nodes], torch.full_like(nodes, -1))
        g.ndata["gid"] = nodes
        return g

    def predict(idx):
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                g, nodes, tg = batch_graphs(sub, graphs, idx[i:i + 512])
                decorate(g, nodes, tg)
                ps.append(torch.sigmoid(model(g, [f[nodes] for f in feats], tg)))
        return torch.cat(ps).numpy()

    best, bad, hist, t0 = ((-1.0, -1.0), None, 0.5, -1), 0, [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), a.accum):
            idx = order[i:i + a.accum]
            g, nodes, tg = batch_graphs(sub, graphs, idx)
            decorate(g, nodes, tg)
            logit = model(g, [f[nodes] for f in feats], tg)
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[idx]).float())
            if hasattr(gnn, "aux_loss"):  # CARE-GNN similarity loss
                loss = loss + gnn.aux_loss
                gnn.aux_loss = torch.tensor(0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if hasattr(gnn, "epoch_end"):  # CARE-GNN threshold update
            gnn.epoch_end()
        if hasattr(gnn, "set_feedback"):  # DGA-GNN needs predictions for every node
            gnn.set_feedback(predict(np.arange(len(y))))
        pv = predict(va)
        f1v = f1_score(y[va], (pv >= 0.5).astype(int), average="macro", zero_division=0)
        aucv = roc_auc_score(y[va], pv)
        f1t, thr = best_threshold_f1(pv, y[va])
        hist.append({"epoch": ep, "loss": tot / len(tr), "val_f1": f1v, "val_auc": aucv, "val_f1_tuned": f1t})
        print(json.dumps(hist[-1]), flush=True)
        if (f1v, aucv) > best[0]:
            best, bad = ((f1v, aucv), {k: v.clone() for k, v in model.state_dict().items()}, thr, ep), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best[1])
    pt = predict(te)
    res = {"test_f1_macro": f1_score(y[te], (pt >= 0.5).astype(int), average="macro", zero_division=0),
           "test_auc": roc_auc_score(y[te], pt), "val_f1_macro": best[0][0], "val_auc": best[0][1],
           "test_f1_macro_tuned": f1_score(y[te], (pt >= best[2]).astype(int), average="macro", zero_division=0),
           "threshold_tuned": best[2],
           "best_epoch": best[3], "epochs_run": len(hist), "train_seconds": time.time() - t0,
           "subgraph_homophily": edge_homophily(sub, adj_set, y), "n_test": int(len(te))}
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
    res["fp_model"] = h.hexdigest()
    res["fp_preds"] = hashlib.sha256(pt.astype(np.float32).tobytes()).hexdigest()
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    (out / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in hist))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print("[result]", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
