"""Per-node 2-hop subgraphs (Sec. 3.1).

"ss" keeps the top-N 2-hop neighbours by SBERT cosine similarity (sim >= delta), "rs" keeps N at random.
"""
import numpy as np
import torch


def adjacency_lists(edge_index, n):
    src, dst = edge_index
    both = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1)
    both = torch.unique(both, dim=1)
    order = torch.argsort(both[0], stable=True)
    s, d = both[0][order].numpy(), both[1][order].numpy()
    ptr = np.searchsorted(s, np.arange(n + 1))
    return [d[ptr[i]:ptr[i + 1]] for i in range(n)]


def two_hop(adj, v):
    h1 = adj[v]
    h2 = np.concatenate([adj[u] for u in h1]) if len(h1) else np.empty(0, dtype=np.int64)
    c = np.unique(np.concatenate([h1, h2]))
    return c[c != v]


def sample_subgraphs(edge_index, n, strategy, emb=None, top_n=10, delta=0.0, seed=0):
    """One array [v, kept...] per node, target first."""
    adj = adjacency_lists(edge_index, n)
    rng = np.random.default_rng([seed, 99])
    if emb is not None:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    out = []
    for v in range(n):
        c = two_hop(adj, v)
        if strategy == "ss":
            sim = emb[c] @ emb[v]
            ok = sim >= delta
            c, sim = c[ok], sim[ok]
            keep = c[np.argsort(-sim, kind="stable")[:top_n]]
        elif strategy == "rs":
            keep = rng.choice(c, min(top_n, len(c)), replace=False) if len(c) else c
        else:
            raise ValueError(strategy)
        out.append(np.concatenate([[v], np.sort(keep)]).astype(np.int64))
    return out


def induced_edges(nodes, adj_set):
    pos = {int(u): i for i, u in enumerate(nodes)}
    src, dst = [], []
    for i, u in enumerate(nodes):
        for w in adj_set[int(u)]:
            j = pos.get(int(w))
            if j is not None:
                src.append(i)
                dst.append(j)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


def edge_homophily(subgraphs, adj_set, y):
    # Eq. 5, averaged over subgraphs
    vals = []
    for nodes in subgraphs:
        s, d = induced_edges(nodes, adj_set)
        if len(s):
            vals.append(float((y[nodes[s]] == y[nodes[d]]).mean()))
    return float(np.mean(vals)) if vals else float("nan")
