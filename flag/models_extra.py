"""Fraud-specific backbones, re-implemented for batches of small subgraphs (single relation).

  PMP       Zhuo et al. 2024, github.com/Xtra-Computing/PMP
  CARE-GNN  Dou et al. 2020, github.com/YingtongDou/CARE-GNN
  DGA-GNN   Duan et al. 2024, github.com/AtwoodDuan/DGA-GNN

They need label info on the graph: g.ndata["label_unk"] (0/1 train label, 2 = unknown, incl. the batch targets so
labels can't leak back to them) and g.ndata["label_trn"] (-1 for non-train nodes). Self-loops carry no label.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _edges_no_self(g):
    src, dst = g.edges()
    keep = src != dst
    return src[keep], dst[keep]


class _LILinear(nn.Module):
    # label-informed linear: W (m * T(h_self)) + b

    def __init__(self, din, dout, dself):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.trans = nn.Linear(dself, din)

    def forward(self, m, h_self):
        return self.lin(m * self.trans(h_self))


class PMPConv(nn.Module):
    def __init__(self, din, dout, hid):
        super().__init__()
        self.fc_self = nn.Linear(din, dout)
        self.fr, self.be = _LILinear(din, dout, din), _LILinear(din, dout, din)
        self.balance = nn.Sequential(nn.Linear(din, hid), nn.ReLU(), nn.Linear(hid, 1))
        nn.init.xavier_uniform_(self.fc_self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, g, h):
        src, dst = _edges_no_self(g)
        lab = g.ndata["label_unk"][src]
        n = h.shape[0]
        groups = []
        for c in (1, 0, 2):  # fraud, benign, unknown
            m = lab == c
            groups.append(torch.zeros_like(h).index_add_(0, dst[m], h[src[m]]))
        m_fr, m_be, m_unk = groups
        b = torch.sigmoid(self.balance(h))
        return (self.fc_self(h) + self.fr(m_fr, h) + self.be(m_be, h)
                + b * self.fr(m_unk, h) + (1 - b) * self.be(m_unk, h))


class PMP(nn.Module):
    def __init__(self, din, hid, layers=2):
        super().__init__()
        dims = [din] + [hid] * layers
        self.convs = nn.ModuleList(PMPConv(a, b, hid) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, g, x):
        for i, c in enumerate(self.convs):
            x = c(g, x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class CAREConv(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.sim = nn.Linear(din, 1)          # label-aware similarity
        self.lin = nn.Linear(2 * din, dout)

    def forward(self, g, h, p):
        src, dst = _edges_no_self(g)
        s = torch.sigmoid(self.sim(h)).squeeze(-1)
        dist = (s[src] - s[dst]).abs()
        n = h.shape[0]
        deg = torch.zeros(n, dtype=torch.long).index_add_(0, dst, torch.ones_like(dst))
        # keep the ceil(p * deg) closest neighbours of each node
        order = torch.argsort(dst.double() * 4.0 + dist.detach().double().clamp(0, 1) * 2.0, stable=True)
        d_sorted = dst[order]
        start = torch.zeros(n, dtype=torch.long)
        start[1:] = torch.cumsum(deg, 0)[:-1]
        rank = torch.arange(len(order)) - start[d_sorted]
        k = torch.clamp(torch.ceil(p * deg.float()).long(), min=1)
        keep = order[rank < k[d_sorted]]
        ks, kd = src[keep], dst[keep]
        agg = torch.zeros_like(h).index_add_(0, kd, h[ks])
        cnt = torch.zeros(n).index_add_(0, kd, torch.ones(len(kd))).clamp(min=1).unsqueeze(-1)
        out = F.relu(self.lin(torch.cat([h, p * agg / cnt], -1)))
        return out, s, dist[keep].detach().mean() if len(keep) else torch.tensor(0.0)


class CAREGNN(nn.Module):
    TAU, LAMBDA1 = 0.02, 2.0

    def __init__(self, din, hid, layers=2):
        super().__init__()
        dims = [din] + [hid] * layers
        self.convs = nn.ModuleList(CAREConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.p = [0.5] * layers
        self._dist_epoch = [[] for _ in range(layers)]
        self._prev = [None] * layers
        self.aux_loss = torch.tensor(0.0)

    def forward(self, g, x):
        lab = g.ndata.get("label_trn")
        aux = 0.0
        for i, c in enumerate(self.convs):
            x, s, dmean = c(g, x, self.p[i])
            if self.training:
                self._dist_epoch[i].append(float(dmean))
                if lab is not None and (lab >= 0).any():
                    m = lab >= 0
                    aux = aux + F.binary_cross_entropy(s[m].clamp(1e-6, 1 - 1e-6), lab[m].float())
        if self.training:  # the training loop adds this to the loss and resets it
            self.aux_loss = self.aux_loss + self.LAMBDA1 * aux
        return x

    def epoch_end(self):
        # RL-style update of p: up if the mean kept distance went down, else down
        for i in range(len(self.p)):
            cur = float(sum(self._dist_epoch[i]) / max(len(self._dist_epoch[i]), 1))
            if self._prev[i] is not None:
                self.p[i] = min(1.0, max(0.02, self.p[i] + (self.TAU if self._prev[i] - cur >= 0 else -self.TAU)))
            self._prev[i] = cur
            self._dist_epoch[i] = []


EXTRA_BACKBONES = {"pmp": PMP, "caregnn": CAREGNN}


def dt_bin_encode(X, y, train_idx, k=4, min_frac=0.01):
    """DGA-GNN's decision-tree binning: per column, fit a small tree on train nodes and append one-hot bins."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    Xn = X.numpy() if hasattr(X, "numpy") else X
    blocks = [Xn]
    for j in range(Xn.shape[1]):
        t = DecisionTreeClassifier(max_leaf_nodes=k, min_samples_leaf=min_frac, random_state=0)
        t.fit(Xn[train_idx, j:j + 1], y[train_idx])
        thr = np.sort(t.tree_.threshold[t.tree_.feature >= 0])
        b = np.searchsorted(thr, Xn[:, j], side="right")
        oh = np.zeros((len(Xn), k), dtype=np.float32)
        oh[np.arange(len(Xn)), np.minimum(b, k - 1)] = 1.0
        blocks.append(oh)
    return torch.from_numpy(np.concatenate(blocks, 1).astype(np.float32))


class _SageMean(nn.Module):
    # mean aggregator from DGA-GNN, with an optional edge mask

    def __init__(self, din, dout, add_self):
        super().__init__()
        self.fc_self, self.fc_neigh = nn.Linear(din, dout), nn.Linear(din, dout, bias=False)
        self.add_self = add_self
        self.norm = nn.BatchNorm1d(dout)

    def forward(self, g, h, edge_mask=None):
        src, dst = _edges_no_self(g)
        w = torch.ones(len(src)) if edge_mask is None else edge_mask
        n = h.shape[0]
        agg = torch.zeros_like(h).index_add_(0, dst, h[src] * w.unsqueeze(-1))
        deg = torch.zeros(n).index_add_(0, dst, w).unsqueeze(-1)
        rst = self.fc_neigh(agg / (deg + 1e-8))
        if self.add_self:
            rst = rst + self.fc_self(h)
        return self.norm(F.relu(rst))


class DGAGNN(nn.Module):
    """Expects bin-encoded features and g.ndata["gid"]; call set_feedback() with all-node predictions each epoch."""
    Z = 0.025

    def __init__(self, din, hid, layers=2, p=0.3):
        super().__init__()
        mods, d = [], din
        for _ in range(layers):
            mods += [nn.Dropout(p), nn.Linear(d, hid), nn.BatchNorm1d(hid), nn.ReLU()]
            d = hid
        self.emb = nn.Sequential(*mods)
        self.all, self.gp0, self.gp1 = _SageMean(hid, hid, True), _SageMean(hid, hid, False), _SageMean(hid, hid, False)
        self.W_f = nn.Sequential(nn.Linear(hid, hid), nn.Tanh())
        self.W_x = nn.Sequential(nn.Linear(hid, hid), nn.Tanh())
        self.register_buffer("mask_p1", torch.empty(0))  # avg fraud prob over the last 10 epochs
        self._hist = []

    def set_feedback(self, prob_fraud):
        self._hist = (self._hist + [torch.as_tensor(prob_fraud, dtype=torch.float32)])[-10:]
        self.mask_p1 = torch.stack(self._hist).mean(0)

    def forward(self, g, x):
        h = self.emb(x)
        src, _ = _edges_no_self(g)
        if self.mask_p1.numel():
            p1 = self.mask_p1[g.ndata["gid"][src]]
        else:  # no feedback yet
            p1 = torch.ones(len(src))
        hs = torch.stack([self.all(g, h), self.gp0(g, h, (p1 <= self.Z).float()), self.gp1(g, h, (p1 > self.Z).float())], 1)
        att = torch.softmax(torch.bmm(self.W_f(hs), self.W_x(h).unsqueeze(-1)), dim=1)  # [n, 3, 1]
        return (att * hs).sum(1)


EXTRA_BACKBONES["dgagnn"] = DGAGNN
