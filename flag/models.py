"""GNN backbones and the FLAG heads (DGL). Each backbone maps (graph, x) -> hidden reps; target = index 0."""
import dgl
import dgl.function as fn
import scipy.special
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv


class GCN(nn.Module):
    def __init__(self, din, hid, layers=2):
        super().__init__()
        dims = [din] + [hid] * layers
        self.convs = nn.ModuleList(GraphConv(a, b, allow_zero_in_degree=True) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, g, x):
        for i, c in enumerate(self.convs):
            x = c(g, x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GAT(nn.Module):
    def __init__(self, din, hid, layers=2):
        super().__init__()
        dims = [din] + [hid] * layers
        self.convs = nn.ModuleList(GATConv(a, b, num_heads=1, allow_zero_in_degree=True)
                                   for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, g, x):
        for i, c in enumerate(self.convs):
            x = c(g, x).squeeze(1)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x


class GeniePath(nn.Module):
    # Liu et al. 2019: attention for breadth, LSTM gates for depth

    def __init__(self, din, hid, layers=2):
        super().__init__()
        self.inp = nn.Linear(din, hid)
        self.breadth = nn.ModuleList(GATConv(hid, hid, num_heads=1, allow_zero_in_degree=True) for _ in range(layers))
        self.depth = nn.ModuleList(nn.LSTMCell(hid, hid) for _ in range(layers))

    def forward(self, g, x):
        h = self.inp(x)
        c = torch.zeros_like(h)
        for b, d in zip(self.breadth, self.depth):
            t = torch.tanh(b(g, h).squeeze(1))
            h, c = d(t, (h, c))
        return h


def _beta_thetas(d):
    # beta wavelet coefficients, from BWGNN's calculate_theta2
    x = sympy.symbols("x")
    thetas = []
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / scipy.special.beta(i + 1, d + 1 - i))
        coeff = f.all_coeffs()
        thetas.append([float(coeff[d - j]) for j in range(d + 1)])
    return thetas


class _PolyConv(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def forward(self, g, feat):
        with g.local_scope():
            dinv = g.in_degrees().float().clamp(min=1).pow(-0.5).unsqueeze(-1).to(feat)

            def lap(f):  # L = I - D^-1/2 A D^-1/2
                g.ndata["h"] = f * dinv
                g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                return f - g.ndata.pop("h") * dinv

            h = self.theta[0] * feat
            for k in range(1, len(self.theta)):
                feat = lap(feat)
                h = h + self.theta[k] * feat
            return h


class BWGNN(nn.Module):
    # homogeneous BWGNN (Tang et al. 2022), adapted from github.com/squareRoot3/Rethinking-Anomaly-Detection

    def __init__(self, din, hid, layers=2, order=2):
        super().__init__()
        self.convs = nn.ModuleList(_PolyConv(t) for t in _beta_thetas(order))
        self.l1, self.l2 = nn.Linear(din, hid), nn.Linear(hid, hid)
        self.l3 = nn.Linear(hid * len(self.convs), hid)

    def forward(self, g, x):
        h = F.relu(self.l2(F.relu(self.l1(x))))
        h = torch.cat([c(g, h) for c in self.convs], -1)
        return F.relu(self.l3(h))


BACKBONES = {"gcn": GCN, "gat": GAT, "geniepath": GeniePath, "bwgnn": BWGNN}
from flag.models_extra import EXTRA_BACKBONES  # noqa: E402
BACKBONES.update(EXTRA_BACKBONES)


class Encoder(nn.Module):
    """Backbone + optional skip connection, Z = GNN(X, A) + Linear(X) (Eq. 6)."""

    def __init__(self, backbone, din, hid, skip):
        super().__init__()
        self.gnn = BACKBONES[backbone](din, hid)
        self.skip = nn.Linear(din, hid) if skip else None

    def forward(self, g, x, targets):
        z = self.gnn(g, x)
        if self.skip is not None:
            z = z + self.skip(x)
        return z[targets]


class SingleView(nn.Module):
    # single feature view (baseline, +text, no-LLM ablations)

    def __init__(self, backbone, din, hid=64, skip=False):
        super().__init__()
        self.enc = Encoder(backbone, din, hid, skip)
        self.out = nn.Linear(hid, 1)

    def forward(self, g, feats, targets):
        return self.out(self.enc(g, feats[0], targets)).squeeze(-1)


class FLAGFusion(nn.Module):
    """Raw and discriminative views through a shared encoder, fused with attention (Sec. 3.4).
    The paper doesn't specify the attention, so this is HAN-style view attention."""

    def __init__(self, backbone, din, hid=64, skip=True):
        super().__init__()
        self.enc = Encoder(backbone, din, hid, skip)
        self.att = nn.Linear(hid, hid)
        self.q = nn.Parameter(torch.randn(hid) * 0.1)
        self.out = nn.Linear(hid, 1)

    def forward(self, g, feats, targets):
        zs = torch.stack([self.enc(g, f, targets) for f in feats], 1)  # [B, views, hid]
        a = torch.softmax(torch.tanh(self.att(zs)) @ self.q, dim=1).unsqueeze(-1)
        return self.out((a * zs).sum(1)).squeeze(-1)


def batch_graphs(subgraphs, edges_cache, idx):
    """Returns (batched graph, global node ids, target positions)."""
    gs = [edges_cache[i] for i in idx]
    g = dgl.batch(gs)
    nodes = torch.cat([torch.from_numpy(subgraphs[i]) for i in idx])
    sizes = torch.tensor([len(subgraphs[i]) for i in idx])
    targets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]])
    return g, nodes, targets
