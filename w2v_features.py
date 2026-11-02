"""Shallow baseline features: skip-gram word2vec trained on each dataset's own texts, averaged per node.

The paper cites word2vec for its baseline but doesn't say which model, so we train one.
(GLBench's own `x` is a 4096-d LLM embedding, not shallow, so it isn't used.)

    python w2v_features.py dump    # needs torch + PyG
    python w2v_features.py train   # needs gensim
"""
import json
import re
import sys

import numpy as np

from flag.paths import RAW, TXT

DATASETS = ["reddit", "instagram"]
DIM, WINDOW, MIN_COUNT, EPOCHS, SEED = 300, 5, 2, 10, 0


def dump():
    import torch
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    TXT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        texts = torch.load(RAW / f"{ds}.pt", weights_only=True).raw_texts
        with open(TXT / f"{ds}_raw.jsonl", "w") as f:
            for i, t in enumerate(texts):
                f.write(json.dumps({"glbench_id": i, "text": t}) + "\n")
        print("[dump]", ds, len(texts), flush=True)


def tokenize(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def train():
    from gensim.models import Word2Vec
    for ds in DATASETS:
        docs = [tokenize(json.loads(l)["text"]) for l in open(TXT / f"{ds}_raw.jsonl")]
        m = Word2Vec(docs, vector_size=DIM, window=WINDOW, min_count=MIN_COUNT, sg=1, epochs=EPOCHS, seed=SEED,
                     workers=1)
        feats = np.zeros((len(docs), DIM), dtype=np.float32)
        empty = 0
        for i, d in enumerate(docs):
            v = [m.wv[w] for w in d if w in m.wv]
            if v:
                feats[i] = np.mean(v, 0)
            else:
                empty += 1
        np.save(TXT / f"{ds}_w2v.npy", feats)
        print("[w2v]", ds, "vocab", len(m.wv), "nodes", len(docs), "no-known-word nodes", empty, flush=True)


if __name__ == "__main__":
    {"dump": dump, "train": train}[sys.argv[1]]()
