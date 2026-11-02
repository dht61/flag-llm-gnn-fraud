"""CPU version of the raw-text SBERT step in gen_texts.py (useful when no GPU is free)."""
import hashlib
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from flag.paths import SBERT, TXT

torch.manual_seed(0)
m = SentenceTransformer(SBERT, device="cpu")
for ds in ("reddit", "instagram"):
    out = TXT / f"{ds}_raw_mpnet.npy"
    if out.exists():
        print("[skip]", out)
        continue
    texts = [json.loads(l)["text"] for l in open(TXT / f"{ds}_raw.jsonl")]
    e = m.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.save(out, e)
    print("[sbert-cpu]", ds, e.shape, hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)
