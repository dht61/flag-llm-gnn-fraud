"""LLM texts + SBERT embeddings for every GLBench node (GPU, done once, independent of the data seed).

  raw   SBERT of the raw text ("+text", and used for similarity sampling)
  disc  Llama-3-8B "discriminative" summary of the node's own text (Table 1 prompt), then SBERT ("+FLAG")
  res   "residual" summary (Table 2 prompt), only needed for FLAG* fine-tuning

Greedy decoding. Prompts follow the paper's Table 1/2 wording; Instagram uses the same template.
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np
import torch

from flag.paths import LLM, RAW, SBERT
from flag.paths import TXT as OUT

DESC = {  # <Dataset Description>
    "reddit": "You are provided with a list of Reddit users' posts. Each user is classified as either popular or normal "
              "based on their interactions and content.",
    "instagram": "You are provided with a list of Instagram users' personal introductions. Each user is classified as "
                 "either commercial or normal based on their account type.",
}
CLASSES = {"reddit": "popular or normal", "instagram": "commercial or normal"}


def task(kind, name):
    c = CLASSES[name]
    if kind == "disc":
        return (f"Your task is to generate a brief discriminative text for each user that directly relates to "
                f"distinguishing them as either {c}. Ensure that the generated text highlights specific features that "
                f"help differentiate the user's classification while avoiding any general or irrelevant information.")
    return (f"Your task is to generate a brief residual text for each user that captures the portions of their text "
            f"that do not directly relate to distinguishing them as either {c}. The residual text should highlight "
            f"non-discriminative features that are not helpful in differentiating the user's classification. Focus on "
            f"generating text that includes general background details, unrelated observations, or information that "
            f"does not influence the classification task.")


def load_texts(name):
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    return list(torch.load(RAW / f"{name}.pt", weights_only=True).raw_texts)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(texts, out_path, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(SBERT, device=device)
    e = m.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.save(out_path, e)
    return e.shape


def generate(name, kind, texts, max_tokens, limit):
    from vllm import SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM)
    sys_msg = f"{DESC[name]} {task(kind, name)}"
    label = "posts" if name == "reddit" else "personal introduction"
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": f"User's {label}: {t.strip() or '(empty)'}\n\nReply with only the {'discriminative' if kind == 'disc' else 'residual'} text for this user: one or two sentences, no preamble, no quotation marks, no explanation."}],
        tokenize=False, add_generation_prompt=True) for t in texts[:limit]]
    llm = generate.llm
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [clean(o.outputs[0].text) for o in outs], sys_msg


def clean(text):
    # strip "Here is ..." style lead-ins and trailing explanations
    t = text.strip()
    m = re.search(r'"([^"]{20,})"', t)
    if m:
        return m.group(1).strip()
    lines = [l for l in t.split("\n\n") if l.strip()]
    if len(lines) > 1 and re.match(r"(?i)^(here is|here's|based on|the (discriminative|residual) text)", lines[0]):
        lines = lines[1:]
    return lines[0].strip() if lines else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["reddit", "instagram"])
    ap.add_argument("--kinds", nargs="+", default=["disc", "res"])
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="first N nodes only (smoke)")
    a = ap.parse_args()
    out = OUT if not a.limit else OUT / f"smoke{a.limit}"
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}

    texts = {n: load_texts(n) for n in a.datasets}
    if a.limit:
        texts = {n: t[:a.limit] for n, t in texts.items()}
    for n, t in texts.items():
        p = out / f"{n}_raw_mpnet.npy"
        if not p.exists():
            print("[sbert] raw", n, encode(t, p, "cuda"), flush=True)
        manifest[p.name] = sha(p)
    torch.cuda.empty_cache()  # free SBERT memory before vLLM allocates

    from vllm import LLM as VLLM
    generate.llm = VLLM(model=LLM, dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=4096,
                        enforce_eager=False)
    for n, t in texts.items():
        for kind in a.kinds:
            jp = out / f"{n}_{kind}.jsonl"
            if not jp.exists():
                gen, sys_msg = generate(n, kind, t, a.max_tokens, len(t))
                with open(jp, "w") as f:
                    for i, g in enumerate(gen):
                        f.write(json.dumps({"glbench_id": i, "text": g}) + "\n")
                (out / f"{n}_{kind}_prompt.txt").write_text(sys_msg + "\n")
                print("[llm]", n, kind, len(gen), "texts; mean words",
                      round(float(np.mean([len(g.split()) for g in gen])), 1), flush=True)
            manifest[jp.name] = sha(jp)
    del generate.llm
    torch.cuda.empty_cache()
    for n in texts:
        for kind in a.kinds:
            p = out / f"{n}_{kind}_mpnet.npy"
            if not p.exists():
                gen = [json.loads(l)["text"] for l in open(out / f"{n}_{kind}.jsonl")]
                print("[sbert]", kind, n, encode(gen, p, "cuda"), flush=True)
            manifest[p.name] = sha(p)
    man_path.write_text(json.dumps(manifest, indent=1))
    print("[manifest]", json.dumps(manifest, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
