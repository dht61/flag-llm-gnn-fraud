"""FLAG* (Sec. 3.3): fine-tune the LLM so its summaries are more useful to the GNN.

The paper's loss sits on embeddings of *generated* text, so it can't be backpropagated into the LLM as written.
We do rejection-sampling fine-tuning instead, one run per (dataset, data seed):

  gnn     train a GCN skip-GNN scorer on the zero-shot disc/res embeddings with Eq. 10
  sample  K=4 disc and K=4 res candidates per train node (T=0.7)
  score   pick the (disc, res) pair with the lowest Eq. 10 loss under the scorer
  sft     LoRA (r=16) SFT on the picked texts, 1 epoch
  regen   regenerate disc texts for every node with the tuned model, SBERT-encode

Only train labels are used. One round of alternation.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from flag.paths import CACHE, LLM, PROC, SBERT, TXT, WORK

ROOT = Path(__file__).resolve().parent
K, TEMP, L1, L2 = 4, 0.7, 1.0, 1.0
MAX_CHARS = 12000  # a few Reddit users are longer than the 4096-token context otherwise


def wdir(a):
    d = WORK / f"{a.dataset}_s{a.seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_seed(a):
    d = torch.load(PROC / a.data_tag / f"{a.dataset}_s{a.seed}.pt", weights_only=False)
    sub = pickle.loads((CACHE / f"{a.data_tag}_{a.dataset}_s{a.seed}_ss_n10_d0.0.pkl").read_bytes())
    return d, sub


def build_graphs(d, sub):
    import dgl
    from flag.sampling import adjacency_lists, induced_edges
    adj_set = [set(x.tolist()) for x in adjacency_lists(d["edge_index"], len(d["y"]))]
    gs = []
    for nodes in sub:
        s, t = induced_edges(nodes, adj_set)
        gs.append(dgl.add_self_loop(dgl.graph((torch.from_numpy(s), torch.from_numpy(t)), num_nodes=len(nodes))))
    return gs


class Scorer(torch.nn.Module):
    def __init__(self, din, hid=64):
        super().__init__()
        from flag.models import Encoder
        self.enc = Encoder("gcn", din, hid, skip=True)
        self.out = torch.nn.Linear(hid, 1)

    def z(self, g, x, tg):
        return self.enc(g, x, tg)


def eq10_terms(model, zd, zr, y):
    # per-node terms of Eq. 10: BCE on disc, KL(res || uniform), (z_d . z_r)^2
    import torch.nn.functional as F
    ld = F.binary_cross_entropy_with_logits(model.out(zd).squeeze(-1), y, reduction="none")
    pr = torch.sigmoid(model.out(zr).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
    lr = pr * torch.log(pr / 0.5) + (1 - pr) * torch.log((1 - pr) / 0.5)
    lo = (zd * zr).sum(-1) ** 2
    return ld, lr, lo


def stage_gnn(a):
    from flag.models import batch_graphs
    d, sub = load_seed(a)
    gid, y = d["glbench_ids"].numpy(), d["y"].float()
    xd = torch.from_numpy(np.load(TXT / f"{a.dataset}_disc_mpnet.npy")[gid]).float()
    xr = torch.from_numpy(np.load(TXT / f"{a.dataset}_res_mpnet.npy")[gid]).float()
    gs = build_graphs(d, sub)
    tr = np.where(d["train_mask"].numpy())[0]
    va = np.where(d["val_mask"].numpy())[0]
    torch.manual_seed(0)
    m = Scorer(xd.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    rng = np.random.default_rng(0)

    def val_loss():  # used to pick the best scorer epoch
        m.eval()
        tot = []
        with torch.no_grad():
            for i in range(0, len(va), 512):
                idx = va[i:i + 512]
                g, nodes, tg = batch_graphs(sub, gs, idx)
                ld, lr, lo = eq10_terms(m, m.z(g, xd[nodes], tg), m.z(g, xr[nodes], tg), y[idx])
                tot.append((ld + L1 * lr + L2 * lo).sum())
        m.train()
        return float(sum(tot)) / len(va)

    best = (float("inf"), None, -1)
    for ep in range(a.gnn_epochs):
        tot = 0.0
        order = rng.permutation(tr)
        for i in range(0, len(order), 10):
            idx = order[i:i + 10]
            g, nodes, tg = batch_graphs(sub, gs, idx)
            ld, lr, lo = eq10_terms(m, m.z(g, xd[nodes], tg), m.z(g, xr[nodes], tg), y[idx])
            loss = (ld + L1 * lr + L2 * lo).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # the orthogonality term blows up without this
            opt.step()
            tot += float(loss)
        vl = val_loss()
        print(json.dumps({"stage": "gnn", "epoch": ep, "loss": tot / max(1, len(tr) // 10), "val": vl}), flush=True)
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, ep)
    print(json.dumps({"stage": "gnn", "best_epoch": best[2], "best_val": best[0]}), flush=True)
    torch.save(best[1], wdir(a) / "scorer.pt")


def prompts(a, kind, texts):
    sys.path.insert(0, str(ROOT))
    from gen_texts import DESC, task
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM)
    label = "posts" if a.dataset == "reddit" else "personal introduction"
    sys_msg = f"{DESC[a.dataset]} {task(kind, a.dataset)}"
    return [tok.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": f"User's {label}: {t.strip()[:MAX_CHARS] or '(empty)'}\n\nReply with only the "
                                     f"{'discriminative' if kind == 'disc' else 'residual'} text for this user: one or "
                                     f"two sentences, no preamble, no quotation marks, no explanation."}],
        tokenize=False, add_generation_prompt=True) for t in texts]


def raw_texts(a):
    return [json.loads(l)["text"] for l in open(TXT / f"{a.dataset}_raw.jsonl")]


def stage_sample(a):
    from vllm import LLM as VLLM, SamplingParams
    from gen_texts import clean
    d, _ = load_seed(a)
    gid = d["glbench_ids"].numpy()
    tr_g = gid[np.where(d["train_mask"].numpy())[0]]
    raw = raw_texts(a)
    llm = VLLM(model=LLM, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=0.85, max_model_len=4096)
    out = {}
    for kind in ("disc", "res"):
        ps = prompts(a, kind, [raw[i] for i in tr_g])
        res = llm.generate(ps, SamplingParams(n=K, temperature=TEMP, max_tokens=128, seed=a.seed), use_tqdm=False)
        out[kind] = [[clean(c.text) for c in r.outputs] for r in res]
    (wdir(a) / "candidates.json").write_text(json.dumps({"glbench_ids": tr_g.tolist(), **out}))
    print("[sample]", a.dataset, a.seed, len(tr_g), "train nodes x", K, "x 2", flush=True)


def stage_score(a):
    from sentence_transformers import SentenceTransformer
    from flag.models import batch_graphs
    d, sub = load_seed(a)
    gid, y = d["glbench_ids"].numpy(), d["y"].float()
    cand = json.loads((wdir(a) / "candidates.json").read_text())
    for k in ("disc", "res"):  # pad if vLLM rejected a prompt
        cand[k] = [(cs + [""] * K)[:K] for cs in cand[k]]
    enc = SentenceTransformer(SBERT, device="cuda")
    E = {k: torch.from_numpy(enc.encode([c for cs in cand[k] for c in cs], batch_size=256, convert_to_numpy=True))
         .float().view(len(cand[k]), K, -1) for k in ("disc", "res")}
    xd = torch.from_numpy(np.load(TXT / f"{a.dataset}_disc_mpnet.npy")[gid]).float()
    xr = torch.from_numpy(np.load(TXT / f"{a.dataset}_res_mpnet.npy")[gid]).float()
    gs = build_graphs(d, sub)
    m = Scorer(xd.shape[1]); m.load_state_dict(torch.load(wdir(a) / "scorer.pt")); m.eval()
    pos = {int(g): i for i, g in enumerate(gid)}
    picks = []
    with torch.no_grad():
        for j, g_id in enumerate(cand["glbench_ids"]):
            v = pos[g_id]
            g, nodes, tg = batch_graphs(sub, gs, [v])
            zd, zr = [], []
            for c in range(K):
                fd, fr = xd[nodes].clone(), xr[nodes].clone()
                fd[0], fr[0] = E["disc"][j, c], E["res"][j, c]  # target node is index 0
                zd.append(m.z(g, fd, tg)); zr.append(m.z(g, fr, tg))
            zd, zr = torch.cat(zd), torch.cat(zr)  # [K, hid]
            ld, lr, _ = eq10_terms(m, zd, zr, y[v].repeat(K))
            lo = (zd @ zr.T) ** 2  # [K, K]
            tot = ld[:, None] + L1 * lr[None, :] + L2 * lo
            bi, bj = divmod(int(torch.argmin(tot)), K)
            picks.append({"glbench_id": g_id, "disc": cand["disc"][j][bi], "res": cand["res"][j][bj],
                          "loss": float(tot[bi, bj])})
    (wdir(a) / "picks.jsonl").write_text("".join(json.dumps(p) + "\n" for p in picks))
    print("[score]", len(picks), "picks; mean loss", round(float(np.mean([p["loss"] for p in picks])), 4), flush=True)


def stage_sft(a):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(LLM)
    tok.pad_token = tok.eos_token
    picks = [json.loads(l) for l in open(wdir(a) / "picks.jsonl")]
    raw = raw_texts(a)
    pairs = []
    for kind in ("disc", "res"):
        ps = prompts(a, kind, [raw[p["glbench_id"]] for p in picks])
        pairs += [(pr, p[kind] + tok.eos_token) for pr, p in zip(ps, picks)]
    rng = np.random.default_rng(a.seed)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    model = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    bs = 8
    for step, i in enumerate(range(0, len(pairs), bs)):
        batch = pairs[i:i + bs]
        ids, labels = [], []
        for pr, tgt in batch:
            p_ids = tok(pr, add_special_tokens=False)["input_ids"][-1024:]
            t_ids = tok(tgt, add_special_tokens=False)["input_ids"][:160]
            ids.append(p_ids + t_ids); labels.append([-100] * len(p_ids) + t_ids)
        L = max(map(len, ids))
        x = torch.full((len(ids), L), tok.pad_token_id); lab = torch.full((len(ids), L), -100)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for r, (s, l) in enumerate(zip(ids, labels)):
            x[r, :len(s)] = torch.tensor(s); lab[r, :len(l)] = torch.tensor(l); att[r, :len(s)] = 1
        loss = model(input_ids=x.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(json.dumps({"stage": "sft", "step": step, "of": len(pairs) // bs, "loss": float(loss)}), flush=True)
    model.save_pretrained(wdir(a) / "lora")


def stage_regen(a):
    from sentence_transformers import SentenceTransformer
    from vllm import LLM as VLLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from gen_texts import clean
    d, _ = load_seed(a)
    gid = d["glbench_ids"].numpy()
    raw = raw_texts(a)
    llm = VLLM(model=LLM, dtype="bfloat16", seed=0, gpu_memory_utilization=0.80, max_model_len=4096,
               enable_lora=True, max_lora_rank=16)
    res = llm.generate(prompts(a, "disc", [raw[i] for i in gid]), SamplingParams(temperature=0.0, max_tokens=128, seed=0),
                       lora_request=LoRARequest("flagstar", 1, str(wdir(a) / "lora")), use_tqdm=False)
    texts = [clean(r.outputs[0].text) for r in res]
    (wdir(a) / "disc_ft.jsonl").write_text("".join(json.dumps({"glbench_id": int(g), "text": t}) + "\n"
                                                   for g, t in zip(gid, texts)))
    del llm
    torch.cuda.empty_cache()
    base = np.load(TXT / f"{a.dataset}_disc_mpnet.npy")
    e = SentenceTransformer(SBERT, device="cuda").encode(texts, batch_size=256, convert_to_numpy=True).astype(np.float32)
    base[gid] = e
    np.save(TXT / f"{a.dataset}_disc_ft_s{a.seed}_mpnet.npy", base)
    print("[regen]", a.dataset, a.seed, len(texts), "texts; mean words", round(float(np.mean([len(t.split()) for t in texts])), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gnn", "sample", "score", "sft", "regen", "all"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_tag", default="implied_tr40")
    ap.add_argument("--gnn_epochs", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "all":  # separate processes so vLLM and HF don't fight over GPU memory
        for st in ("gnn", "sample", "score", "sft", "regen"):
            subprocess.run([sys.executable, __file__, st, "--dataset", a.dataset, "--seed", str(a.seed),
                            "--data_tag", a.data_tag, "--gnn_epochs", str(a.gnn_epochs)], check=True)
        return
    sys.path.insert(0, str(ROOT))
    {"gnn": stage_gnn, "sample": stage_sample, "score": stage_score, "sft": stage_sft, "regen": stage_regen}[a.stage](a)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
