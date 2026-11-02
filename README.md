# FLAG: LLM-enhanced GNN fraud detection (reproduction)

Re-implementation of

> C. Yang, H. Liu, D. Wang, Z. Zhang, C. Yang, C. Shi. *FLAG: Fraud Detection with LLM-enhanced Graph Neural
> Network.* KDD 2025. [doi:10.1145/3711896.3737220](https://doi.org/10.1145/3711896.3737220)

The paper comes from BUPT and Ant Group, and no code was released. Only open models are used here.

## What FLAG does

FLAG wraps around an existing graph fraud detector and adds three things:

1. **Semantic neighbour sampling.** For each user, keep the 10 two-hop neighbours whose text is most similar
   (Sentence-BERT cosine similarity), instead of random ones.
2. **LLM summaries.** An LLM rewrites each user's text into a short "discriminative" summary: the part that matters
   for the label. Its embedding becomes a second feature view, fused with the raw-text view by attention.
3. **Skip connection.** Z = GNN(X, A) + Linear(X).

**FLAG\*** additionally fine-tunes the LLM so its summaries help the GNN more.

The paper tests 7 backbones: GCN, GAT, GeniePath, CARE-GNN, BWGNN, DGA-GNN and PMP. Each one is run as a baseline
(shallow word2vec features), +text (raw-text embeddings), +FLAG and +FLAG\*.

**Data.** The paper's own financial dataset (Alipay Huabei) is private. The public experiments, which are what this
repo reproduces, use two GLBench social graphs:
- **Reddit:** popular vs normal users;
- **Instagram:** commercial vs normal accounts.

In both, the minority class is subsampled so it looks like rare fraud.

## Changes from the paper

- **LLM:** Llama-3-8B-Instruct instead of Gemma-9B. The text encoder is `all-mpnet-base-v2`.
- **FLAG\* fine-tuning.** The paper's loss is computed on embeddings of *generated* text, so gradients can't reach
  the LLM as written. I used rejection-sampling fine-tuning instead. Sample 4 candidate summaries per training user,
  score them with the paper's loss under a GNN scorer, then LoRA-tune on the best ones (`finetune_llm.py`).
- **Class ratio.** The paper says "about 1:10", but its all-normal baseline rows imply 16.7% minority on Reddit and
  10.3% on Instagram. Those are the ratios used here. With them, our all-normal F1 matches the paper's collapsed
  baseline rows exactly (45.46 / 47.29).
- **Baseline features.** These are word2vec trained on each dataset's own text, since the paper doesn't say which
  model it used. They turn out stronger than the paper's (higher AUC), so the baseline rows don't line up.
- **Other details** the paper leaves unspecified got standard choices: the attention form, the loss weights and
  the split ratios.

## Results

Every cell is 25 runs (5 data seeds × 5 inits). The deltas compare paired runs (same seed and init) and are averaged
over the 14 (backbone, dataset) cells.

| Claim | Paper | Ours |
|---|---|---|
| +FLAG vs +text (F1 / AUC) | +1.6 / +1.5 | **+1.1 / +0.4**, positive in 9/14 cells |
| Where it helps | all backbones | mostly PMP, DGA-GNN, GeniePath |
| FLAG\* vs FLAG (F1 / AUC) | +0.8 / +0.4 | **−0.04 / −0.2** |
| Table 5: LLM component (F1) | +1.0 | +0.1 |
| Table 5: skip connection (F1) | +0.3 | **+0.9** |

**Reading the results:**
- The direction of the main claim partly holds: LLM summaries help on F1 for some detectors. The size of the gain,
  especially on AUC, doesn't.
- In the paper's own ablation, most of the gain comes from the skip connection, not the LLM text.
- Fine-tuning the LLM (our version of FLAG\*) adds nothing.
- These tasks are close to chance (AUC 50–62), so all the effects are 1–3 points, with run-to-run spread of a
  similar size.

The full Table 4 / 5 comparison is in [`results/main/summary.md`](results/main/summary.md). The paired deltas come
from:

```bash
python compare_variants.py flag text
python compare_variants.py flag_star flag
python compare_variants.py --table5
```

## Running it

`FLAG_DATA` (default `./data`) holds everything large. `FLAG_LLM` points to the LLM; it defaults to the Hugging Face
id, which is gated, so a local copy works too. Run the scripts from the repo root. Only text generation and
fine-tuning need a GPU.

```bash
bash scripts/setup_env.sh              # conda env "flag": torch 2.4, DGL 2.4, PyG 2.6, vLLM 0.6.3
bash scripts/download_data.sh          # GLBench reddit.pt / instagram.pt
sbatch scripts/prep_data.sh            # imbalanced versions, 5 data seeds
sbatch scripts/w2v.sh                  # baseline features
sbatch scripts/gen_texts.sh            # GPU: SBERT + Llama summaries (~45 min on one A100)
sbatch --array=0-39 --export=ALL,GRID=baseline_A scripts/run_grid.sh
sbatch --array=0-199%16 --export=ALL,GRID=phase2 scripts/run_grid.sh    # remaining baselines, +text, ablations
sbatch --array=0-119%16 --export=ALL,GRID=phase3 scripts/run_grid.sh    # +FLAG, SS+LLM ablation
sbatch scripts/finetune.sh             # GPU: FLAG* fine-tuning, one task per (dataset, seed)
sbatch --array=0-34 --export=ALL,GRID=star_ig,EXTRA="--disc_tag _ft_s{seed}" scripts/run_grid.sh
sbatch --array=0-34 --export=ALL,GRID=star_rd,EXTRA="--disc_tag _ft_s{seed}" scripts/run_grid.sh
python summarize_flag.py --md results/main/summary.md
```

A single run, for example:

```bash
python run_flag.py --dataset instagram --backbone pmp --variant flag --seed 0 --init 0 --out results/test
```

GNN training runs on CPU and is deterministic. Every run stores sha256 fingerprints of the model and predictions.

## Layout

```
flag/              sampling, backbones (GCN/GAT/GeniePath/BWGNN + PMP/CARE-GNN/DGA-GNN), FLAG heads, paths
prep_data.py       imbalanced datasets + splits
w2v_features.py    baseline features
gen_texts.py       LLM summaries + SBERT embeddings
finetune_llm.py    FLAG* (rejection-sampling LoRA)
run_flag.py        train/eval one (dataset, backbone, variant, seed, init)
grids/             the run grids used
results/main/      metrics.json for all 2,150 runs + summary
```

The PMP, CARE-GNN, DGA-GNN and BWGNN layers are re-implemented after their official repos (linked in the code).
The generated texts aren't included because they're derived from users' posts and bios.
