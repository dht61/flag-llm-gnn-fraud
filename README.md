# flag-llm-gnn-fraud

Reproduction of FLAG (Yang et al., *FLAG: Fraud Detection with LLM-enhanced Graph Neural Network*, KDD 2025,
[paper](https://doi.org/10.1145/3711896.3737220)).

FLAG sits on top of an existing GNN fraud detector and adds three things:
- it picks each user's neighbours by text similarity;
- an LLM rewrites each user's text into a short summary, which becomes a second feature view;
- it adds a skip connection.

FLAG\* also fine-tunes the LLM.

The paper's financial dataset is private, so this repo covers only the public experiments. Those use the Reddit and
Instagram graphs from GLBench, with the minority class subsampled so it looks like fraud. No code was released, so
I used open models: Llama-3-8B-Instruct as the LLM (the paper used Gemma-9B) and all-mpnet-base-v2 for embeddings.

There are 7 backbones: GCN, GAT, GeniePath, CARE-GNN, BWGNN, DGA-GNN and PMP. Each setting has 25 runs (5 data
seeds x 5 inits), 2,150 runs in total.

## Results

Each number is the average gain over the 14 backbone/dataset pairs, comparing paired runs:

| | paper | here |
|---|---|---|
| FLAG vs raw text features (F1 / AUC) | +1.6 / +1.5 | +1.1 / +0.4 |
| FLAG\* vs FLAG (F1 / AUC) | +0.8 / +0.4 | -0.04 / -0.2 |

FLAG does help F1 for some backbones, mostly PMP, DGA-GNN and GeniePath. But the gain is smaller than the paper
reports, and there's very little of it on AUC. In the ablation, most of the improvement comes from the skip
connection (+0.9 F1), not the LLM summaries (+0.1). Fine-tuning the LLM didn't add anything. These tasks are hard
(AUC 50-62), so every effect here is only 1-3 points.

This repo differs from the paper in three ways:
- **FLAG\* fine-tuning:** the paper's FLAG\* loss can't backprop through generated text. I used rejection-sampling
  LoRA instead (`finetune_llm.py`).
- **Class ratios:** the paper's numbers imply ratios of 16.7% (Reddit) and 10.3% (Instagram), and those are what
  I use.
- **Baseline features:** word2vec trained on each dataset.

Full tables are in `results/main/summary.md`. The paired comparisons come from `compare_variants.py`.

## Running

Large files go under `FLAG_DATA` (default `./data`). `FLAG_LLM` can point to a local copy of the Llama weights.
Only text generation and fine-tuning need a GPU.

```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh
sbatch scripts/prep_data.sh
sbatch scripts/w2v.sh
sbatch scripts/gen_texts.sh
sbatch --array=0-39 --export=ALL,GRID=baseline_A scripts/run_grid.sh
sbatch --array=0-199%16 --export=ALL,GRID=phase2 scripts/run_grid.sh
sbatch --array=0-119%16 --export=ALL,GRID=phase3 scripts/run_grid.sh
sbatch scripts/finetune.sh
sbatch --array=0-34 --export=ALL,GRID=star_ig,EXTRA="--disc_tag _ft_s{seed}" scripts/run_grid.sh
sbatch --array=0-34 --export=ALL,GRID=star_rd,EXTRA="--disc_tag _ft_s{seed}" scripts/run_grid.sh
python summarize_flag.py --md results/main/summary.md
```

To do a single run:

```bash
python run_flag.py --dataset instagram --backbone pmp --variant flag --seed 0 --init 0 --out results/test
```

The PMP, CARE-GNN, DGA-GNN and BWGNN layers are re-implemented from their official repos; the links are in the
code. The LLM summaries aren't included, because they're generated from users' posts and bios.
