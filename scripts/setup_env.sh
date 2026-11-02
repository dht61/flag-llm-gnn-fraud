#!/bin/bash
# one env for both the GNN side (DGL, PyG) and the LLM side (transformers, peft, vLLM).
# vLLM 0.6.3 pins torch 2.4.0; the HF stack is pinned too, otherwise vllm drags in transformers 5.x.
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^flag\s"; then
  conda create -y -n flag -c conda-forge --override-channels python=3.10 pip
fi
conda activate flag
PIN="torch==2.4.0 numpy==1.26.4"
HF="transformers==4.46.3 tokenizers==0.20.3 huggingface-hub==0.26.2 datasets==3.1.0 accelerate==1.1.1 fsspec==2024.9.0"
export DGLBACKEND=pytorch
pip install --no-cache-dir $PIN --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple
pip install --no-cache-dir $PIN $HF vllm==0.6.3.post1
pip install --no-cache-dir $PIN dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install --no-cache-dir $PIN $HF torch_geometric==2.6.1 peft==0.13.2 \
  sentence-transformers==3.2.1 scikit-learn==1.5.2 pandas==2.2.3 scipy==1.13.1 pyyaml==6.0.2 matplotlib==3.9.2
# word2vec features use gensim (installed separately so it can't move numpy)
pip install --no-cache-dir numpy==1.26.4 gensim==4.3.3
