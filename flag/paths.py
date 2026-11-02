import os
from pathlib import Path

# everything large lives under FLAG_DATA (default: ./data)
DATA = Path(os.environ.get("FLAG_DATA", Path(__file__).resolve().parent.parent / "data"))
RAW = DATA / "raw" / "glbench"
PROC = DATA / "processed"
TXT = DATA / "texts"
CACHE = DATA / "subgraphs"
WORK = DATA / "ft"

LLM = os.environ.get("FLAG_LLM", "meta-llama/Meta-Llama-3-8B-Instruct")
SBERT = "sentence-transformers/all-mpnet-base-v2"
