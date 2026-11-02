#!/bin/bash
# GLBench reddit.pt / instagram.pt from the GLBench Google Drive folder (needs `pip install gdown`)
set -eo pipefail
OUT=${FLAG_DATA:-$PWD/data}/raw/glbench
mkdir -p $OUT
python - "$OUT" <<'PY'
import os, sys, gdown
out = sys.argv[1]
folder = "https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj"
for f in gdown.download_folder(folder, output=out, skip_download=True, quiet=True):
    name = os.path.basename(f.path)
    if name in ("reddit.pt", "instagram.pt") and not os.path.exists(os.path.join(out, name)):
        gdown.download(id=f.id, output=os.path.join(out, name), quiet=False)
PY
ls -l $OUT
