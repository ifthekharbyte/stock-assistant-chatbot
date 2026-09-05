"""
download_embedding_model.py

Ported near-verbatim from 02-vector-search/embed/download.py. Run this
once to fetch the ONNX embedding model used by embedder.py /
knowledge_base.py -- after that it's cached locally under models/ and
this script doesn't need to run again (or re-run on every container
build/start).

Only huggingface-hub is needed for this script; it's not needed at
runtime by embedder.py itself.

Run with: python download_embedding_model.py
Or: make download-model
"""

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]


def download(repo, dest="models"):
    dest = Path(dest) / repo
    dest.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id=repo)
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model found in {repo}")

    for remote, local in [
        ("tokenizer.json", "tokenizer.json"),
        (onnx_file, "model.onnx"),
    ]:
        src = hf_hub_download(repo_id=repo, filename=remote)
        dst = dest / local
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")

    onnx_ext = onnx_file + "_data"
    if onnx_ext in files:
        src = hf_hub_download(repo_id=repo, filename=onnx_ext)
        dst = dest / "model.onnx_data"
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")


if __name__ == "__main__":
    # Same model as before the ONNX swap (384-dim, all-MiniLM-L6-v2),
    # just the ONNX build of it -- so knowledge_base.py's dimension-
    # agnostic dot-product logic needed no changes. See
    # 02-vector-search/lessons/09-onnx-embedder.md for other available
    # models if you want to swap this later.
    download("Xenova/all-MiniLM-L6-v2")
