"""
embedder.py

Ported near-verbatim from 02-vector-search/embed/embedder.py
(see 02-vector-search/lessons/09-onnx-embedder.md). This replaces the
sentence-transformers-based embedding originally used in
knowledge_base.py.

Why: sentence-transformers depends on PyTorch, and PyPI's default
PyTorch wheels bundle full CUDA/GPU support regardless of whether the
machine has a GPU -- the course lesson measured this at 4.8GB / 58
packages for a sentence-transformers virtualenv vs. 147MB / 27 packages
for the ONNX Runtime equivalent, same model, same results. Verified
directly while making this change: this file + download_embedding_model.py
produce the same 384-dim, unit-norm vectors as sentence-transformers'
all-MiniLM-L6-v2 did, with no torch/nvidia packages anywhere in the
dependency tree.

Requires the model files to exist locally first -- run
`download_embedding_model.py` (or `make download-model`) once before
using this.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Embedder:
    def __init__(self, path="models/Xenova/all-MiniLM-L6-v2"):
        path = Path(path)
        if not (path / "model.onnx").exists():
            raise FileNotFoundError(
                f"No ONNX model found at {path}. Run `python download_embedding_model.py` "
                "(or `make download-model`) once before using the knowledge base's vector search."
            )
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text, normalize=True):
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts, normalize=True):
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled
