"""Pre-compute frozen text embeddings for object categories (SEEK-style RSN encoder).

SEEK's Relational Semantic Network feeds the target object's *name* through a frozen
pre-trained text encoder (BGE-small) before the MLP. That is what gives the RSN its
open-vocabulary behavior: an object never seen in training still lands near
semantically similar objects in embedding space, so the MLP produces a sensible
prediction instead of failing.

Embeddings are computed once and cached, so training never loads the encoder and the
encoder's weights stay frozen (as in the paper).

Category names are underscore-separated ("wall_mounted_tv"), which the encoder would
otherwise tokenize poorly, so underscores become spaces before encoding.
"""

import os as _os, sys as _sys
# Runnable as a script from anywhere. The other stages are sibling folders under src/, which
# are not on the path when this file is the one being executed, so find the repo root by
# marker and add every stage. A no-op when an entry point has already done it.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, 'src')):
    _d = _os.path.dirname(_d)
_roots = [_d, _os.path.join(_d, 'omnigibson_runtime')]
_roots += [_f.path for _r in ('src', 'benchmark')
           for _f in _os.scandir(_os.path.join(_d, _r))
           if _f.is_dir() and not _f.name.startswith(('.', '_'))]
for _p in _roots:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import json
import os

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def name_to_text(category):
    """"wall_mounted_tv" -> "wall mounted tv"."""
    return category.replace("_", " ").strip()


def embed_names(names, model_name=DEFAULT_MODEL, batch_size=64):
    """Return an L2-normalized (len(names), dim) float32 embedding matrix."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    texts = [name_to_text(n) for n in names]
    vecs = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", default="data/vocab.json")
    parser.add_argument(
        "--field",
        default="merged_categories",
        choices=["merged_categories", "object_categories"],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="data/category_embeddings.npz")
    args = parser.parse_args()

    with open(args.vocab) as f:
        names = json.load(f)[args.field]

    vecs = embed_names(names, args.model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, names=np.array(names, dtype=object), embeddings=vecs, model=args.model)

    print(f"encoded {len(names)} categories with {args.model}")
    print(f"embedding dim: {vecs.shape[1]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
