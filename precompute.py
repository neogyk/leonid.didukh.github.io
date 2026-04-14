"""
Pre-compute embeddings and t-SNE coordinates for all visualization modes.

Run once (or whenever traj.json changes):
    pip install sentence-transformers scikit-learn
    python visualization/precompute.py

Writes: visualization/agents/coords.json
The browser loads this file and skips the in-browser model download entirely.
"""

import json
import pathlib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

TRAJ_PATH  = pathlib.Path(__file__).parent / "agents" / "traj.json"
COORDS_PATH = pathlib.Path(__file__).parent / "agents" / "coords.json"
MODEL_NAME = "all-MiniLM-L6-v2"   # same model used by the browser fallback
TSNE_SEED  = 42                    # fixed seed → stable layout across sessions

def embed(model, texts):
    print(f"  Embedding {len(texts)} texts...")
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

def tsne3d(vectors):
    n = len(vectors)
    perplexity = min(5, n - 1)     # sklearn requires perplexity < n_samples
    print(f"  Running t-SNE (n={n}, perplexity={perplexity})...")
    coords = TSNE(
        n_components=3,
        perplexity=perplexity,
        n_iter=1000,
        random_state=TSNE_SEED,
    ).fit_transform(vectors)
    # Normalise to roughly the same scale the browser used (embeddingScale=5)
    scale = 5.0 / (coords.std() + 1e-8)
    return (coords * scale).tolist()

def main():
    print(f"Loading data from {TRAJ_PATH}")
    steps = json.loads(TRAJ_PATH.read_text())
    print(f"  {len(steps)} steps found\n")

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print()

    coords = {}

    # --- triplets (one point per step: combined p+s+c) ---
    print("Mode: triplets")
    texts = [f"{s['p']}\n{s['s']}\n{s['c']}" for s in steps]
    coords["triplets"] = tsne3d(embed(model, texts))
    print()

    # --- components (three points per step: p, s, c separately) ---
    print("Mode: components")
    texts = []
    for s in steps:
        texts += [s["p"], s["s"], s["c"]]
    coords["components"] = tsne3d(embed(model, texts))
    print()

    # --- problems ---
    print("Mode: problems")
    coords["problems"] = tsne3d(embed(model, [s["p"] for s in steps]))
    print()

    # --- methods ---
    print("Mode: methods")
    coords["methods"] = tsne3d(embed(model, [s["s"] for s in steps]))
    print()

    # --- conclusions ---
    print("Mode: conclusions")
    coords["conclusions"] = tsne3d(embed(model, [s["c"] for s in steps]))
    print()

    COORDS_PATH.write_text(json.dumps(coords, separators=(",", ":")))
    print(f"Saved → {COORDS_PATH}")
    print("Reload the visualization — the browser will now skip model download.")

if __name__ == "__main__":
    main()
