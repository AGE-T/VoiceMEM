#!/usr/bin/env python3
"""scripts/measure_semantic_matrix.py — v0.9.0 Phase 1 evidence.

Measures REAL local multilingual-E5 cosine similarity for every semantic
case of the memory-semantics matrix (docs/SEMANTIC_MATRIX.md) on THIS
machine. The numbers are the design evidence for the trait contradiction
gate: "Do not use invented thresholds" — the merge threshold (0.95) and the
trait retrieval gate (0.88) are compared against the measured values below.

Pure E5 via transformers (mean pooling + L2 norm — identical maths to the
vendor's sentence-transformers path: LocalE5Embedder encodes
"passage: <text>" / "query: <text>" with normalize_embeddings=True).
The model is loaded from the operator-placed local directory when present
(models/embedding/multilingual-e5-small), else from the HF cache.

Read-only: touches no memory database. Writes JSON to stdout (and a
--json path when given) for the semantic-matrix doc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_DIR = REPO / "models" / "embedding" / "multilingual-e5-small"


def _e5_dir() -> str:
    if any(LOCAL_DIR.glob("*.safetensors")) or (LOCAL_DIR / "model.onnx").is_file():
        return str(LOCAL_DIR)
    return "intfloat/multilingual-e5-small"


def load_e5():
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = _e5_dir()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return tok, model


def embed_passages(texts: list[str]):
    """E5 passage embeddings (mean pooling + L2 norm) — same maths as the
    vendor's LocalE5Embedder.embed_texts (sentence-transformers under the
    hood, which itself is exactly this pooling)."""
    import numpy as np
    import torch

    tok, model = load_e5()
    out = []
    with torch.no_grad():
        batch = [f"passage: {t}" for t in texts]
        enc = tok(batch, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        hidden = model(**enc).last_hidden_state                # (B, L, D)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out = pooled.cpu().numpy().astype(np.float32)
    return out


#: The Phase 1 semantic matrix. Each case: existing claim → incoming claim.
#: Labels document the expected semantic category; the measured cosine is
#: the evidence the design decision is checked against.
CASES = [
    # A. REINFORCEMENT (duplicate)
    ("A-reinforce-en", "Thomas likes motorcycles", "Thomas likes motorcycles"),
    ("A-reinforce-hu", "Thomas szereti a motorokat", "Thomas szereti a motorokat"),
    # B. RESTATEMENT / PARAPHRASE
    ("B-paraphrase-en", "Thomas likes motorcycles", "Thomas enjoys riding motorcycles"),
    ("B-paraphrase-hu", "Thomas szereti a motorokat", "Thomas imád motorozni"),
    ("B-cross-paraphrase", "Thomas likes motorcycles", "Thomas szereti a motorokat"),
    # C. CONTRADICTION (negation)
    ("C-negation-en", "Thomas likes motorcycles", "Thomas no longer likes motorcycles"),
    ("C-negation-en2", "Thomas likes motorcycles", "Thomas does not like motorcycles"),
    ("C-negation-hu", "Thomas szereti a motorokat", "Thomas már nem szereti a motorokat"),
    ("C-negation-hu2", "Thomas szereti a motorokat", "Thomas nem szereti a motorokat"),
    ("C-negation-xlang", "Thomas szereti a motorokat", "Thomas no longer likes motorcycles"),
    ("C-negation-xlang2", "Thomas likes motorcycles", "Thomas már nem szereti a motorokat"),
    # C'. ANTIPATHY (opposite affect, no negation word)
    ("C-antipathy-en", "Thomas likes motorcycles", "Thomas hates motorcycles"),
    ("C-antipathy-hu", "Thomas szereti a motorokat", "Thomas utálja a motorokat"),
    # D. EXPLICIT REPLACEMENT
    ("D-replace", "Thomas lives in Budapest", "Thomas moved to Vienna"),
    ("D-replace-trait", "likes coffee", "gave up coffee"),
    ("D-replace-trait2", "likes coffee", "stopped liking coffee"),
    # flip-back (the negative state exists, a positive returns)
    ("flip-back-en", "no longer likes motorcycles", "likes motorcycles again"),
    ("flip-back-plain", "no longer likes motorcycles", "likes motorcycles"),
    ("flip-back-hu", "már nem szereti a motorokat", "megint szereti a motorokat"),
    # noise floor: related-but-different topic (must NOT be a supersede match)
    ("noise-different-topic", "likes motorcycles", "no longer likes bicycles"),
    ("noise-different-topic2", "likes coffee", "gave up tea"),
    # E. TEMPORAL COEXISTENCE
    ("E-temporal-en", "Thomas owns a Renault", "Thomas previously owned a BMW"),
    ("E-temporal-hu", "Thomasnak Renault-ja van", "Thomasnak korábban BMW-je volt"),
    ("E-temporal-past-hu", "Thomas szereti a motorokat", "Thomas régen szerette a motorokat"),
    ("E-temporal-past-en", "Thomas likes motorcycles", "Thomas used to like motorcycles"),
    # F. QUALIFIED CONTRADICTION
    ("F-qualified-en", "Thomas likes motorcycles", "Thomas likes motorcycles except during winter"),
    ("F-qualified-hu", "Thomas szereti a motorokat", "Thomas szereti a motorokat, kivéve télen"),
    # G. UNCERTAIN CONTRADICTION
    ("G-uncertain-en", "Thomas likes motorcycles", "Thomas might no longer like motorcycles"),
    ("G-uncertain-hu", "Thomas szereti a motorokat", "Thomas talán már nem szereti a motorokat"),
    # Short trait-claim style (the labels the extractor actually writes)
    ("claim-pos-neg", "likes motorcycles", "no longer likes motorcycles"),
    ("claim-pos-neg2", "likes motorcycles", "does not like motorcycles"),
    ("claim-pos-antipathy", "likes motorcycles", "hates motorcycles"),
    ("claim-hu-pos-neg", "szereti a motorokat", "már nem szereti a motorokat"),
    # Query-side: how a current-state question sees the stored claims
    ("query-lives", "Where does Thomas live?", "Thomas lives in Vienna"),
    ("query-likes", "does Thomas like motorcycles?", "likes motorcycles"),
    ("query-likes-neg", "does Thomas like motorcycles?", "no longer likes motorcycles"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the results JSON here")
    args = ap.parse_args()

    texts = []
    for _, a, b in CASES:
        texts.extend((a, b))
    vecs = embed_passages(texts)

    results = []
    for i, (label, a, b) in enumerate(CASES):
        va, vb = vecs[2 * i], vecs[2 * i + 1]
        sim = float(va @ vb)
        results.append({"case": label, "existing": a, "incoming": b,
                        "cosine": round(sim, 4),
                        "merge_threshold_095": sim >= 0.95,
                        "retrieval_gate_088": sim >= 0.88})

    width = max(len(r["case"]) for r in results)
    print(f"{'case'.ljust(width)}  cosine   >=0.95(merge)  >=0.88(retrieval)")
    for r in results:
        print(f"{r['case'].ljust(width)}  {r['cosine']:.4f}   "
              f"{'YES' if r['merge_threshold_095'] else 'no ':<12}   "
              f"{'YES' if r['retrieval_gate_088'] else 'no '}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
