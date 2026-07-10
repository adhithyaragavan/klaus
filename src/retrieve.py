from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.chunker import Clause
from src.index import IndexHandle, get_embedding_model

DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4


@dataclass
class RetrievedClause:
    clause: Clause
    score: float


def retrieve(query: str, index: IndexHandle, top_k: int = 6) -> list[RetrievedClause]:
    n = len(index.clauses)
    if n == 0:
        return []

    model = get_embedding_model()
    query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    dense_scores, dense_idxs = index.dense_index.search(query_vec, n)
    dense_by_idx = np.zeros(n, dtype="float32")
    for score, idx in zip(dense_scores[0], dense_idxs[0]):
        dense_by_idx[idx] = score

    bm25_scores = np.asarray(index.bm25_index.get_scores(query.lower().split()), dtype="float32")

    combined = DENSE_WEIGHT * _min_max_normalize(dense_by_idx) + BM25_WEIGHT * _min_max_normalize(bm25_scores)

    ranked_idxs = np.argsort(-combined)[:top_k]
    return [RetrievedClause(clause=index.clauses[i], score=float(combined[i])) for i in ranked_idxs]


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
