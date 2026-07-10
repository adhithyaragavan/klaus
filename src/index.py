from __future__ import annotations

from dataclasses import dataclass

from src.chunker import Clause

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = None  # lazy-loaded singleton; sentence-transformers model load is expensive


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


@dataclass
class IndexHandle:
    clauses: list[Clause]
    dense_index: object  # FAISS flat index over sentence-transformers embeddings, aligned to `clauses` by position
    bm25_index: object  # rank_bm25 index over the same clause text, aligned to `clauses` by position


def build_index(clauses: list[Clause]) -> IndexHandle:
    if not clauses:
        raise ValueError("build_index requires at least one clause")

    import faiss
    from rank_bm25 import BM25Okapi

    texts = [clause.text for clause in clauses]

    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    dense_index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product over normalized vectors == cosine similarity
    dense_index.add(embeddings)

    bm25_index = BM25Okapi([text.lower().split() for text in texts])

    return IndexHandle(clauses=clauses, dense_index=dense_index, bm25_index=bm25_index)
