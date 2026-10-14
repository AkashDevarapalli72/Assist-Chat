from functools import lru_cache

from .config import EMBED_MODEL, RERANK_MODEL


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _embedder().passage_embed(texts)]


def embed_query(text: str) -> list[float]:
    return next(iter(_embedder().query_embed([text]))).tolist()


def rerank(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    return [float(s) for s in _reranker().rerank(query, docs)]
