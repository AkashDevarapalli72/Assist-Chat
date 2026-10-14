"""Hybrid retrieval: Postgres FTS + pgvector, fused with RRF, reranked by a cross-encoder."""
import re
from datetime import date

from .db import tenant_conn
from .embed import embed_query, rerank

_STOP = set("the a an of to in on for and or is are was were be what when who how which does do can may must "
            "should shall that this with by at from as it its my our me i we they their there than then if "
            "any all about after before during under over per".split())
_YEAR = re.compile(r"\b(20[0-3]\d|19\d\d)\b")
COLS = ("id, policy_number, policy_title, section_number, section_title, body, "
        "effective_date::text as effective_date, version, status")


def detect_as_of(query: str) -> date | None:
    """A year in the question ("as of 2022") means: answer from the revision in force then."""
    m = _YEAR.search(query)
    return date(int(m.group(1)), 12, 31) if m else None


def _tsquery(q: str) -> str | None:
    toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 2 and t not in _STOP]
    return " | ".join(dict.fromkeys(toks)) or None


def search(tenant: str, query: str, as_of: date | None = None, mode: str = "hybrid",
           rerank_top: int = 20, top: int = 8, k_each: int = 30) -> list[dict]:
    """mode: 'hybrid' | 'vector' | 'fts'. Reranks (and sets rerank_score) except mode 'hybrid-norerank'."""
    as_of = as_of or detect_as_of(query)
    if as_of is None:
        scope, p = "status = 'active'", {}
    else:
        scope = ("status in ('active','superseded') and effective_date <= %(asof)s "
                 "and (superseded_date is null or superseded_date > %(asof)s)")
        p = {"asof": as_of}
    qvec = "[" + ",".join(f"{x:.6f}" for x in embed_query(query)) + "]"
    tsq = _tsquery(query)
    vec, fts = [], []
    with tenant_conn(tenant) as conn:
        if mode in ("hybrid", "vector", "hybrid-norerank"):
            vec = conn.execute(f"select {COLS} from chunks where {scope} "
                               f"order by embedding <=> %(q)s::vector limit {k_each}", {**p, "q": qvec}).fetchall()
        if tsq and mode in ("hybrid", "fts", "hybrid-norerank"):
            fts = conn.execute(f"select {COLS} from chunks where {scope} and tsv @@ to_tsquery('english', %(t)s) "
                               f"order by ts_rank_cd(tsv, to_tsquery('english', %(t)s)) desc limit {k_each}",
                               {**p, "t": tsq}).fetchall()
    fused: dict[str, dict] = {}
    for rows in (vec, fts):
        for rank, r in enumerate(rows):
            e = fused.setdefault(r["id"], {**r, "rrf": 0.0})
            e["rrf"] += 1.0 / (60 + rank)
    cands = sorted(fused.values(), key=lambda r: -r["rrf"])[:rerank_top]
    if mode == "hybrid-norerank":
        for r in cands:
            r["rerank_score"] = r["rrf"]
        return cands[:top]
    scores = rerank(query, [f"{r['policy_title']} - {r['section_title']}. {r['body']}" for r in cands])
    for r, s in zip(cands, scores):
        r["rerank_score"] = s
    return sorted(cands, key=lambda r: -r["rerank_score"])[:top]
