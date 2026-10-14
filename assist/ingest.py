"""python -m assist.ingest  — parse corpus/, quarantine suspicious chunks, embed, load."""
from .config import CORPUS_DIR, TENANTS
from .db import admin_conn, init_schema
from .embed import embed_passages
from .guard import sanitize
from .parse import parse_html


def main():
    init_schema()
    rows, removed = [], 0
    for tenant in TENANTS:
        for f in sorted((CORPUS_DIR / tenant).glob("*.html")):
            for c in parse_html(f):
                c["tenant_id"] = tenant
                c["id"] = f"{tenant}:{c['policy_number']}:{c['section_number']}:v{c['version']}"
                c["body"], n = sanitize(c["body"])
                removed += n
                rows.append(c)
    texts = [f"{r['policy_title']} - {r['section_title']}. {r['body']}" for r in rows]
    vecs = embed_passages(texts)
    with admin_conn() as conn:
        for r, v in zip(rows, vecs):
            conn.execute(
                """insert into chunks (id, tenant_id, policy_number, policy_title, section_number,
                   section_title, body, effective_date, superseded_date, version, status, embedding)
                   values (%(id)s, %(tenant_id)s, %(policy_number)s, %(policy_title)s, %(section_number)s,
                   %(section_title)s, %(body)s, %(effective_date)s, %(superseded_date)s, %(version)s,
                   %(status)s, %(emb)s::vector)""",
                {**r, "emb": "[" + ",".join(f"{x:.6f}" for x in v) + "]"},
            )
    q = removed
    s = sum(r["status"] == "superseded" for r in rows)
    print(f"ingested {len(rows)} chunks ({s} superseded, {q} injection-like sentences removed) for {len(TENANTS)} tenants")


if __name__ == "__main__":
    main()
