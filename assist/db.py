"""Embedded Postgres (pgserver) + pgvector, with row-level security.

The app connects as `assist_app`, a non-superuser role with SELECT only. Every
request sets `app.tenant` inside its transaction; the RLS policy compares each
row's tenant_id with that setting. With no setting, no rows are visible.
"""
from contextlib import contextmanager

import pgserver
import psycopg
from psycopg.rows import dict_row

from .config import EMBED_DIM, PGDATA

_srv = None


def _server():
    global _srv
    if _srv is None:
        _srv = pgserver.get_server(PGDATA, cleanup_mode=None)
    return _srv


def admin_uri() -> str:
    return _server().get_uri()


def app_uri() -> str:
    return admin_uri().replace("//postgres:@", "//assist_app@")


SCHEMA = f"""
create extension if not exists vector;
do $$ begin
  if not exists (select from pg_roles where rolname = 'assist_app') then
    create role assist_app login;
  end if;
end $$;

drop table if exists chunks;
create table chunks (
  id text primary key,
  tenant_id text not null,
  policy_number text not null,
  policy_title text not null,
  section_number text not null,
  section_title text not null,
  body text not null,
  effective_date date not null,
  superseded_date date,
  version int not null,
  status text not null check (status in ('active','superseded','quarantined')),
  embedding vector({EMBED_DIM}),
  tsv tsvector generated always as (
    to_tsvector('english', policy_title || ' ' || section_title || ' ' || body)
  ) stored
);
create index chunks_tenant_idx on chunks (tenant_id);
create index chunks_tsv_idx on chunks using gin (tsv);
create index chunks_emb_idx on chunks using hnsw (embedding vector_cosine_ops);

alter table chunks enable row level security;
alter table chunks force row level security;
create policy tenant_isolation on chunks for select to assist_app
  using (tenant_id = current_setting('app.tenant', true));
grant usage on schema public to assist_app;
grant select on chunks to assist_app;
"""


def init_schema():
    with psycopg.connect(admin_uri(), autocommit=True) as c:
        c.execute(SCHEMA)


def admin_conn():
    return psycopg.connect(admin_uri(), autocommit=True)


@contextmanager
def tenant_conn(tenant: str | None):
    """Connection as the restricted role, scoped to one tenant for this transaction."""
    with psycopg.connect(app_uri(), row_factory=dict_row) as conn:
        if tenant is not None:
            conn.execute("select set_config('app.tenant', %s, true)", (tenant,))
        yield conn
