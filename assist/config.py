import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PGDATA = ROOT / ".pgdata"
CORPUS_DIR = ROOT / "corpus"
EVAL_DATA = ROOT / "eval" / "data"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.environ.get("ASSIST_MODEL", "claude-sonnet-5-5")

# Tenants are fictional agencies; the corpus is synthetic (see README).
TENANTS = {
    "northgate": {"name": "Northgate Police Department", "code": "NGPD", "idx": 0},
    "lakeshore": {"name": "Lakeshore County Sheriff's Office", "code": "LCSO", "idx": 1},
    "cedarfalls": {"name": "Cedar Falls Police Department", "code": "CFPD", "idx": 2},
}

REFUSAL = "I don't know based on this agency's policy documents."


def refusal_threshold() -> float:
    p = ROOT / "threshold.json"
    return json.loads(p.read_text())["threshold"] if p.exists() else 0.0
