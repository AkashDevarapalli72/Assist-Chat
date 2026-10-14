"""uvicorn assist.api:app — tenant comes from the bearer token only; any tenant field in the body is ignored."""
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .answer import answer
from .auth import issue_token, verify_token
from .config import TENANTS

app = FastAPI(title="Assist Chat")


class Ask(BaseModel):
    question: str
    as_of: date | None = None  # extra fields (e.g. "tenant") are ignored by design


def current_tenant(authorization: str = Header(default="")) -> str:
    try:
        return verify_token(authorization.removeprefix("Bearer ").strip())
    except PermissionError:
        raise HTTPException(401, "invalid or missing tenant token")


@app.get("/api/tenants")
def tenants():
    return [{"id": k, "name": v["name"]} for k, v in TENANTS.items()]


@app.post("/api/dev-token/{tenant}")
def dev_token(tenant: str):
    """Demo only: a real deployment issues tokens from the agency's identity provider."""
    if tenant not in TENANTS:
        raise HTTPException(404, "unknown tenant")
    return {"token": issue_token(tenant)}


@app.post("/api/ask")
def ask(body: Ask, tenant: str = Depends(current_tenant)):
    return answer(tenant, body.question, as_of=body.as_of)


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")
