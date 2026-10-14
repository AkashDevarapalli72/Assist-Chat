"""Signed tenant tokens. The tenant comes ONLY from the token, never from the request body."""
import hashlib
import hmac
import os

from .config import TENANTS

_SECRET = os.environ.get("ASSIST_SECRET", "dev-secret-change-me").encode()


def _sig(tenant: str) -> str:
    return hmac.new(_SECRET, tenant.encode(), hashlib.sha256).hexdigest()[:32]


def issue_token(tenant: str) -> str:
    if tenant not in TENANTS:
        raise ValueError("unknown tenant")
    return f"{tenant}.{_sig(tenant)}"


def verify_token(token: str) -> str:
    tenant, _, sig = (token or "").partition(".")
    if tenant not in TENANTS or not hmac.compare_digest(sig, _sig(tenant)):
        raise PermissionError("invalid token")
    return tenant
