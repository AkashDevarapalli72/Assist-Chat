"""Section-aware HTML parsing: one chunk per policy section, with metadata."""
import re
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

_SEC = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*)$")


def parse_html(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    art = soup.find("article")
    a = art.attrs
    h1 = art.find("h1").get_text(" ", strip=True)
    title = re.sub(r"^Policy\s+\S+\s+[—-]\s+", "", h1)
    chunks = []
    for h2 in art.find_all("h2"):
        m = _SEC.match(h2.get_text(" ", strip=True))
        if not m:
            continue
        parts = []
        for sib in h2.find_next_siblings():
            if sib.name in ("h1", "h2"):
                break
            parts.append(sib.get_text(" ", strip=True))
        chunks.append({
            "policy_number": a["data-policy-number"],
            "policy_title": title,
            "section_number": m.group(1),
            "section_title": m.group(2),
            "body": " ".join(parts),
            "effective_date": date.fromisoformat(a["data-effective"]),
            "superseded_date": date.fromisoformat(a["data-superseded"]) if a.get("data-superseded") else None,
            "version": int(a["data-version"]),
            "status": a["data-status"],
        })
    return chunks
