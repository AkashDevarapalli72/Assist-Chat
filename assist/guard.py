"""Heuristic prompt-injection screen applied at ingestion (quarantine) and on sources."""
import re

_PATTERNS = [
    r"ignore (all |any |your |the )?(previous|prior|above|earlier)?\s*(instructions|rules|directions)",
    r"disregard (all |any |your |the )?(previous|prior|above|earlier)?\s*(instructions|rules|guidelines|limitations)",
    r"\bsystem\s*(:|override|prompt)",
    r"instruction(s)? to (the )?(ai|assistant|model|llm)",
    r"you are now\b",
    r"(reveal|show|list|print|output) (the )?(documents|manuals|policies) (belonging )?(to|of|from) (other|all) (agenc|tenant)",
    r"override (the )?tenant",
    # added after the first eval run showed paraphrased injections slipping through (dev-set tuning, see README)
    r"\b(automated readers?|summari[sz]er|language model|chatbot)\b",
    r"(code word|prefaced with|(end|close) (each|every) (reply|answer)|every answer must)",
    r"(neighbou?ring|other) (agenc|department)",
    r"(set aside|lifted|from this point forward).{0,80}(limitation|boundary|answer|rule)",
    r"\bno policy exists\b",
]
_RE = re.compile("|".join(_PATTERNS), re.I)


def is_suspicious(text: str) -> bool:
    return bool(_RE.search(text))


_SENT = re.compile(r"(?<=[.!?])\s+")


def sanitize(text: str) -> tuple[str, int]:
    """Drop instruction-like sentences, keep the rest of the section. Returns (clean_text, n_removed)."""
    keep = [s for s in _SENT.split(text) if not is_suspicious(s)]
    return " ".join(keep), len(_SENT.split(text)) - len(keep)
