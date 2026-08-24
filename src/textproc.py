"""
Shared, dependency-free text processing used by both the indexer (offline,
build-time) and the retriever (inference-time).

BM25 is exact-token matching: the query tokenizer and the index tokenizer
MUST agree on stemming or matches silently break. Having a single copy of
these functions removes that drift risk entirely.
"""
import re
from typing import List


def simple_stem(tok: str) -> str:
    """Crude suffix-stripping stemmer. Deliberately dependency-free.

    Without it, "container" (query) never matches "containers" (document)
    in BM25. Rules are intentionally conservative to avoid over-stemming
    domain terms (e.g. keeps "storage" intact via the 'ss' guard).
    """
    if len(tok) > 4 and tok.endswith('ies'):
        return tok[:-3] + 'y'
    if len(tok) > 3 and tok.endswith('es') and not tok.endswith('ses'):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss'):
        return tok[:-1]
    return tok


def fast_tokenize(text: str) -> List[str]:
    """Lowercased, stemmed word tokens via one regex pass."""
    return [simple_stem(w) for w in re.findall(r'\w+', text.lower())]


def base_section(section: str) -> str:
    """Collapses '(Part N)' chunk-split suffixes so section-diversity capping
    counts logical sections, not their split parts."""
    return re.sub(r'\s*\(Part \d+\)\s*$', '', section).strip()


def indexed_text(chunk: dict) -> str:
    """Retrieval text for a chunk: the contextually-enriched form when present,
    otherwise section heading + body."""
    return chunk.get("indexed_text") or f"{chunk['section']}. {chunk['text']}"
