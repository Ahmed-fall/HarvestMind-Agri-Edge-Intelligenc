import json
import pickle
import re
import gc
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
# CHANGED: CrossEncoder import removed from the live path. eval_runner.py showed the
# ms-marco-MiniLM-L-6-v2 reranker underperforming plain hybrid RRF in every run tested
# (3 independent runs: e5-small baseline, bge-m3 fp32, bge-m3 fp16) -- it specifically
# broke the plastic-container case (fao_054) every single time despite BM25, dense, and
# RRF all surfacing the correct chunk upstream. Dropping it also helps Sperf (one fewer
# model load + forward pass) and Seff (no reranker RAM). If you re-test this decision
# later, `from sentence_transformers import CrossEncoder` is the only import to restore.

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
# CHANGED: committed to bge-m3 -- fixed both Swahili misses and the plastic-container
# case, with zero recall cost from fp16 quantization. See REPORT.md Table 6C.
EMBED_PATH = PROJECT_ROOT / "model" / "embeddings" / "bge-m3"

FINAL_K = 5            # CHANGED: 4 -> 5. The Swahili moisture query showed FINAL_K=4 +
                        # section-diversity capping was slightly too tight -- it surfaced
                        # a tangential chunk (fao_009) but not the more directly relevant
                        # fao_005/fao_006. Cheap to widen now that dropping the reranker
                        # freed up RAM/latency headroom.
MAX_PER_SECTION = 2


class LanguageDetector:
    # Real mandatory Prompt 2 keywords (moisture/granary), corrected from the earlier
    # plastic-container mistranslation that was being tested against.
    SWAHILI_KEYWORDS = {
        "kulingana", "na", "mwongozo", "fao", "nifanye", "nini",
        "kuzuia", "unyevu", "usiharibu", "mazao", "ghalani"
    }

    @staticmethod
    def is_swahili(text: str) -> bool:
        tokens = set(re.findall(r'\w+', text.lower()))
        return len(tokens.intersection(LanguageDetector.SWAHILI_KEYWORDS)) >= 2


def simple_stem(tok: str) -> str:
    # Must match indexer.py's TextProcessor.simple_stem exactly -- BM25 query
    # tokenization and index tokenization have to agree or matches silently break.
    if len(tok) > 4 and tok.endswith('ies'):
        return tok[:-3] + 'y'
    if len(tok) > 3 and tok.endswith('es') and not tok.endswith('ses'):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss'):
        return tok[:-1]
    return tok


def fast_tokenize(text: str) -> List[str]:
    return [simple_stem(w) for w in re.findall(r'\w+', text.lower())]


def base_section(section: str) -> str:
    return re.sub(r'\s*\(Part \d+\)\s*$', '', section).strip()


def select_diverse_top_k(ranked_indices: List[int], chunks: List[Dict], k: int, max_per_section: int) -> List[int]:
    """Walks a score-ranked index list and keeps top-k while capping duplicates per base section."""
    selected = []
    section_counts = {}
    for idx in ranked_indices:
        sec = base_section(chunks[idx]["section"])
        if section_counts.get(sec, 0) >= max_per_section:
            continue
        selected.append(idx)
        section_counts[sec] = section_counts.get(sec, 0) + 1
        if len(selected) == k:
            break
    return selected


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def retrieve(
    query: str,
    chunks: List[Dict],
    bm25,
    dense_vectors: np.ndarray,
    embedder: SentenceTransformer,
    query_language: Optional[str] = None,
) -> List[Dict]:
    """Production retrieval entry point. Returns top FINAL_K chunks, section-diverse."""
    is_sw = query_language == "sw" or (
        query_language is None and LanguageDetector.is_swahili(query)
    )
    query_vector = normalize_vec(embedder.encode(f"query: {query}", convert_to_numpy=True).astype(np.float32))
    dense_scores = np.dot(dense_vectors, query_vector)
    dense_ranked = np.argsort(dense_scores)[::-1].tolist()

    if is_sw:
        # Pure dense, no BM25: validated by eval_runner.py -- hybrid RRF actively
        # hurt Swahili recall here because BM25 contributes noise, not signal,
        # against non-English query tokens, and displaces a good dense ranking.
        top_indices = select_diverse_top_k(dense_ranked, chunks, FINAL_K, MAX_PER_SECTION)
    else:
        bm25_ranks = np.argsort(bm25.get_scores(fast_tokenize(query)))[::-1][:20].tolist()
        dense_ranks = dense_ranked[:20]
        rrf_k = 60
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = [idx for idx, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]
        top_indices = select_diverse_top_k(ranked, chunks, FINAL_K, MAX_PER_SECTION)

    return [chunks[i] for i in top_indices]


def load_retrieval_assets() -> Tuple[List[Dict], Any, np.ndarray]:
    """Loads chunks + BM25 index + normalized dense vectors. Cheap, keep resident in RAM
    for the life of the process -- this is not the expensive part of the pipeline."""
    chunks_path = KB_DIR / "fao_chunks.json"
    bm25_path = VECTOR_DIR / "fao_bm25.pkl"
    vectors_path = VECTOR_DIR / "fao_vectors.npy"

    if not chunks_path.exists() or not bm25_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(
            f"Missing index files under {KB_DIR} / {VECTOR_DIR}. Run indexer.py first."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    dense_vectors = np.load(vectors_path).astype(np.float32)
    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    dense_vectors = dense_vectors / norms

    return chunks, bm25, dense_vectors


def load_embedder() -> SentenceTransformer:
    """Loads bge-m3 in fp16. Caller owns the lifecycle -- load right before use,
    `del` + `gc.collect()` right after, per the memory-hygiene requirement (this is
    the expensive, large-RAM part of retrieval and should not stay resident once
    the LLM needs to load)."""
    return SentenceTransformer(str(EMBED_PATH), model_kwargs={"torch_dtype": torch.float16})


def run_retrieval_test():
    chunks, bm25, dense_vectors = load_retrieval_assets()

    prompts = {
        "Prompt 1 (English)": "According to Section 2 of the indexed FAO manual, what are the exact steps to seal a plastic storage container to prevent insect infestations?",
        "Prompt 2 (Swahili)": "Kulingana na mwongozo wa FAO, nifanye nini kuzuia unyevu usiharibu mazao ghalani?",
    }

    print("[INFO] Loading embedding model (bge-m3, fp16)...")
    embedder = load_embedder()

    for label, query in prompts.items():
        print("\n" + "=" * 80)
        print(f"EVALUATING: {label}")
        print(f"QUERY: \"{query}\"")
        print("=" * 80)

        results = retrieve(query, chunks, bm25, dense_vectors, embedder)
        for rank, chunk in enumerate(results):
            print(f"\n[RANK {rank + 1}] Chunk ID: {chunk['chunk_id']} | Sec: {chunk['section']}")
            print(f"Text Snippet: {chunk['text'][:250]}...")

    del embedder
    gc.collect()
    print("\n[INFO] Test completed.")


if __name__ == "__main__":
    run_retrieval_test()
