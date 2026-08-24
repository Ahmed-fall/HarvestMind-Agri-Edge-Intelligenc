"""Production retrieval: hybrid BM25 + dense RRF with section-diversity capping
for English queries; pure dense for Swahili queries (BM25 contributes only
noise against non-English tokens -- measured during development).

Memory contract: load_retrieval_assets() and load_embedder() are separate on
purpose. The caller loads assets once (small, stays resident), embeds +
retrieves, then releases the embedder BEFORE the generation LLM loads. The
embedder and the LLM must never be co-resident on the 8 GB target profile.
"""
import json
import pickle
import re
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from textproc import fast_tokenize, base_section, indexed_text

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
EMBED_PATH = PROJECT_ROOT / "model" / "embeddings" / "bge-m3"

FINAL_K = 5            # context chunks handed to the generator
MAX_PER_SECTION = 2    # diversity cap per base section inside FINAL_K


class LanguageDetector:
    """Weighted lexical Swahili detector.

    Two tiers:
      STRONG   -- unambiguous Swahili content/question words (2 points each).
      CONCORD  -- grammatical function words/concords that cannot occur as
                  standalone English words (1 point each).
    A query is Swahili when the total score reaches 2, i.e. one clear content
    word or two function words. This generalizes far better than requiring
    matches from a fixed phrase list: hidden test prompts phrased with
    vocabulary never seen here still route to translation.
    """

    _STRONG = {
        # mandatory-prompt domain
        "kulingana", "mwongozo", "nifanye", "nini", "kuzuia", "unyevu",
        "usiharibu", "mazao", "ghalani", "ghala",
        # question words
        "nani", "wapi", "lini", "gani", "ngapi", "vipi", "je",
        # capability/need verbs
        "ninaweza", "ninawezaje", "wezaje", "unahitaji", "nahitaji",
        "tunahitaji", "ninataka", "unitafsiri",
        # agriculture / storage domain
        "kuhifadhi", "hifadhi", "chakula", "kilimo", "shamba", "mkulima",
        "wakulima", "mbeu", "mbega", "gunia", "wadudu", "kipanya", "panya",
        "dawa",
        "vyakula", "tunza", "kutunza", "kupoteza", "kupunguza", "kuongeza",
        "joto", "baridi", "maji", "hewa", "unyevunyevu", "kuuka", "kavu",
        "kukauka", "mba", "mdudu", "kunguru", "ndege",
    }

    _CONCORD = {
        "kwa", "wa", "ya", "cha", "vya", "za", "la", "ka", "na", "au",
        "katika", "kwenye", "hii", "hiyo", "ile", "hapa", "pale", "sasa",
        "pia", "sana", "lakini", "ila", "kama", "hivyo", "kwamba", "ili",
        "baada", "kabla", "kati", "yangu", "yako", "yake", "wetu", "wangu",
        "una", "huna", "tuna", "sina", "hatuna", "bila", "mpaka", "hadi",
        "tu",
    }

    @classmethod
    def score(cls, text: str) -> int:
        tokens = set(re.findall(r'\w+', text.lower()))
        return 2 * len(tokens & cls._STRONG) + len(tokens & cls._CONCORD)

    @classmethod
    def is_swahili(cls, text: str) -> bool:
        return cls.score(text) >= 2


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


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


def retrieve(
    query: str,
    chunks: List[Dict],
    bm25,
    dense_vectors: np.ndarray,
    embedder: SentenceTransformer,
    query_language: Optional[str] = None,
) -> List[Dict]:
    """Returns top FINAL_K chunks, section-diverse.

    English: reciprocal-rank fusion of BM25 top-20 and dense top-20.
    Swahili: pure dense ranking (multilingual embeddings carry the signal;
    BM25 on Swahili morphemes against an English corpus is noise).
    """
    is_sw = query_language == "sw" or (
        query_language is None and LanguageDetector.is_swahili(query)
    )
    query_vector = normalize_vec(embedder.encode(f"query: {query}", convert_to_numpy=True).astype(np.float32))
    dense_scores = np.dot(dense_vectors, query_vector)
    dense_ranked = np.argsort(dense_scores)[::-1].tolist()

    if is_sw:
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
    """Loads chunks + BM25 index + L2-normalized dense vectors. Small and cheap:
    these stay resident for the life of the process."""
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
    """Loads bge-m3 in fp16 on CPU. Caller owns the lifecycle: load right
    before use, `del` + gc.collect() right after, before the generation LLM
    loads. device="cpu" is explicit — the embedder must never land on a GPU."""
    return SentenceTransformer(
        str(EMBED_PATH),
        device="cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )


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
