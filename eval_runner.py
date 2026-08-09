"""
Runs eval_set.json against the live retrieval scenarios (A: dense-only,
B: BM25-only, C: hybrid RRF, D: production hybrid RRF + section diversity)
and prints Recall@5 per scenario.

This produces the real ablation numbers your experiment plan's Table 6B and
REPORT.md Section 4 need -- rather than eyeballing two mandatory prompts.

Run from your project root once fao_bm25.pkl / fao_vectors.npy exist:
    python eval_runner.py
"""
import json
import pickle
import re
import sys
import gc
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

import os

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR  # adjust if this file lives outside src/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from translator import ENG, SWH, load_translator, translate

KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
# CHANGED: override via env var so you can A/B the embedding model without editing
# the file, e.g.:  EMBED_MODEL_DIR=model/embeddings/bge-m3 python eval_runner.py
EMBED_PATH = Path(os.environ.get("EMBED_MODEL_DIR", str(PROJECT_ROOT / "model" / "embeddings" / "bge-m3")))
EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"

K = 5
VERBOSE = os.environ.get("VERBOSE", "0") == "1"  # per-item pass/fail detail
MAX_PER_SECTION = 2


def simple_stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith('ies'):
        return tok[:-3] + 'y'
    if len(tok) > 3 and tok.endswith('es') and not tok.endswith('ses'):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss'):
        return tok[:-1]
    return tok


def tokenize(text: str):
    return [simple_stem(w) for w in re.findall(r'\w+', text.lower())]


def indexed_text(chunk):
    return chunk.get("indexed_text") or f"{chunk['section']}. {chunk['text']}"


def base_section(section: str) -> str:
    return re.sub(r'\s*\(Part \d+\)\s*$', '', section).strip()


def select_diverse_top_k(ranked_indices, chunks, k, max_per_section):
    selected = []
    section_counts = {}
    for idx in ranked_indices:
        section = base_section(chunks[idx]["section"])
        if section_counts.get(section, 0) >= max_per_section:
            continue
        selected.append(idx)
        section_counts[section] = section_counts.get(section, 0) + 1
        if len(selected) == k:
            break
    return selected


def hit_at_k(ranked_ids, expected_ids, k):
    top_k = set(ranked_ids[:k])
    return 1 if top_k.intersection(expected_ids) else 0


def main():
    if not EVAL_SET_PATH.exists():
        print(f"[ERROR] {EVAL_SET_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    chunks = json.load(open(KB_DIR / "fao_chunks.json", encoding="utf-8"))
    bm25 = pickle.load(open(VECTOR_DIR / "fao_bm25.pkl", "rb"))
    eval_items = json.load(open(EVAL_SET_PATH, encoding="utf-8"))

    sw_items = [item for item in eval_items if item.get("lang") == "sw"]
    if sw_items:
        print(f"[INFO] Translating {len(sw_items)} Swahili eval queries to English...")
        translator_tok, translator_model = load_translator()
        for item in sw_items:
            item["retrieval_query"] = translate(
                item["query"],
                SWH,
                ENG,
                translator_tok,
                translator_model,
            )
        del translator_tok, translator_model
        gc.collect()

    default_embed_path = PROJECT_ROOT / "model" / "embeddings" / "bge-m3"
    using_default_embedder = (EMBED_PATH.resolve() == default_embed_path.resolve())

    print("[INFO] Loading embedding model...")
    embed_kwargs = {}
    if os.environ.get("EMBED_FP16", "0") == "1":
        # CHANGED: opt-in fp16 loading -- roughly halves the embedder's resident RAM.
        # Set EMBED_FP16=1 alongside EMBED_MODEL_DIR to test bge-m3's real footprint
        # before deciding whether it fits your 7GB RSS budget.
        import torch
        embed_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        print("[INFO] Loading embedder in fp16 (EMBED_FP16=1)")
    embedder = SentenceTransformer(str(EMBED_PATH), **embed_kwargs)

    if using_default_embedder and (VECTOR_DIR / "fao_vectors.npy").exists():
        # Reuse the cached index -- it matches this model's dimensionality.
        dense_vectors = np.load(VECTOR_DIR / "fao_vectors.npy").astype(np.float32)
    else:
        # CHANGED: a different embedding model (e.g. bge-m3) produces a different
        # vector space than fao_vectors.npy was built with -- reusing the cached
        # e5-small index here would silently compare wrong-dimension vectors or
        # crash. Re-encode the corpus in memory instead.
        print(f"[INFO] Embedding model differs from cached index ({EMBED_PATH.name}). Re-encoding corpus in memory...")
        passages = [f"passage: {indexed_text(c)}" for c in chunks]
        dense_vectors = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    dense_vectors = dense_vectors / norms

    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    results = {
        "A_dense": [],
        "B_bm25": [],
        "C_hybrid_rrf": [],
        "D_production": [],
    }
    per_item_log = []

    for item in eval_items:
        query = item.get("retrieval_query", item["query"])
        expected = set(item["expected_chunk_ids"])

        q_prefix = "query: " + query
        qvec = embedder.encode(q_prefix, convert_to_numpy=True).astype(np.float32)
        qn = np.linalg.norm(qvec)
        if qn > 0:
            qvec = qvec / qn

        # A: dense only
        dense_scores = np.dot(dense_vectors, qvec)
        dense_rank = np.argsort(dense_scores)[::-1]
        dense_ids = [chunks[i]["chunk_id"] for i in dense_rank]
        results["A_dense"].append(hit_at_k(dense_ids, expected, K))

        # B: BM25 only
        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_rank = np.argsort(bm25_scores)[::-1]
        bm25_ids = [chunks[i]["chunk_id"] for i in bm25_rank]
        results["B_bm25"].append(hit_at_k(bm25_ids, expected, K))

        # C: hybrid RRF (BM25 + dense, top20 each)
        rrf_k = 60
        rrf_scores = {}
        for rank, idx in enumerate(bm25_rank[:20]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_rank[:20]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        rrf_ids = [chunks[i]["chunk_id"] for i, _ in rrf_sorted]
        results["C_hybrid_rrf"].append(hit_at_k(rrf_ids, expected, K))

        # D: production retrieval selector, matching retriever.py's section diversity.
        production_indices = select_diverse_top_k(
            [i for i, _ in rrf_sorted],
            chunks,
            K,
            MAX_PER_SECTION,
        )
        production_ids = [chunks[i]["chunk_id"] for i in production_indices]
        d_hit = hit_at_k(production_ids, expected, K)
        results["D_production"].append(d_hit)

        per_item_log.append({
            "id": item["id"], "lang": item.get("lang", "en"), "difficulty": item["difficulty"],
            "A": results["A_dense"][-1], "B": results["B_bm25"][-1],
            "C": results["C_hybrid_rrf"][-1], "D": d_hit,
            "top5_D": production_ids[:5],
        })

    if VERBOSE:
        print("\nPer-item results (1=hit, 0=miss within top-{}):".format(K))
        print(f"{'id':10s} {'lang':4s} {'difficulty':16s}  A  B  C  D   top-5 (scenario D)")
        for r in per_item_log:
            print(f"{r['id']:10s} {r['lang']:4s} {r['difficulty']:16s}  {r['A']}  {r['B']}  {r['C']}  {r['D']}   {r['top5_D']}")

    print("\n" + "=" * 60)
    print(f"RECALL@{K} across {len(eval_items)} eval items")
    print("=" * 60)
    for scenario, hits in results.items():
        recall = sum(hits) / len(hits)
        print(f"  {scenario:20s}: {recall:.3f}  ({sum(hits)}/{len(hits)})")

    # Break out by difficulty tag so you can see whether failures cluster on
    # semantic-hard / cross-lingual items specifically, per plan Section 3B/3D.
    print("\nBreakdown by difficulty (Scenario D production):")
    by_diff = {}
    for item, hit in zip(eval_items, results["D_production"]):
        d = item["difficulty"]
        by_diff.setdefault(d, []).append(hit)
    for d, hits in by_diff.items():
        print(f"  {d:20s}: {sum(hits)}/{len(hits)}")

    print("\nBreakdown by language (Scenario D production):")
    by_lang = {}
    for item, hit in zip(eval_items, results["D_production"]):
        l = item.get("lang", "en")
        by_lang.setdefault(l, []).append(hit)
    for l, hits in by_lang.items():
        print(f"  {l:20s}: {sum(hits)}/{len(hits)}")


if __name__ == "__main__":
    main()
