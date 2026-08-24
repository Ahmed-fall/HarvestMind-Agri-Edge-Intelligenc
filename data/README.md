# Data Directory — Provenance

This directory contains the knowledge base and the retrieval indexes derived
from it. **All files here are committed** so that evaluation machines never
need to rebuild anything: inference is 100% offline.

## knowledge_base/

| File | What it is |
|---|---|
| `fao-1.pdf` | Source document: FAO manual *"Appropriate Seed and Grain Storage Systems for Small-scale Farmers"* (southern Africa edition). Official source: https://openknowledge.fao.org/server/api/core/bitstreams/a0b28a0c-0d9b-431f-9716-c9d78ee9ebfd/content |
| `fao_chunks.json` | **Shipped corpus.** 54 chunks produced by `src/chunker.py`, then contextually enriched by `src/contextual_retrieval.py`. Each chunk has `text` (clean body, what the LLM sees) and `indexed_text` (context prefix + body, what BM25/dense indexes see). |
| `fao_chunks_orig.json` | Pre-enrichment snapshot (chunker output only). Kept for provenance/diffing; not used at inference time. |

## vector_store/

Built once by `python src/indexer.py` from `fao_chunks.json`:

| File | What it is |
|---|---|
| `fao_bm25.pkl` | `rank_bm25.BM25Okapi` sparse index over `indexed_text`, tokenized with `src/textproc.py::fast_tokenize`. |
| `fao_vectors.npy` | float16 dense embedding matrix (54 × 1024, bge-m3), L2-normalized. |

## Rebuilding

```bash
# 1. place fao-1.pdf in this directory
python src/chunker.py
# 2. (optional but shipped) contextual enrichment — requires a GGUF LLM
python src/contextual_retrieval.py
# 3. rebuild both indexes — requires model/embeddings/bge-m3
python src/indexer.py
```

Judges do not need any of this: `download_model.sh` fetches the runtime models,
and these index files are already committed.
