"""Offline index builder. Run once after placing the FAO PDF and running
chunker.py (+ optionally contextual_retrieval.py):

    python src/indexer.py

Produces data/vector_store/fao_bm25.pkl and fao_vectors.npy, which are
committed to the repo so judges never need to rebuild them.
"""
import json
import logging
import gc
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from textproc import fast_tokenize, indexed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HarvestMindIndexer")


@dataclass
class IndexerConfig:
    """Configuration schema for vector and sparse index generation."""
    kb_dir: Path
    vector_dir: Path
    model_path: Path
    chunk_filename: str = "fao_chunks.json"
    bm25_filename: str = "fao_bm25.pkl"
    vector_filename: str = "fao_vectors.npy"
    batch_size: int = 32


class HybridIndexBuilder:
    """Orchestrates the creation of sparse and dense indexes with strict memory controls."""

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.config.vector_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = self._load_chunks()

    def _load_chunks(self) -> List[Dict[str, Any]]:
        chunk_path = self.config.kb_dir / self.config.chunk_filename
        if not chunk_path.exists():
            logger.error(f"Required chunk payload not found at {chunk_path}.")
            sys.exit(1)

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            logger.error("Chunk payload is empty. Halting index generation.")
            sys.exit(1)

        logger.info(f"Successfully loaded {len(chunks)} chunks into memory.")
        return chunks

    def build_sparse_index(self) -> None:
        """Generates and serializes the BM25 Okapi index over section heading + text."""
        logger.info("Initializing BM25 tokenization sequence.")
        tokenized_corpus = [
            fast_tokenize(indexed_text(chunk)) for chunk in self.chunks
        ]

        bm25_index = BM25Okapi(tokenized_corpus)
        out_path = self.config.vector_dir / self.config.bm25_filename

        with open(out_path, "wb") as f:
            pickle.dump(bm25_index, f)

        logger.info(f"BM25 index successfully serialized to {out_path}.")

    def build_dense_index(self) -> None:
        """Generates L2-normalized float16 dense embeddings using a batched,
        memory-safe approach."""
        if not self.config.model_path.exists():
            logger.error(f"Embedding model directory missing: {self.config.model_path}")
            sys.exit(1)

        logger.info("Loading SentenceTransformer model (fp16, offline).")
        import torch
        embedder = SentenceTransformer(str(self.config.model_path), model_kwargs={"torch_dtype": torch.float16})

        passages = [f"passage: {indexed_text(chunk)}" for chunk in self.chunks]

        logger.info(f"Encoding {len(passages)} passages. Batch limit: {self.config.batch_size}.")
        embeddings = embedder.encode(
            passages,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # L2-normalize before storage: retriever scores via raw np.dot(), which is
        # only cosine similarity when both operands are unit vectors.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embeddings = embeddings / norms

        # Downcast to float16 to halve the inference-phase memory footprint.
        embeddings_fp16 = embeddings.astype(np.float16)
        out_path = self.config.vector_dir / self.config.vector_filename

        np.save(out_path, embeddings_fp16)
        logger.info(f"Dense vector matrix saved to {out_path}. Dimensions: {embeddings_fp16.shape}.")

        del embedder
        del embeddings
        gc.collect()
        logger.info("Embedding architecture purged from RAM.")


def main():
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent

    config = IndexerConfig(
        kb_dir=project_root / "data" / "knowledge_base",
        vector_dir=project_root / "data" / "vector_store",
        model_path=project_root / "model" / "embeddings" / "bge-m3"
    )

    builder = HybridIndexBuilder(config)
    builder.build_sparse_index()
    builder.build_dense_index()


if __name__ == "__main__":
    main()
