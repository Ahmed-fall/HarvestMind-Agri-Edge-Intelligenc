import json
import pickle
import logging
import re
import gc
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Configure strict, enterprise-standard logging
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

class TextProcessor:
    """Isolated text processing utilities to prevent dependency bloat."""

    @staticmethod
    def simple_stem(tok: str) -> str:
        # CHANGED: crude suffix-stripping, no NLTK/Spacy dependency (matches this
        # class's own stated design goal). Verified against real query traffic:
        # without this, "container" (query) vs "containers" (document) never match
        # in BM25 since it's exact-token matching with no stemming at all. This single
        # fix moved the correct chunk for the plastic-container test query from BM25
        # rank 9 -> rank 5 (measured directly against fao_chunks.json).
        if len(tok) > 4 and tok.endswith('ies'):
            return tok[:-3] + 'y'
        if len(tok) > 3 and tok.endswith('es') and not tok.endswith('ses'):
            return tok[:-2]
        if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss'):
            return tok[:-1]
        return tok

    @staticmethod
    def fast_tokenize(text: str) -> List[str]:
        # Extracts alphanumeric tokens; avoids NLTK/Spacy memory overhead
        return [TextProcessor.simple_stem(w) for w in re.findall(r'\w+', text.lower())]

    @staticmethod
    def indexed_text(chunk: Dict[str, Any]) -> str:
        """Returns retrieval text, using contextual enrichment when available."""
        return chunk.get("indexed_text") or f"{chunk['section']}. {chunk['text']}"

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
        """Generates and serializes the BM25 Okapi index."""
        logger.info("Initializing BM25 tokenization sequence.")
        # CHANGED: tokenize section heading + text together, same as dense passages below,
        # so BM25 also benefits from heading vocabulary (e.g. "Small containers").
        tokenized_corpus = [
            TextProcessor.fast_tokenize(TextProcessor.indexed_text(chunk))
            for chunk in self.chunks
        ]

        bm25_index = BM25Okapi(tokenized_corpus)
        out_path = self.config.vector_dir / self.config.bm25_filename

        with open(out_path, "wb") as f:
            pickle.dump(bm25_index, f)

        logger.info(f"BM25 index successfully serialized to {out_path}.")

    def build_dense_index(self) -> None:
        """Generates float16 dense embeddings using a batched, memory-safe approach."""
        if not self.config.model_path.exists():
            logger.error(f"Embedding model directory missing: {self.config.model_path}")
            sys.exit(1)

        logger.info("Loading SentenceTransformer model (Offline Mode).")
        # CHANGED: fp16 load -- measured 1.51GB RSS vs fp32's much larger footprint,
        # with zero recall difference across the full eval set. See eval_runner.py runs.
        import torch
        embedder = SentenceTransformer(str(self.config.model_path), model_kwargs={"torch_dtype": torch.float16})

        # CHANGED: prefix each passage with its section heading before the "passage: " marker
        # required by e5-family models. This is the single highest-leverage fix for the
        # "Small containers" / "plastic storage container" retrieval miss — the heading
        # text ("Small containers", "Hermetic bags", etc.) now becomes part of what gets
        # embedded, not just something attached after the fact in metadata.
        passages = [
            f"passage: {TextProcessor.indexed_text(chunk)}" for chunk in self.chunks
        ]

        logger.info(f"Encoding {len(passages)} passages. Batch limit: {self.config.batch_size}.")
        embeddings = embedder.encode(
            passages,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # CHANGED: L2-normalize before storage. retriever.py uses a raw np.dot() as its
        # similarity function, which is only a valid cosine similarity if both sides are
        # unit vectors. Without this, the dot product is influenced by embedding magnitude,
        # which biases results toward whichever chunks happen to produce larger-norm vectors
        # (typically longer or more lexically dense chunks) rather than true semantic closeness.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embeddings = embeddings / norms

        # Downcast to float16 to halve memory footprint for inference phase
        embeddings_fp16 = embeddings.astype(np.float16)
        out_path = self.config.vector_dir / self.config.vector_filename

        np.save(out_path, embeddings_fp16)
        logger.info(f"Dense vector matrix saved to {out_path}. Dimensions: {embeddings_fp16.shape}.")

        # Enforce strict memory purge
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
        # CHANGED: committed to bge-m3 after eval_runner.py showed it fixing both
        # Swahili failures and the plastic-container multi-hop case, confirmed
        # across 3 runs (e5-small baseline, bge-m3 fp32, bge-m3 fp16) with no
        # accuracy cost from fp16. See REPORT.md Table 6C for the numbers.
        model_path=project_root / "model" / "embeddings" / "bge-m3"
    )

    builder = HybridIndexBuilder(config)
    builder.build_sparse_index()
    builder.build_dense_index()

if __name__ == "__main__":
    main()
