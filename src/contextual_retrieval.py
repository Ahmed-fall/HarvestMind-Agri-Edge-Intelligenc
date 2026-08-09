"""

Implements Anthropic's Contextual Retrieval technique (Sept 2024):
  For each chunk, call the LLM once with the full document text and ask it to
  write 1-2 sentences explaining what this chunk is about in context.
  Prepend that context to the chunk text before indexing.

The context prefix is used ONLY for BM25 tokenization and dense embedding.
The original chunk text is preserved under the key "text" so pipeline.py
builds prompts from clean content only — the model never sees the prefix.


    python src/indexer.py --chunks fao_chunks_ctx.json \
                          --bm25   fao_bm25_ctx.pkl \
                          --vectors fao_vectors_ctx.npy

"""

import gc
import json
import sys
import time
import logging
from pathlib import Path

from llama_cpp import Llama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ContextualRetrieval")

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
KB_DIR      = PROJECT_ROOT / "data" / "knowledge_base"
LLM_PATH    = PROJECT_ROOT / "model" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

INPUT_CHUNKS  = KB_DIR / "fao_chunks.json"
OUTPUT_CHUNKS = KB_DIR / "fao_chunks_ctx.json"

# ── LLM settings for context generation ──────────────────────────────────────
# These are deliberately conservative:
# - n_ctx=2048: enough for the full document summary + one chunk.
#   The full FAO manual text is ~25K tokens which exceeds context. We use a
#   concise document summary instead (see DOCUMENT_SUMMARY below).
# - max_tokens=120: context prefix should be 1-2 tight sentences, not an essay.
# - temperature=0.1: low variance — we want consistent, factual descriptions.
# - n_threads: use all cores here since this is an offline batch job, not
#   the judged inference path. Speed matters, thermals do not.
N_CTX        = 2048
MAX_TOKENS   = 120
TEMPERATURE  = 0.1
REPEAT_PENALTY = 1.1

# ── Concise document summary for context ─────────────────────────────────────
# The full FAO manual (~25K tokens) exceeds the 3B model's context window.
# Using a condensed summary keeps the prompt under N_CTX while still giving
# the model enough document-level context to situate each chunk.
# This covers the five major topic areas across the 47-page manual.
DOCUMENT_SUMMARY = """This document is the FAO manual "Appropriate Seed and Grain Storage Systems
for Small-scale Farmers." It covers five main areas:
1. Physical factors affecting grain storage: moisture content, temperature, relative humidity.
2. Common storage pests: insects (weevils, grain borers, moths, cowpea beetles), mould, termites, rodents, birds.
3. Integrated pest management (IPM): prestorage pest control, storage management, biological control, pesticide use and safety.
4. Prestorage handling by crop: rice, groundnuts, maize, sorghum, millet, beans — covering harvesting, drying, threshing, cleaning.
5. Small-scale storage facilities: traditional (open, semi-open, closed/bancos), modern (grain bags, granaries, cribs, metal silos, hermetic bags, insecticide-treated bags, small containers), and step-by-step silo use instructions."""

# ── Prompt template ───────────────────────────────────────────────────────────
# Instructs the model to write a SHORT situating sentence only.
# "Do not summarize the chunk itself" prevents it from just paraphrasing the text.
# "In the context of this FAO storage manual" keeps it domain-anchored.
CONTEXT_SYSTEM = (
    "You are a precise technical assistant. "
    "Write exactly 1-2 sentences that situate the provided text excerpt "
    "within its parent document. Do not summarize or paraphrase the chunk itself. "
    "State which topic area of the document it belongs to and what specific "
    "sub-topic or question it addresses. Be concrete and specific."
)

def build_context_prompt(section: str, chunk_text: str) -> str:
    return (
        f"<|im_start|>system\n{CONTEXT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Document summary:\n{DOCUMENT_SUMMARY}\n\n"
        f"Section heading: {section}\n\n"
        f"Text excerpt:\n{chunk_text[:600]}\n\n"
        f"Write 1-2 sentences situating this excerpt within the FAO storage manual:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def clean_context_output(raw: str) -> str:
    """Strip any ChatML tokens or trailing whitespace the model may emit."""
    raw = raw.strip()
    # Remove any im_end or im_start tokens that leaked into output
    raw = raw.replace("<|im_end|>", "").replace("<|im_start|>", "")
    # Collapse multiple spaces/newlines
    import re
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw

def vmrss_mb() -> float:
    """Read current process RSS from /proc/self/status. Linux only."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def generate_context(llm: Llama, section: str, chunk_text: str) -> str:
    """Call LLM once to generate a situating context sentence for one chunk."""
    prompt = build_context_prompt(section, chunk_text)
    raw = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        repeat_penalty=REPEAT_PENALTY,
        stop=["<|im_end|>", "<|im_start|>", "\n\n"],
    )
    return clean_context_output(raw["choices"][0]["text"])

def main():
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not INPUT_CHUNKS.exists():
        logger.error(f"Input chunks not found: {INPUT_CHUNKS}")
        logger.error("Run chunker.py first.")
        sys.exit(1)

    if not LLM_PATH.exists():
        logger.error(f"LLM not found: {LLM_PATH}")
        logger.error("Run download_model.sh first.")
        sys.exit(1)

    # ── Load chunks ───────────────────────────────────────────────────────────
    with open(INPUT_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {INPUT_CHUNKS}")

    # ── Check for resume (if output already exists, skip done chunks) ─────────
    done_ids: set = set()
    enriched: list = []
    if OUTPUT_CHUNKS.exists():
        with open(OUTPUT_CHUNKS, encoding="utf-8") as f:
            enriched = json.load(f)
        done_ids = {c["chunk_id"] for c in enriched if c.get("context_prefix")}
        logger.info(f"Resuming: {len(done_ids)} chunks already processed, "
                    f"{len(chunks) - len(done_ids)} remaining.")
    else:
        enriched = []

    remaining = [c for c in chunks if c["chunk_id"] not in done_ids]
    if not remaining:
        logger.info("All chunks already processed. Nothing to do.")
        sys.exit(0)

    # ── Load LLM once ─────────────────────────────────────────────────────────
    logger.info(f"Loading LLM from {LLM_PATH} ...")
    logger.info(f"RSS before LLM load: {vmrss_mb():.1f} MB")
    llm = Llama(
        model_path=str(LLM_PATH),
        n_ctx=N_CTX,
        n_threads=None,    # use all cores — this is offline batch, not judged inference
        verbose=False,
    )
    logger.info(f"RSS after LLM load: {vmrss_mb():.1f} MB")

    # ── Process chunks ────────────────────────────────────────────────────────
    total = len(remaining)
    t0 = time.time()

    for i, chunk in enumerate(remaining, 1):
        chunk_id = chunk["chunk_id"]
        section  = chunk["section"]
        text     = chunk["text"]

        try:
            ctx = generate_context(llm, section, text)
        except Exception as e:
            logger.warning(f"[{chunk_id}] LLM call failed: {e} — using empty context")
            ctx = ""

        # Build enriched chunk: preserve all original fields, add context_prefix
        # and indexed_text (what BM25/dense will see).
        enriched_chunk = {
            **chunk,
            "context_prefix": ctx,
            # indexed_text = what goes into BM25 tokenizer and sentence-transformer.
            # pipeline.py builds prompts from chunk["text"] only — it never reads
            # indexed_text — so the model never sees the prefix at generation time.
            "indexed_text": f"{ctx} {text}".strip() if ctx else text,
        }
        enriched.append(enriched_chunk)

        elapsed = time.time() - t0
        avg_sec = elapsed / i
        eta_sec = avg_sec * (total - i)
        logger.info(
            f"[{i}/{total}] {chunk_id} | section: {section[:40]!r} | "
            f"ctx: {ctx[:80]!r} | ETA: {eta_sec/60:.1f} min"
        )

        # Save after every chunk — allows safe resume if interrupted
        with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del llm
    gc.collect()
    logger.info(f"RSS after LLM release: {vmrss_mb():.1f} MB")

    total_time = time.time() - t0
    logger.info(
        f"Done. {len(enriched)} enriched chunks saved to {OUTPUT_CHUNKS}. "
        f"Total time: {total_time/60:.1f} min "
        f"({total_time/len(remaining):.1f} sec/chunk avg)"
    )

    # ── Spot-check 3 random chunks ────────────────────────────────────────────
    import random
    sample = random.sample(enriched, min(3, len(enriched)))
    print("\n" + "="*60)
    print("SPOT CHECK — 3 random enriched chunks")
    print("="*60)
    for c in sample:
        print(f"\n[{c['chunk_id']}] {c['section']}")
        print(f"  CONTEXT PREFIX: {c.get('context_prefix', '(none)')}")
        print(f"  INDEXED TEXT (first 200 chars): {c.get('indexed_text', '')[:200]}")

if __name__ == "__main__":
    main()