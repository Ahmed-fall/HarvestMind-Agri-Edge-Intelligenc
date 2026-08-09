#!/usr/bin/env bash
# Download your model weight files.
#
# Rules:
#   - Must be idempotent (safe to run multiple times).
#   - Must download without any credentials (public URL only).
#   - The output path must match `_runtime.model_path` in metadata.json.
#
# Default invocation downloads exactly the locked submission stack:
#   Qwen2.5-3B-Instruct Q4_K_M + multilingual-e5-small + ms-marco-MiniLM-L-6-v2
#
# CHANGED: two extra models are now available behind opt-in env flags, for the
# EMB-3 and SW-B experiments your own plan scopes in Table 6C / Section 3D but
# that haven't been run yet:
#   WITH_BGE_M3=1   ./download_model.sh   -> also fetches BAAI/bge-m3
#   WITH_NLLB=1     ./download_model.sh   -> also fetches NLLB-200-distilled-600M
# Neither is touched by the default run, so this stays idempotent and correct
# for final submission. Only fold one into the real pipeline (indexer.py /
# retriever.py model_path, metadata.json) once you've measured it beats the
# current stack on eval_runner.py -- don't ship an untested model swap.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"

WITH_BGE_M3="${WITH_BGE_M3:-0}"
WITH_NLLB="${WITH_NLLB:-0}"

# ── 1. LLM Configuration ───────────────────────────────────────────────────────
LLM_MODEL_FILE="$MODEL_DIR/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
LLM_MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# ── 2. Retrieval Models Configuration ──────────────────────────────────────────
EMBED_DIR="$MODEL_DIR/embeddings/multilingual-e5-small"
RERANKER_DIR="$MODEL_DIR/reranker/ms-marco-MiniLM-L-6-v2"

# ── 3. Experimental / opt-in models ────────────────────────────────────────────
BGE_M3_DIR="$MODEL_DIR/embeddings/bge-m3"
NLLB_DIR="$MODEL_DIR/translation/nllb-200-distilled-600M"
# ───────────────────────────────────────────────────────────────────────────────

mkdir -p "$MODEL_DIR"

# ==========================================
# 1. Download LLM Weights
# ==========================================
if [[ -f "$LLM_MODEL_FILE" ]]; then
  echo "LLM already present at $LLM_MODEL_FILE — skipping download"
else
  echo "Downloading $LLM_MODEL_URL → $LLM_MODEL_FILE (~2.2 GB)…"

  # Robust fallback between curl and wget
  if command -v curl > /dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$LLM_MODEL_FILE.partial" "$LLM_MODEL_URL"
  elif command -v wget > /dev/null 2>&1; then
    wget --show-progress -O "$LLM_MODEL_FILE.partial" "$LLM_MODEL_URL"
  else
    echo "error: neither curl nor wget found" >&2
    exit 1
  fi

  # Atomic move to prevent corrupted partial downloads from being read
  mv "$LLM_MODEL_FILE.partial" "$LLM_MODEL_FILE"
  echo "Done: $LLM_MODEL_FILE"
fi

# ==========================================
# 2. Download Embedding Model (multilingual-e5-small)
# ==========================================
if [[ -d "$EMBED_DIR" ]]; then
  echo "Embedding model already present at $EMBED_DIR — skipping download"
else
  echo "Downloading Embedding model to $EMBED_DIR…"
  python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('intfloat/multilingual-e5-small')
m.save('$EMBED_DIR')
print('Embedding model saved.')
"
fi

# ==========================================
# 3. Download Reranker Model (ms-marco-MiniLM-L-6-v2)
# ==========================================
if [[ -d "$RERANKER_DIR" ]]; then
  echo "Reranker model already present at $RERANKER_DIR — skipping download"
else
  echo "Downloading Reranker model to $RERANKER_DIR…"
  python -c "
from sentence_transformers import CrossEncoder
m = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
m.save('$RERANKER_DIR')
print('Reranker saved.')
"
fi

# ==========================================
# 4. [OPT-IN] Download BAAI/bge-m3 -- for EMB-3 experiment (Table 6C)
# ==========================================
if [[ "$WITH_BGE_M3" == "1" ]]; then
  if [[ -d "$BGE_M3_DIR" ]]; then
    echo "bge-m3 already present at $BGE_M3_DIR — skipping download"
  else
    echo "Downloading BAAI/bge-m3 to $BGE_M3_DIR (~1.2 GB)…"
    python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-m3')
m.save('$BGE_M3_DIR')
print('bge-m3 saved.')
"
  fi
else
  echo "Skipping bge-m3 (set WITH_BGE_M3=1 to fetch it for the EMB-3 experiment)"
fi

# ==========================================
# 5. [OPT-IN] Download NLLB-200-distilled-600M -- for SW-B experiment (Section 3D)
# ==========================================
if [[ "$WITH_NLLB" == "1" ]]; then
  if [[ -d "$NLLB_DIR" ]]; then
    echo "NLLB-200-distilled-600M already present at $NLLB_DIR — skipping download"
  else
    echo "Downloading facebook/nllb-200-distilled-600M to $NLLB_DIR (~2.5 GB fp32 on disk)…"
    python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
name = 'facebook/nllb-200-distilled-600M'
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name)
tok.save_pretrained('$NLLB_DIR')
model.save_pretrained('$NLLB_DIR')
print('NLLB-200-distilled-600M saved.')
"
  fi
else
  echo "Skipping NLLB (set WITH_NLLB=1 to fetch it for the SW-B experiment)"
fi

echo "All requested models successfully downloaded and ready for offline inference."