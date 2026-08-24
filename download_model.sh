#!/usr/bin/env bash
# Download all model weight files required by this submission.
#
# Rules satisfied:
#   - Idempotent: safe to run multiple times; existing files are skipped.
#   - Credential-free: every URL is public Hugging Face.
#   - Output paths match `_runtime.model_path` in metadata.json.
#
# The default invocation downloads exactly the stack that src/ loads:
#   1. Qwen2.5-1.5B-Instruct Q5_K_M (GGUF, generation)
#   2. BAAI/bge-m3                  (multilingual embeddings, fp16 at runtime)
#   3. facebook/nllb-200-distilled-600M (Swahili <-> English translation)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"

fetch_file() {
  # fetch_file <url> <dest> — curl/wget fallback with atomic rename
  local url="$1" dest="$2"
  echo "Downloading $url → $dest …"
  if command -v curl > /dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$dest.partial" "$url"
  elif command -v wget > /dev/null 2>&1; then
    wget --show-progress -O "$dest.partial" "$url"
  else
    echo "error: neither curl nor wget found" >&2
    exit 1
  fi
  mv "$dest.partial" "$dest"
}

run_python() {
  if command -v python > /dev/null 2>&1; then
    python -c "$1"
  else
    python3 -c "$1"
  fi
}

mkdir -p "$MODEL_DIR"

# ==========================================
# 1. Generation LLM (Qwen2.5-1.5B-Instruct, GGUF Q5_K_M)
# ==========================================
LLM_MODEL_FILE="$MODEL_DIR/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf"
LLM_MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf"

if [[ -f "$LLM_MODEL_FILE" ]]; then
  echo "LLM already present at $LLM_MODEL_FILE — skipping download"
else
  fetch_file "$LLM_MODEL_URL" "$LLM_MODEL_FILE"
  echo "Done: $LLM_MODEL_FILE"
fi

# ==========================================
# 2. Embedding model (BAAI/bge-m3) — used by retriever.py / indexer.py
# ==========================================
EMBED_DIR="$MODEL_DIR/embeddings/bge-m3"
if [[ -d "$EMBED_DIR" ]]; then
  echo "Embedding model already present at $EMBED_DIR — skipping download"
else
  echo "Downloading embedding model to $EMBED_DIR (~2.3 GB on disk)…"
  run_python "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-m3')
m.save('$EMBED_DIR')
print('bge-m3 saved.')
"
fi

# ==========================================
# 3. Translation model (NLLB-200-distilled-600M) — used by translator.py
# ==========================================
NLLB_DIR="$MODEL_DIR/translation/nllb-200-distilled-600M"
if [[ -d "$NLLB_DIR" ]]; then
  echo "Translation model already present at $NLLB_DIR — skipping download"
else
  echo "Downloading NLLB-200-distilled-600M to $NLLB_DIR (~2.4 GB on disk)…"
  run_python "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
name = 'facebook/nllb-200-distilled-600M'
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name)
tok.save_pretrained('$NLLB_DIR')
model.save_pretrained('$NLLB_DIR')
print('NLLB-200-distilled-600M saved.')
"
fi

echo "All requested models downloaded and ready for offline inference."
