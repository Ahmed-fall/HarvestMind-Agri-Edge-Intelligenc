# HarvestMind

**Offline multilingual post-harvest advisory system grounded in FAO storage guidance.**

HarvestMind answers agricultural storage questions in English and Swahili from an indexed FAO manual, running entirely on-device with no internet dependency. Built for the Africa Deep Tech Challenge (ADTC) 2026 Laptop LLM track.

---

## At a glance

| | |
|---|---|
| **Track** | ADTC 2026 · Laptop LLM · `agriculture` |
| **Languages** | English (`en`), Swahili (`sw`) |
| **Generation model** | Qwen2.5-1.5B-Instruct · GGUF Q5_K_M · llama.cpp |
| **Retrieval** | BM25Okapi + BAAI/bge-m3 dense embeddings · Reciprocal Rank Fusion |
| **Translation** | NLLB-200-distilled-600M |
| **Runtime** | 100% offline · CPU only · no model co-residency |
| **RAM profile** | 2–4 GB peak RSS · 8 GB budget · no OOM risk |

---

## Contents

1. [Problem and approach](#problem-and-approach)
2. [Quick start](#quick-start)
3. [System architecture](#system-architecture)
4. [Engineering decisions](#engineering-decisions)
5. [Multilingual handling](#multilingual-handling)
6. [Resource management](#resource-management)
7. [Offline compliance](#offline-compliance)
8. [CPU-only execution](#cpu-only-execution)
9. [Validation and profiling](#validation-and-profiling)
10. [Repository layout](#repository-layout)
11. [Rebuilding the knowledge base](#rebuilding-the-knowledge-base)
12. [Troubleshooting](#troubleshooting)

---

## Problem and approach

Small-scale farmers across southern Africa lose stored grain to causes the FAO documents precisely — excess moisture, insect pests, inadequate sealing — yet the manual itself is a 47-page English PDF, inaccessible at the point of need for a farmer without connectivity who may ask in Swahili.

HarvestMind solves this with a retrieval-augmented pipeline that answers post-harvest storage questions strictly from that manual. Every response is composed from retrieved source text. The system states when the manual does not cover a question rather than improvising. All models load from local disk; inference performs no network access of any kind.

---

## Quick start

**Requirements:** Linux (Ubuntu 22.04 validated) · Python ≥ 3.11 · 4-core CPU · 8 GB RAM · No GPU required.

```bash
# 0. One-time system prerequisites (llama-cpp-python builds from source)
sudo apt update && sudo apt install -y build-essential cmake python3-dev

# 1. Create and activate an isolated environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download all model weights (~5.8 GB, idempotent, credential-free)
bash download_model.sh

# 4. Ask a question
python src/main.py "What is the safe moisture content for storing maize?"
```

`download_model.sh` fetches the complete inference stack — Qwen2.5-1.5B GGUF (~1.1 GB), bge-m3 embeddings (~2.3 GB), NLLB-600M (~2.4 GB) — using atomic `.partial` renames so interrupted downloads never leave corrupt files.

`stdout` carries the answer only. `stderr` is silent by default. For per-stage memory readings, retrieved chunk IDs, prompt token counts, and translation traces:

```bash
HARVESTMIND_VERBOSE=1 python src/main.py "<question>"
```

Example queries:

```bash
# English
python src/main.py "According to the FAO manual, what are the exact steps \
to seal a plastic storage container to prevent insect infestations?"

# Swahili — detected automatically, answered, and returned in Swahili
python src/main.py "Kulingana na mwongozo wa FAO, nifanye nini kuzuia unyevu \
usiharibu mazao ghalani?"
```

Exit codes: `0` on success (including graceful degraded-answer fallbacks), `1` when the environment cannot serve a request at all (missing weights or indexes).

---

## System architecture

```
query (en | sw)
  │
  ├── language detection — weighted lexical score
  │
  ├── [sw] ──► NLLB sw→en translation        [loads → translates → releases]
  │
  ▼
english question
  │
  ├── retrieval
  │     BM25 top-20 ⊕ dense top-20
  │     → RRF fusion (k=60)
  │     → section-diversity cap (≤ 2 per section)
  │     → top-5 chunks                        [embedder loads → encodes → releases]
  │
  ▼
generation — Qwen2.5-1.5B · ChatML · n_ctx 2048 · greedy decoding
                                               [LLM loads after all other models gone]
  │
  └── [sw query] ──► NLLB en→sw translation ──► answer (sw)
```

### Memory profile by stage

| Stage | Model | Precision | Peak RSS | Lifetime |
|---|---|---|---|---|
| Sparse retrieval | BM25Okapi index | — | ~5 MB | Process lifetime |
| Dense retrieval | BAAI/bge-m3 | fp16 | ~2.0 GB | One query encode |
| Translation | NLLB-200-distilled-600M | fp16 | ~1.3 GB | One text pass (×2 per Swahili turn) |
| Generation | Qwen2.5-1.5B-Instruct | Q5_K_M GGUF | ~1.0 GB | One completion |

No two large models are ever resident simultaneously. End-to-end peak RSS stays in the 2–4 GB range — well within the 8 GB evaluation budget.

---

## Engineering decisions

| Decision | Rationale | Alternative rejected |
|---|---|---|
| RAG over the FAO manual | Answers must be checkable against source text; raw small-model generation hallucinates unsafe advice | Fine-tuning — no citation path; amplifies hallucination risk |
| Contextual enrichment before indexing | LLM-written situating prefixes bridge farmer vocabulary ("plastic container") and manual headings ("Small containers"); prefixes reach indexes only, never the generator | No alternative retained |
| Hybrid BM25 + dense, RRF k=60 | Exact-term hits plus paraphrase and cross-lingual recall without score calibration overhead | Cross-encoder reranker — degraded the correct top result in repeated ablations while adding latency and RAM |
| Section-diversity cap (max 2 per section) | Chunk splits share vocabulary and otherwise crowd the context window with redundant content | Uncapped top-k |
| Translate-around-generation for Swahili | Direct Swahili generation at this quantization level degenerates into repetition regardless of decoding parameters; NLLB is purpose-built for the task | Direct generation with tuned penalties |
| Pure dense retrieval for Swahili queries | BM25 receives only noise from Swahili morphemes against an English corpus; dense path is unaffected | Hybrid fusion for Swahili — actively hurt recall in ablations |
| bge-m3 embeddings, fp16 | Fixed cross-lingual and heading-vocabulary misses that multilingual-e5-small missed; zero fp32 recall cost confirmed across eval set; vectors L2-normalized for valid dot-product cosine similarity | multilingual-e5-small (78–83% Recall@5 vs 94–100%) |
| Greedy decoding (temperature 0) | Judge-facing reproducibility; repeat_penalty 1.1 guards degenerate loops | Sampled decoding |
| Physical-core thread cap (8) | SMT siblings add heat and latency without throughput benefit at this scale | `os.cpu_count()` logical thread count |
| Qwen2.5-1.5B Q5_K_M | Thermal headroom on the reference machine; quality recovered by Q5 precision and retrieval grounding (see REPORT.md §4) | 3B Q4_K_M — measured throttle on reference hardware; sub-Q4 quants — measurable quality loss |

---

## Multilingual handling

**Language detection** uses weighted lexical scoring: unambiguous Swahili content words score 2 points, grammatical concords that cannot appear as standalone English words (`kwa`, `wa`, `ya`, `hii`, …) score 1, threshold ≥ 2. This generalizes to unseen vocabulary rather than matching a fixed phrase list; English questions score zero.

**Pipeline for Swahili queries:**

```
Swahili query
  → NLLB sw→en (beam 4, max 1024 tokens with explicit truncation warning)
  → English retrieval (BM25 disabled, dense only)
  → English generation (Qwen2.5-1.5B)
  → NLLB en→sw
  → Swahili answer
```

If any stage fails, the system logs to `stderr` and continues — routing the untranslated query through the dense path, or returning the English answer — rather than surfacing an error to the user.

---

## Resource management

The pipeline enforces a strict staged memory budget, observable via `HARVESTMIND_VERBOSE=1` (`VmRSS` checkpoints at each stage):

1. Chunks, BM25 index, and the fp16 vector matrix load first — tens of MB, resident for the process lifetime.
2. The embedder loads, encodes one query, and is explicitly released (`del` + `gc.collect()`) before any other large model appears.
3. The translator loads and releases between uses — twice per Swahili turn.
4. The LLM loads last, after the embedder is confirmed gone.

Prompt assembly enforces a conservative context-word budget computed from `n_ctx`, then re-verifies the exact token count via the model tokenizer and sheds lowest-ranked chunks until `prompt_tokens ≤ n_ctx − max_new_tokens`. Context overflow is prevented structurally, not probabilistically.

---

## Offline compliance

The system is verified offline-compliant at runtime:

- Every model loads from a local path under `model/` — no model name strings that could trigger a network lookup.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set before any `transformers` import, eliminating accidental cache-miss network attempts.
- Indexes and the chunk corpus are committed to the repository; nothing is fetched or rebuilt at inference time.
- After `download_model.sh` completes (pre-profiling), the system performs zero outbound requests under any code path.

---

## CPU-only execution

Inference never touches a GPU, enforced at three independent layers:

1. **Environment** — `CUDA_VISIBLE_DEVICES=""` is set before any third-party import in both entry points (`src/main.py`, `src/pipeline.py`), so even a CUDA build of PyTorch sees zero devices.
2. **Explicit device pinning** — the embedder is constructed with `device="cpu"` (`src/retriever.py`); transformers translation models default to CPU and are never moved.
3. **No offload** — llama.cpp is constructed with `n_gpu_layers=0` (`src/pipeline.py`), and the ADTC profiler itself invokes `llama-bench -ngl 0`.

Verify on any machine (from inside the activated environment):

```bash
python - <<'PY'
import os, sys
sys.path.insert(0, "src")
import main  # applies env hardening
import torch
assert torch.cuda.device_count() == 0
print("CPU-only confirmed:", torch.cuda.device_count(), "CUDA devices visible")
PY

# While a query runs, this must list no python process:
nvidia-smi   # errors out harmlessly on machines without NVIDIA drivers
```

Optional: to also shrink the install footprint, swap the CUDA-bundled torch wheel for the CPU build before `pip install -r requirements.txt`:

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## Validation and profiling

The ADTC profiler is open source; install it directly from the official repository:

```bash
# Option A — install straight into the current environment
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"

# Option B — clone for inspection, then install
git clone https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git
pip install ./adtc-profiler
adtc-profiler --help
```

Run a local smoke test after `bash download_model.sh`:

```bash
# Quick pass (~2 min) — throughput/memory/thermal only:
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json \
  --skip-accuracy

# Full pass — includes the arc_easy accuracy component (~10 min):
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json

cat submission.json
```

A valid run produces `submission.json` with `"measured_on": "participant_laptop"` and `"params_match": true`. Profiler source, including thermal monitoring and scoring inputs: [github.com/Africa-Deep-Tech-Foundation/adtc-profiler](https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler).

---

## Repository layout

```
├── metadata.json                  Submission manifest (team, domain, prompts, model)
├── download_model.sh              Weight fetching — idempotent, credential-free
├── REPORT.md                      Technical report and benchmark results
├── requirements.txt               Pinned Python dependencies
├── eval_set.json                  18-item retrieval evaluation set (en/sw, tagged by difficulty)
├── src/
│   ├── main.py                    CLI entry point (judges run this)
│   ├── pipeline.py                Stage orchestration, prompting, fallbacks, memory logging
│   ├── retriever.py               Hybrid retrieval, RRF fusion, diversity cap, language detection
│   ├── translator.py              NLLB wrapper with length and truncation safeguards
│   ├── textproc.py                Shared stemmer and tokenizer (indexer ↔ retriever parity)
│   ├── indexer.py                 Builds BM25 and dense indexes (offline, run once)
│   ├── chunker.py                 FAO PDF → structured chunks (offline, run once)
│   └── contextual_retrieval.py    Contextual prefix generation (offline, run once)
├── data/
│   ├── knowledge_base/            FAO PDF and enriched chunk corpus
│   ├── vector_store/              Committed BM25 pickle and fp16 embedding matrix
│   └── README.md                  Data provenance and rebuild instructions
└── model/                         Downloaded weights (gitignored)
```

---

## Rebuilding the knowledge base

Not required for evaluation. Provided for auditors and reproducibility.

```bash
python src/chunker.py               # FAO PDF → data/knowledge_base/fao_chunks.json
python src/contextual_retrieval.py  # Adds LLM-generated situating prefixes (requires a GGUF)
python src/indexer.py               # Rebuilds data/vector_store/* (requires bge-m3)
```

`src/textproc.py` is imported by both `indexer.py` and `retriever.py`, ensuring query and document tokenization use identical stemming logic and cannot drift between index-build time and query time.

---

## Troubleshooting

| Symptom | Likely cause | Remedy |
|---|---|---|
| `FileNotFoundError: LLM not found …Qwen2.5-1.5B…gguf` | Weights not downloaded | `bash download_model.sh` |
| `NLLB model not found …` | Translation weights missing | `bash download_model.sh` |
| `[TRANSLATE] input hit the 1024-token cap` | Long answer being translated back to Swahili | Informational — tail is deliberately truncated |
| Slow first response | Staged model loading per the RAM budget | Expected — several seconds between stages |
| Context appears trimmed (`[CONTEXT]` on stderr) | Budget guard shed lowest-ranked chunks | Expected under long-chunk worst cases — context stays valid |
| `Missing index files` error | Vector store not committed or deleted | `python src/indexer.py` to rebuild |

---

## License

GPL v3 — see [LICENSE](LICENSE).

The knowledge base derives from an FAO publication available under the FAO open-access policy. See `data/README.md` for full provenance.