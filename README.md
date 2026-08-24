# HarvestMind

**Offline agricultural advisory grounded in FAO storage guidance — English and Swahili, CPU-only, within an 8 GB RAM budget.**

| | |
|---|---|
| Track | ADTC 2026 · Laptop LLM · `agriculture` |
| Languages | English (`en`), Swahili (`sw`) |
| Generation model | Qwen2.5-1.5B-Instruct, GGUF Q5_K_M, llama.cpp |
| Retrieval | BM25Okapi + bge-m3 dense embeddings, reciprocal-rank fusion |
| Translation | NLLB-200-distilled-600M |
| Runtime profile | 100% offline, CPU only, no model co-residency |

The technical write-up is in [REPORT.md](REPORT.md).

---

## Contents

1. [Overview](#overview)
2. [Requirements and installation](#requirements-and-installation)
3. [Usage](#usage)
4. [System architecture](#system-architecture)
5. [Engineering decisions](#engineering-decisions)
6. [Multilingual handling](#multilingual-handling)
7. [Resource management](#resource-management)
8. [Offline compliance](#offline-compliance)
9. [Repository layout](#repository-layout)
10. [Rebuilding the knowledge base](#rebuilding-the-knowledge-base)
11. [Benchmark tooling](#benchmark-tooling)
12. [Troubleshooting](#troubleshooting)

## Overview

Small-scale farmers lose stored grain to causes the FAO manual documents
precisely: excess moisture, insect pests, inadequate sealing. The manual
itself is a 47-page English PDF, unusable at the point of need for a farmer
without connectivity who may ask in Swahili.

HarvestMind answers post-harvest storage questions strictly from that manual.
Every response is composed from retrieved manual text; the system states when
the manual does not cover a question rather than improvising. All models load
from local disk, so inference performs no network access of any kind.

## Requirements and installation

- Linux (validated on Ubuntu 22.04), Python ≥ 3.11, 4 vCPU class CPU, 8 GB RAM
- No GPU required

```bash
pip install -r requirements.txt
bash download_model.sh        # ~5.8 GB of public Hugging Face weights; idempotent
```

`download_model.sh` fetches exactly the stack loaded by the source code:
Qwen2.5-1.5B GGUF (~1.1 GB), bge-m3 embeddings (~2.3 GB), NLLB-600M (~2.4 GB),
using atomic `.partial` renames so interrupted runs never leave corrupt files.
An optional thermal A/B candidate (Qwen2.5-3B IQ4_XS) is available via
`WITH_3B=1`.

Retrieval indexes (`data/vector_store/`) are committed to the repository;
no embedding or index construction occurs at inference time.

## Usage

```bash
# English
python src/main.py "According to the FAO manual, what are the exact steps \
to seal a plastic storage container to prevent insect infestations?"

# Swahili (detected, translated, answered, translated back)
python src/main.py "Kulingana na mwongozo wa FAO, nifanye nini kuzuia unyevu \
usiharibu mazao ghalani?"
```

stdout carries the answer only. Diagnostics go to stderr:

```bash
HARVESTMIND_VERBOSE=1 python src/main.py "... "
```

which reports per-stage resident memory, retrieved chunk IDs, prompt token
count, and translation results.

Exit codes: `0` on success (including degraded-answer fallbacks), `1` when the
environment cannot serve a request at all (missing weights or indexes).

## System architecture

```
query (en | sw)
  │  language detection — weighted lexical score
  ├─ sw ──► NLLB sw→en                      [translator loaded, then released]
  ▼
english question
  │  retrieval — BM25 top-20 ⊕ dense top-20 → RRF k=60
  │             → section-diversity cap (≤2 per section) → top-5 chunks
  │                                          [embedder loaded for one encode, then released]
  ▼
generation — Qwen2.5-1.5B · ChatML · ctx 2048 · greedy decoding
  │                                         [LLM loaded after other models are gone]
  ▼
answer (en)
  └─ sw query ──► NLLB en→sw ──► answer (sw)
```

Models are loaded sequentially and released between stages:

| Stage | Model | Precision | Resident (approx.) | Lifetime |
|---|---|---|---|---|
| Sparse retrieval | BM25Okapi index | — | ~5 MB | process lifetime |
| Dense retrieval | BAAI/bge-m3 | fp16 | ~2.0 GB | one query encode |
| Translation (sw) | NLLB-200-distilled-600M | fp16 | ~1.3 GB | one text pass, twice per turn |
| Generation | Qwen2.5-1.5B-Instruct | Q5_K_M | ~2.0 GB | one completion |

No two large models are ever resident simultaneously; see
[Resource management](#resource-management).

## Engineering decisions

| Decision | Rationale | Alternative rejected |
|---|---|---|
| RAG over the FAO manual | Answers must be checkable against source text; raw small-model generation hallucinates unsafe advice | Fine-tuning (no citation path; amplifies hallucination risk) |
| Contextual enrichment before indexing | LLM-written situating prefixes bridge farmer vocabulary ("plastic container") and manual headings ("Small containers"); prefixes reach indexes only, never the generator | None retained |
| Hybrid BM25 + dense, RRF k=60 | Exact-term hits plus paraphrase/cross-lingual recall without score calibration | Cross-encoder reranker — degraded the correct top result in repeated ablations while adding latency and RAM |
| Section-diversity cap (max 2) | Chunk splits share vocabulary and otherwise crowd the context window | Uncapped top-k |
| Translate-around-generation for Swahili | Direct Swahili generation degenerates into repetition; NLLB is purpose-built | Direct generation with tuned penalties |
| Pure dense retrieval for Swahili queries | BM25 receives only noise from Swahili morphemes against an English corpus | Hybrid fusion for sw |
| bge-m3 embeddings, fp16 | Fixed cross-lingual and heading-vocabulary misses at no fp32 recall cost; vectors L2-normalized so dot products equal cosine similarity | multilingual-e5-small |
| Greedy decoding (temperature 0) | Judge-facing reproducibility; repeat_penalty 1.1 guards degenerate loops | Sampled decoding |
| Physical-core thread count (cap 8) | SMT siblings add heat and latency without throughput benefit at this scale | `os.cpu_count()` logical threads |
| Qwen2.5-1.5B Q5_K_M | Thermal headroom on the reference machine; quality recovered by Q5 precision and retrieval grounding (see REPORT.md §4) | 3B Q4_K_M (measured throttle), sub-Q4 quants (quality loss) |

## Multilingual handling

Swahili detection uses weighted lexical scoring: unambiguous content words
score 2 points, grammatical concords that cannot occur as standalone English
words (`kwa`, `wa`, `ya`, `hii`, …) score 1, threshold ≥ 2. This generalizes
to unseen vocabulary rather than matching a fixed phrase list; English
questions score zero.

Pipeline: translate query (beam 4) → English retrieval and generation →
translate answer back. If any stage fails, the system logs to stderr and
continues (untranslated query through the multilingual dense path, or the
English answer) rather than discarding the result. NLLB sequences are capped
at 1024 tokens with explicit truncation warnings.

## Resource management

The pipeline enforces a staged memory budget, observable via
`HARVESTMIND_VERBOSE=1` (`VmRSS` checkpoints):

1. Chunks, BM25 index, and vector matrix load first (tens of MB, resident).
2. The embedder loads, encodes one query, and is released before any other
   large model appears.
3. The translator loads twice per Swahili turn and is released between uses.
4. The LLM loads last; peak process RSS stays in the 2–4 GB band end-to-end,
   inside the 8 GB target with margin.

Prompt assembly enforces a conservative context-word budget computed from
`n_ctx`, then re-verifies the exact token count with the model tokenizer and
sheds lowest-ranked chunks until `prompt_tokens ≤ n_ctx − max_new_tokens`.
Context overflow is prevented structurally, not probabilistically.

## Offline compliance

- Every model loads from a local directory under `model/`.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set before any
  transformers import, eliminating accidental network attempts.
- Indexes and corpus are committed; nothing is fetched or rebuilt at runtime.
- After `download_model.sh` completes (pre-profiling), the system performs
  zero outbound requests.

## Repository layout

```
metadata.json                  Submission manifest (team, domain, prompts, model)
download_model.sh              Weight fetching; idempotent, credential-free
REPORT.md                      Technical report and benchmarks
requirements.txt               Pinned dependencies
src/
  main.py                      CLI entry point used by judges
  pipeline.py                  Staging, prompting, fallbacks, memory logging
  retriever.py                 Hybrid retrieval, diversity cap, language detection
  translator.py                NLLB wrapper with length/truncation safeguards
  textproc.py                  Shared stemmer/tokenizer (indexer↔retriever parity)
  indexer.py                   Builds BM25 + dense indexes (offline)
  chunker.py                   PDF → chunks (offline)
  contextual_retrieval.py      Contextual prefix generation (offline)
scripts/
  bench_candidates.sh          Profiler-equivalent thermal/perf A/B for GGUFs
data/
  knowledge_base/              FAO PDF, enriched chunk corpus (see data/README.md)
  vector_store/                Committed BM25 pickle and fp16 embedding matrix
eval_set.json                  18-item retrieval evaluation set (en/sw, tagged)
model/                         Downloaded weights (gitignored)
```

## Rebuilding the knowledge base

Not required for evaluation; provided for auditors.

```bash
python src/chunker.py               # PDF → data/knowledge_base/fao_chunks.json
python src/contextual_retrieval.py  # adds contextual prefixes (requires a GGUF)
python src/indexer.py               # rebuilds data/vector_store/* (requires bge-m3)
```

`src/textproc.py` is imported by both indexer and retriever, so query and
document tokenization cannot drift apart.

## Benchmark tooling

`scripts/bench_candidates.sh` reproduces the profiler's invocation
(`llama-bench -ngl 0 -p 512 -n 128`) for every GGUF present and samples peak
CPU temperature during each window:

```bash
WITH_3B=1 bash download_model.sh
./scripts/bench_candidates.sh
```

Output columns: weights size, generation rate (t/s), first-token latency
projection, peak temperature (°C). The scorer flags throttling at ≥ 85 °C.

## Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| `FileNotFoundError: LLM not found …Qwen2.5-1.5B…gguf` | Weights not fetched | `bash download_model.sh` |
| `NLLB model not found …` | Translation weights missing | `bash download_model.sh` |
| `[TRANSLATE] input hit the 1024-token cap` warning | Very long answer being translated | Informational; tail truncated deliberately |
| Slow first response | Staged model loads honor the RAM budget | Expected behavior; several seconds |
| Context looks trimmed (`[CONTEXT]` stderr lines) | Budget guard dropped low-ranked chunks | Expected under long-chunk worst cases |

## License

GPL v3 — see [LICENSE](LICENSE). The knowledge base is an FAO publication.
