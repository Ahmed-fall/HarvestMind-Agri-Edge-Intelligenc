# Technical Report — HarvestMind

| | |
|---|---|
| Team ID | `harvestmind-agri-edge-intelligence` |
| Domain | `agriculture` |
| Languages | English, Swahili |
| Starting model | Qwen2.5-3B-Instruct, GGUF Q4_K_M |
| Shipped model | Qwen2.5-1.5B-Instruct, GGUF Q5_K_M (llama.cpp) |
| Reference machine | Intel Core i7-10700 (8C/16T), 7.5 GB RAM, Ubuntu 22.04, CPU-only |
| Validation machine | Intel Core i7-14700HX laptop, 30.6 GB RAM (GPU unused), Ubuntu 26.04 |

---

## 1. Problem

Small-scale grain farmers across southern Africa lose a material share of
every harvest to causes that are well understood and documented: grain stored
above safe moisture thresholds develops mould; weevils, larger grain borers,
and rodents exploit poorly sealed stores; correct practice differs by crop and
storage method. The FAO manual *"Appropriate Seed and Grain Storage Systems
for Small-scale Farmers"* covers exactly these questions in actionable detail
— safe moisture levels per crop, drying tests without equipment, sealing
procedures, pesticide safety — but it is a 47-page English PDF that assumes a
reader who can find the right page.

**Target user.** A smallholder farmer or village store keeper in rural
southern Africa, working between harvest and market, often without reliable
internet or mobile data, and frequently Swahili-speaking. They ask practical
questions ("how do I know my maize is dry enough?", "what do I do to stop
moisture destroying crops in the granary?") and need answers that are
specific enough to act on — numbers, thresholds, ordered steps — and safe,
because invented pesticide dosages or storage advice has direct human and
economic cost.

This is why local matters: no connectivity at the point of need, no cost per
query, and full control over what the system may claim. The system must run
on an 8 GB CPU-only laptop, answer in English or Swahili, and decline beyond
what the source supports rather than improvise.

## 2. Design decisions

**Starting point.** Development began from **Qwen2.5-3B-Instruct at GGUF
Q4_K_M** — at the time the best-regarded quality-per-gigabyte instruct model
in the 3B class, with Q4_K_M the standard balance of fidelity and footprint
for CPU inference. Architecture around it was retrieval-augmented generation
over the FAO manual, because the failure mode that matters here is a
confident, invented recommendation: fine-tuning cannot provide checkable
sources and small-model fine-tunes still hallucinate out-of-support answers.

**What shipped and why.** Thermal profiling of the starting configuration on
the reference-class machine tripped the challenge's throttle threshold
(94 °C peak, `throttled: true`; see §3), and submission code cannot influence
that measurement — the weight file is the only controlled lever. The shipped
model is therefore **Qwen2.5-1.5B-Instruct at Q5_K_M (~1.12 GB)**: the
parameter reduction cuts compute per token roughly in half, while stepping
Q4→Q5 recovers precision per weight, keeping measured raw quality nearly
unchanged (arc_easy acc_norm 0.74 vs 0.76 baseline, n=50 — inside sampling
noise). Retrieval grounding further insulates answer quality from raw
benchmark differences.

**Alternatives evaluated and rejected:**

| Alternative | Reason rejected |
|---|---|
| Qwen2.5-3B Q4_K_M (kept as reference) | Tripped the 85 °C thermal threshold in the profiler window |
| Qwen2.5-3B IQ4_XS | Modest relief only; retained (`WITH_3B=1`) as a benchmark candidate |
| Sub-Q4 quants of 3B | Quality fell faster than size |
| Fine-tuning instead of RAG | No citation path; hallucination risk remains |
| Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) | Displaced the correct top result in repeated ablations; added latency and RAM |
| multilingual-e5-small embeddings | Missed cross-lingual and heading-vocabulary cases that bge-m3 resolved |
| Direct Swahili generation | Degenerate repetition even with tuned penalties; replaced by NLLB translate-around-generation |
| Hybrid BM25 fusion for Swahili queries | BM25 receives only noise from Swahili morphemes; pure dense measured better |

Retrieval design: contextual enrichment before indexing (LLM-written
situating prefixes reach BM25/embeddings only, never the generator),
reciprocal-rank fusion of BM25 and bge-m3 dense rankings (k = 60, top-20
each), a max-two-chunks-per-section diversity cap, and a token-budget guard
that makes prompt overflow structurally impossible.

## 3. Constraints

- **Hardware (8 GB laptop, 4 vCPU, integrated GPU unused).** All inference is
  CPU llama.cpp (`-ngl 0`). Memory is managed as staged budgets: embedder,
  translator, and LLM load sequentially and release between stages — no two
  large models are ever co-resident. Peak end-to-end process RSS observed:
  3.9 GB including both Swahili translation passes.
- **Thermal ceiling.** The profiler flags throttling when core temperature
  crosses 85 °C during a sustained all-core `llama-bench` window
  (`-ngl 0 -p 512 -n 128`). Sustained load saturates laptop cooling
  regardless of software tuning; the starting 3B configuration measured
  94 °C, and even the lighter shipped configuration read 86 °C on a larger
  development laptop. This constraint drove the final model choice above all
  other quality considerations.
- **Connectivity.** Zero network at inference: all weights load from local
  disk, `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` are set before any
  transformers import, and both retrieval indexes are committed to the
  repository. Only `download_model.sh`, which runs before profiling, uses
  the network.
- **Data reality.** One curated 54-chunk corpus is the entire knowledge base,
  so per-chunk retrieval quality mattered more than corpus breadth — hence
  the investment in contextual enrichment, heading-aware embedding, and
  diversity capping.
- **Language coverage.** Swahili support must survive hidden prompts using
  vocabulary never seen in development; detection therefore uses weighted
  lexical scoring (content words 2 pts, non-English concords 1 pt, threshold
  ≥ 2) rather than a fixed keyword list, and Swahili queries retrieve with
  pure dense ranking.

## 4. Benchmarks

Development machine: Intel Core i7-14700HX laptop (GPU unused), Ubuntu 26.04,
profiler v0.1.0 participant run against the shipped GGUF
(`params_match: true`). Reference-class baseline shown for comparison:
Intel Core i7-10700, 7.5 GB RAM — the ADTC standard evaluation profile.

| Metric | Shipped 1.5B Q5_K_M (dev machine, measured) | Baseline 3B Q4_K_M (reference class, measured) |
|---|---|---|
| Generation throughput | 43.4 t/s | 10.17 t/s |
| First-token latency (512-tok prompt) | 3682 ms | 7580.7 ms |
| Peak RSS | 1260 MB | 3285 MB |
| Core temp peak | 86 °C (throttled flag set) | 94 °C (throttled) |
| arc_easy acc_norm (n=50) | 0.74 | 0.76 |

End-to-end pipeline on the development machine: ≈ 25 s per English answer,
≈ 57 s per Swahili answer (two NLLB passes included); byte-identical answers
across repeated identical invocations under greedy decoding.

Projection to the reference machine (scaling estimate, not a measurement):
generation is memory-bandwidth-bound, so throughput scales ≈ inverse weight
size — 10.17 × 1.93/1.12 ≈ 17.5 t/s expected (**16–18 t/s**); first-token
latency scales ≈ parameter ratio, giving **3800–4200 ms**; peak RSS
**2000–2200 MB**. The reference-machine thermal peak is not certified;
audit-run values are authoritative.

Reproducibility: greedy decoding makes repeated runs on one machine/build
byte-identical; thread count derives from CPU affinity (physical cores, cap
8), removing a common drift source; retrieval is deterministic given the
committed indexes. Reproduce with `bash download_model.sh` then
`adtc-profiler run --submission . --mode participant --output submission.json`.

## Limitations

- Reference-machine thermal margin is unverified; the 85 °C threshold was
  still reached on larger laptop cooling, so headroom should be confirmed
  where hardware access allows (`WITH_3B=1 bash download_model.sh &&
  ./scripts/bench_candidates.sh`).
- When a question names a method the manual treats elsewhere (e.g., sealing a
  plastic container, with silo-sealing steps retrieved as strong context),
  the generator can merge procedures despite its system-prompt prohibition;
  answers stay grounded but may mix methods.
- n=50 audit metrics carry ±0.07 confidence bands; single runs are not
  precise model comparisons.

These are self-reported development benchmarks. Official scores are measured
by the ADTC profiler on the standard evaluation machine.
