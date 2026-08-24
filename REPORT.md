# Technical Report — HarvestMind

| | |
|---|---|
| Team ID | `harvestmind-agri-edge-intelligence` |
| Team | Ahmed Madi (RA), Dr. Rami Zewail (PI) — E-JUST |
| Domain | `agriculture` |
| Languages | English, Swahili |
| Starting model | Qwen2.5-3B-Instruct, GGUF Q4_K_M |
| Shipped model | Qwen2.5-1.5B-Instruct, GGUF Q5_K_M (llama.cpp) |
| Evaluation target | Intel i5 10th–12th gen, 4 vCPUs, 8 GB DDR4, Ubuntu 22.04, CPU-only |

---

## 1. Problem

Small-scale grain farmers across southern Africa lose a material share of
every harvest to causes that are well understood and documented: grain stored
above safe moisture thresholds develops mould; weevils, larger grain borers,
and rodents exploit poorly sealed stores; correct practice differs by crop and
storage method. The FAO manual *"Appropriate Seed and Grain Storage Systems
for Small-scale Farmers"* (47 pages) covers exactly these questions in
actionable detail — safe moisture levels per crop, drying tests without
equipment, sealing procedures, pesticide safety — but it assumes a reader who
can find the right page.

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
on the 8 GB evaluation profile, answer in English or Swahili, and decline
beyond what the source supports rather than improvise — including where the
question presumes something the manual does not recommend (e.g., the manual
describes sealing small containers via jars, wax, and airtight lids rather
than endorsing plastic containers specifically).

## 2. Design decisions

**Starting point.** Development began from **Qwen2.5-3B-Instruct at GGUF
Q4_K_M**, selected over peer models in a controlled sweep on the development
laptop:

| Candidate | Config | Gen. rate | RAM | Peak temp | Verdict |
|---|---|---|---|---|---|
| **Qwen2.5-3B-Instruct** | Q4_K_M | 21.63 t/s | 3.29 GB | 82 °C | selected |
| Phi-3-mini-4k-instruct | Q4_K_M, 4 threads | 18.87 t/s | 3.83 GB | 89 °C | rejected — thermal headroom |
| Phi-3-mini-4k-instruct | Q4_K_M, 3 threads | 14.48 t/s | 3.83 GB | 64 °C | rejected — throughput deficit |

Q4_K_M was the starting quantization as the standard quality-per-gigabyte
balance for CPU inference. Architecture around it was retrieval-augmented
generation over the FAO manual: the failure mode that matters here is a
confident, invented recommendation, and fine-tuning cannot provide checkable
sources while small-model fine-tunes still hallucinate out-of-support answers.

**Why the shipped quantization changed.** Thermal profiling of the starting
configuration against the challenge's sustained-load window tripped the
85 °C throttle threshold (94 °C peak on the development laptop; 93 °C on an
i7-10700 desktop proxy — both air-cooled machines, indicating a cooling-bound
workload rather than a software property). Submission code cannot influence
that measurement; the weight file is the only controlled lever. Under the
challenge scoring model (`S_total = 0.5·S_acc + 0.3·S_perf + 0.2·S_eff −
P_thermal`, with `S_eff = 100·(7 GB − peak RSS)/7 GB` and `P_thermal` a flat
penalty when throttled), the arithmetic favors downsizing decisively:

- Moving 3.29 GB → ~2.0 GB peak RSS raises `S_eff` from 53.0 to ≈ 71 (derived);
- avoiding the throttle flag recovers up to 10 points;
- measured accuracy moved 0.76 → 0.74 (arc_easy acc_norm, n = 50 each),
  inside the ±0.07 sampling band of a 50-item run.

The shipped model is therefore **Qwen2.5-1.5B-Instruct at Q5_K_M (~1.12 GB)**:
the parameter reduction roughly halves compute per token while stepping
Q4→Q5 preserves precision per weight. Retrieval grounding further insulates
answer fidelity from raw-benchmark differences.

**Retrieval engineering** (all choices made against an internal 18-question
held-out evaluation set spanning the manual, tagged by difficulty and
language — not tuned solely to the mandatory prompts):

- Contextual enrichment before indexing: one LLM pass per chunk writes a
  situating prefix that reaches BM25/embeddings only, never the generator.
- Embeddings: bge-m3 replaced multilingual-e5-small after an A/B measured
  Recall@5 of 94–100% versus 78–83%; fp16 halved embedder memory with no
  recall loss; passages are prefixed with their section heading before
  embedding; all vectors are L2-normalized so dot products equal cosine.
- Hybrid BM25 + dense reciprocal-rank fusion (k = 60, top-20 each) for
  English; pure dense ranking for Swahili, where BM25 contributes only noise.
- Section-diversity cap (≤ 2 chunks per base section) with top-5 context;
  the window was raised 3→4→5 through testing.
- A cross-encoder reranker (ms-marco-MiniLM-L-6-v2) was tested three times
  and removed: it consistently reduced accuracy and specifically broke the
  plastic-container case despite correct upstream retrieval.
- A token-budget guard plus post-load tokenizer verification makes prompt
  overflow structurally impossible.

**Reliability work during development.** Fixes that materially changed
behavior: a custom suffix stemmer so BM25 matches singular/plural query terms
(moved the correct chunk from rank 9 to rank 5 for the container question);
a chunker heading-detection regex that split mid-paragraph; a block-parsing
bug that silently discarded the "Small containers" section — the answer
source for the first mandatory prompt; a model-refusal failure mode addressed
in the system contract; and numeric-direction errors (thresholds restated
inverted) mitigated through explicit prompt instructions, retained in the
shipped system prompt.

## 3. Constraints

- **Evaluation hardware.** 4 vCPUs (i5 10th–12th gen class), 8 GB DDR4,
  integrated GPU unused — pure CPU llama.cpp with GGUF weights on Ubuntu
  22.04. Memory is managed as staged budgets: embedder, translator, and LLM
  load sequentially and release between stages; no two large models are ever
  co-resident (peak end-to-end process RSS observed: 3.9 GB including both
  Swahili translation passes). Out-of-memory is disqualifying, and the
  efficiency score budgets 7 GB, so headroom is deliberate.
- **Thermal ceiling.** The scorer applies a flat penalty when core temperature
  crosses 85 °C or the throttle flag sets during the sustained `llama-bench`
  window. Both test machines tripped it (94 °C / 93 °C) under all-core load —
  a workload-cooling property, not a software defect — which drove the model
  downsize above all other quality considerations.
- **Connectivity.** Zero network at inference: weights load from local disk,
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` are set before any transformers
  import, and both indexes are committed to the repository. Only
  `download_model.sh`, which runs before profiling, uses the network.
- **Data reality.** One curated 54-chunk corpus built from the FAO manual is
  the entire knowledge base, so per-chunk retrieval quality mattered more
  than corpus breadth — hence contextual enrichment, heading-aware embedding,
  and diversity capping.
- **Language coverage.** Swahili support must survive hidden prompts using
  vocabulary never seen in development; detection therefore uses weighted
  lexical scoring (content words 2 pts, non-English concords 1 pt, threshold
  ≥ 2) rather than a fixed keyword list. Direct Swahili generation was tested
  and found unreliable at this scale — degenerate repetition regardless of
  decoding parameters — so queries translate sw→en, run the proven English
  pipeline, and translate back via NLLB-200-distilled-600M.

## 4. Benchmarks

All figures below are measured outputs of `llama-bench` or `adtc-profiler`
v0.1.0 runs against the named model files; configurations differ per row and
are labeled. Development machine: Lenovo Legion 5, i7-14700HX (GPU unused).
Proxy/reference machine: i7-10700 desktop, 7.5 GB RAM.

**Shipped model (Qwen2.5-1.5B-Instruct Q5_K_M):**

| Metric | 4 threads (eval-profile match) | Default threads (profiler run) |
|---|---|---|
| Generation throughput | 34.23 t/s | 43.4 t/s |
| First-token latency (512-tok prompt) | ≈ 7000 ms | 3682 ms |
| Peak RSS | — | 1260 MB |
| Core temp peak | — | 86 °C (throttled flag set) |
| arc_easy acc_norm (n=50) | — | 0.74 |

**Baseline configuration (Qwen2.5-3B-Instruct Q4_K_M):**

| Machine | Throughput | Peak RSS | Temp | arc_easy |
|---|---|---|---|---|
| Development laptop (profiler) | 24.66 t/s | 3313 MB | 94 °C | 0.76 |
| i7-10700 desktop proxy (profiler) | 10.17 t/s | 3285 MB | 93 °C | 0.76 |

**Projection to the evaluation target** (i5-class, 4 vCPUs; scaling estimate,
not a measurement): generation is memory-bandwidth-bound, so throughput
scales ≈ inverse weight size from the proxy baseline — 10.17 × 1.93/1.12 ≈
17.5 t/s expected (**16–18 t/s** band); first-token latency scales ≈
parameter ratio from the proxy's 7580 ms (**3800–4200 ms**); peak RSS
**2000–2200 MB** (`S_eff` ≈ 71 vs 53 for the baseline, derived). The
evaluation machine's thermal outcome is unknown and cannot be certified from
here; audit values are authoritative.

End-to-end pipeline on the development machine: ≈ 25 s per English answer,
≈ 57 s per Swahili answer (two NLLB passes included); byte-identical answers
across repeated identical invocations under greedy decoding.

Reproducibility: greedy decoding makes repeated runs on one machine/build
bit-identical; thread count derives from CPU affinity (physical cores, cap
8), removing a common drift source; retrieval is deterministic given the
committed indexes. Reproduce with `bash download_model.sh` then
`adtc-profiler run --submission . --mode participant --output submission.json`.

## Limitations

- Reference-machine thermal margin is unverified; both available test
  machines tripped the threshold under sustained all-core load. The 4-thread
  configuration matching the evaluation profile reduces heat substantially,
  but headroom should be confirmed where hardware access allows.
- When a question names a method the manual treats elsewhere (e.g., sealing a
  plastic container, with silo-sealing steps retrieved as strong context),
  the generator can merge procedures despite its system-prompt prohibition;
  answers stay grounded in manual text but may mix methods.
- Numeric-direction errors (a threshold stated as "maintain X" instead of
  "avoid X") were observed during development and mitigated via explicit
  system-prompt instructions; residual risk remains on unseen phrasings.
- n=50 audit metrics carry ±0.07 confidence bands; single runs are not
  precise model comparisons.

These are self-reported development benchmarks. Official scores are measured
by the ADTC profiler on the standard evaluation machine.
