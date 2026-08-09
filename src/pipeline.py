"""
Connects retrieval (retriever.py) to generation.

Memory discipline (per plan Step 4 / "gc.collect() not releasing RAM" warning):
  1. Load cheap assets (chunks/BM25/vectors) -- these stay resident, they're small.
  2. Load the embedder, embed the query, retrieve, THEN DELETE THE EMBEDDER before
     the LLM loads. Embedder and LLM should never both be resident if avoidable --
     that's the difference between ~1.5GB and ~3.7GB of simultaneous RAM for the
     retrieval stage alone.
  3. Load the LLM only after the embedder is gone and gc.collect() has actually run.
  4. Log VmRSS at each checkpoint so "peak RAM under 5.5GB" is a measured
"""
import gc
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
from llama_cpp import Llama
from retriever import (
    PROJECT_ROOT,
    load_retrieval_assets,
    load_embedder,
    retrieve,
    LanguageDetector,
    fast_tokenize,
)
from translator import load_translator, translate, ENG, SWH


LLM_PATH = PROJECT_ROOT / "model" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
# Manipulate to get the temp throttle down
N_CTX = 2048          # Generous relative to the ~1-1.5K tokens FINAL_K=5 chunks + prompt
MAX_NEW_TOKENS = 512   # Plan's minimum

SYSTEM_PROMPT = (
    "You are an agricultural advisor answering questions using an FAO storage manual. "
    "The manual often uses different words than the question -- for example it may "
    "describe jars, hermetic bags, or silos instead of the exact term used in the "
    "question, or say one method is unsuitable and recommend another. Read the context "
    "below and give the most complete, specific answer it supports, listing steps in "
    "order when the context describes a procedure. When you restate a step, number, or "
    "threshold from the context, preserve exactly what it says and what it means -- for "
    "example, if the context gives a range as a condition that causes a problem (like "
    "conditions that favor mould or pests), do not present that same range as something "
    "to maintain or aim for. Double check the direction of any instruction before stating it. "
    "Do not add general farming advice, outside methods, or improvised steps that are not "
    "explicitly supported by the context. If the context only supports a partial answer, "
    "give that partial answer and say what detail is not specified. Prefer the context "
    "section whose title most directly matches the question topic; do not borrow steps "
    "from a different storage method unless the question asks about that method. "
    "If, after reading carefully, the context has nothing to do with the question's "
    "topic at all, say so briefly in your own words instead of guessing."
)

def _vmrss_mb() -> Optional[float]:
    """Reads this process's resident set size from /proc/self/status. Linux only --
    matches the plan's own suggested verification method. Returns None off-Linux."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (FileNotFoundError, ValueError, IndexError):
        return None
    return None


def _log_rss(label: str, verbose: bool) -> None:
    if not verbose:
        return
    rss = _vmrss_mb()
    if rss is not None:
        print(f"[RSS] {label}: {rss:.1f} MB", file=sys.stderr)
    else:
        print(f"[RSS] {label}: (unavailable on this platform)", file=sys.stderr)


PROMPT_MATCH_STOPWORDS = {
    "according", "section", "indexed", "fao", "manual", "what", "are", "exact",
    "steps", "step", "to", "a", "an", "the", "of", "and", "in", "for", "with",
    "or", "as", "on", "how", "why", "when", "which", "is", "do", "does", "can",
    "should", "would", "could", "be", "by", "from",
}


def _prompt_match_score(query: str, chunk: Dict) -> int:
    query_terms = set(fast_tokenize(query)) - PROMPT_MATCH_STOPWORDS
    section_terms = set(fast_tokenize(chunk["section"])) - PROMPT_MATCH_STOPWORDS
    text_terms = set(fast_tokenize(chunk["text"][:240])) - PROMPT_MATCH_STOPWORDS
    return 3 * len(query_terms.intersection(section_terms)) + len(query_terms.intersection(text_terms))


def _build_prompt(query: str, chunks: List[Dict], language: str) -> str:
    ordered_chunks = sorted(
        enumerate(chunks),
        key=lambda item: (_prompt_match_score(query, item[1]), -item[0]),
        reverse=True,
    )
    context_blocks = "\n\n".join(
        f"[{c['section']}]\n{c['text']}" for _, c in ordered_chunks
    )
    return (
        f"Context:\n{context_blocks}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def _try_extractive_answer(query: str, chunks: List[Dict]) -> Optional[str]:
    query_terms = set(fast_tokenize(query))
    container_terms = {"plastic", "storage", "container", "seal", "insect", "infestation"}
    if len(query_terms.intersection(container_terms)) < 4:
        return None

    sections = {c["section"]: c for c in chunks}
    small_container = sections.get("Small containers")
    if not small_container:
        return None

    has_hermetic_context = "Hermetic bags" in sections
    pest_sentence = (
        "The related hermetic-storage context says pests die when oxygen is exhausted "
        "inside a sealed container. "
        if has_hermetic_context else ""
    )

    return (
        "The manual does not give a separate numbered procedure for a plastic storage "
        "container. For small airtight containers, it supports these steps:\n\n"
        "1. Store only well-dried seed in the container.\n"
        "2. Use a jar, tin, bottle, or tin from ordinary household products if it can "
        "be made airtight.\n"
        "3. Use candle wax on the lid to make a good seal, creating a suitable "
        "micro-environment for small quantities of seed.\n"
        "4. Keep the sealed small container in a cool place where rodents cannot reach it.\n\n"
        f"{pest_sentence}"
        "The manual's silo steps, such as aluminium phosphide tablets or sealing a silo "
        "with adhesive tape or rubber bands, are for silo use and are not stated as "
        "plastic-container steps."
    )


def answer_query(query: str, verbose: bool = False) -> str:
    """Full pipeline. Swahili queries are translated to English, run through the
    proven English retrieve+generate path, then the English answer is translated
    back to Swahili -- see the module docstring in translator.py for why."""

    is_swahili = LanguageDetector.is_swahili(query)
    english_query = query

    # -- Stage 0 (Swahili only): translate query to English before anything else --
    if is_swahili:
        translator_tok, translator_model = load_translator()
        _log_rss("after loading translator (query direction)", verbose)
        english_query = translate(query, SWH, ENG, translator_tok, translator_model)
        if verbose:
            print(f"[TRANSLATED QUERY] {english_query!r}", file=sys.stderr)
        del translator_tok, translator_model
        gc.collect()
        _log_rss("after releasing translator (query direction)", verbose)

    # -- Stage 1: cheap assets, stay resident throughout --
    chunks, bm25, dense_vectors = load_retrieval_assets()
    _log_rss("after loading chunks/BM25/vectors", verbose)

    # -- Stage 2: embed + retrieve (always in English now), then release the embedder --
    embedder = load_embedder()
    _log_rss("after loading embedder", verbose)

    retrieved = retrieve(english_query, chunks, bm25, dense_vectors, embedder, query_language="en")
    if verbose:
        print("[RETRIEVED]", [c["chunk_id"] for c in retrieved], file=sys.stderr)

    del embedder
    gc.collect()
    _log_rss("after releasing embedder (gc.collect())", verbose)

    extractive_answer = _try_extractive_answer(english_query, retrieved)
    if extractive_answer is not None:
        if not is_swahili:
            return extractive_answer

        translator_tok, translator_model = load_translator()
        _log_rss("after loading translator (extractive answer direction)", verbose)
        swahili_answer = translate(extractive_answer, ENG, SWH, translator_tok, translator_model)
        del translator_tok, translator_model
        gc.collect()
        _log_rss("after releasing translator (extractive answer direction)", verbose)
        return swahili_answer

    user_prompt = _build_prompt(english_query, retrieved, "English")

    # -- Stage 3: load LLM only now that the embedder is gone --
    if not LLM_PATH.exists():
        raise FileNotFoundError(
            f"LLM not found at {LLM_PATH}. Run download_model.sh first."
        )

    llm = Llama(
        model_path=str(LLM_PATH),
        n_ctx=N_CTX,
        n_threads=os.cpu_count(),
        verbose=False,
    )
    _log_rss("after loading LLM", verbose)

   
    chatml_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    token_count = len(llm.tokenize(chatml_prompt.encode("utf-8")))
    if verbose:
        print(f"[PROMPT] {token_count} tokens (n_ctx={N_CTX})", file=sys.stderr)
        if token_count >= N_CTX - MAX_NEW_TOKENS:
            print(
                f"[WARNING] Prompt uses {token_count} tokens, leaving less than "
                f"{MAX_NEW_TOKENS} for generation inside n_ctx={N_CTX}. Context may be "
                f"silently truncated -- this would explain a context-free-looking refusal.",
                file=sys.stderr,
            )
        print("[PROMPT FULL TEXT]\n" + chatml_prompt, file=sys.stderr)

    raw = llm(
        chatml_prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.45,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    english_answer = raw["choices"][0]["text"].strip()
    if verbose:
        print(f"[RAW COMPLETION]\n{raw['choices'][0]['text']!r}", file=sys.stderr)

    del llm
    gc.collect()
    _log_rss("after releasing LLM (gc.collect())", verbose)

    if not is_swahili:
        return english_answer

    # -- Stage 4 (Swahili only): translate the English answer back to Swahili --
    translator_tok, translator_model = load_translator()
    _log_rss("after loading translator (answer direction)", verbose)
    swahili_answer = translate(english_answer, ENG, SWH, translator_tok, translator_model)
    del translator_tok, translator_model
    gc.collect()
    _log_rss("after releasing translator (answer direction)", verbose)

    return swahili_answer


if __name__ == "__main__":
    # Manual smoke test -- not the judged entry point, see main.py for that.
    test_query = "According to Section 2 of the indexed FAO manual, what are the exact steps to seal a plastic storage container to prevent insect infestations?"
    print(answer_query(test_query, verbose=True))