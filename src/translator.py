"""
Swahili <-> English translation via NLLB-200-distilled-600M.

Used as a fallback around the proven-reliable English retrieval+generation path,
rather than generating Swahili directly with the LLM -- see pipeline.py for why:
Qwen2.5-3B-Instruct-Q4_K_M's direct Swahili generation was measured to degenerate
into repetitive, semantically broken output even with repeat_penalty tuned, across
multiple runs. NLLB is a dedicated translation model and does not share that weakness.
"""
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retriever import PROJECT_ROOT

NLLB_PATH = PROJECT_ROOT / "model" / "translation" / "nllb-200-distilled-600M"

# FLORES-200 language codes
ENG = "eng_Latn"
SWH = "swh_Latn"


def load_translator() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    if not NLLB_PATH.exists():
        raise FileNotFoundError(
            f"NLLB model not found at {NLLB_PATH}. Run: WITH_NLLB=1 ./download_model.sh"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(NLLB_PATH))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(NLLB_PATH), torch_dtype=torch.float16)
    return tokenizer, model


def translate(text: str, src_lang: str, tgt_lang: str, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM) -> str:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # CHANGED: convert_tokens_to_ids is the version-robust way to get the target
    # language's forced BOS token -- tokenizer.lang_code_to_id was removed in some
    # newer transformers releases, so don't rely on it existing.
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,
        num_beams=4,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()