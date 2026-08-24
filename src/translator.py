"""
Swahili <-> English translation via NLLB-200-distilled-600M.

Used as a wrapper around the proven English retrieval+generation path rather
than generating Swahili directly with the LLM: Qwen2.5's direct Swahili
generation degenerates into repetitive, semantically broken output even with
repeat_penalty tuned. NLLB is a dedicated translation model and does not share
that weakness.
"""
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retriever import PROJECT_ROOT

NLLB_PATH = PROJECT_ROOT / "model" / "translation" / "nllb-200-distilled-600M"

# FLORES-200 language codes
ENG = "eng_Latn"
SWH = "swh_Latn"

# NLLB-200 supports sequences up to 1024 tokens. The input cap must exceed the
# generator's longest possible English answer (MAX_NEW_TOKENS=512 new tokens on
# top of a prompt-derived answer), otherwise Swahili users silently lose the
# tail of long answers.
INPUT_MAX_TOKENS = 1024
OUTPUT_MAX_TOKENS = 1024


def load_translator() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    if not NLLB_PATH.exists():
        raise FileNotFoundError(
            f"NLLB model not found at {NLLB_PATH}. Run download_model.sh first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(NLLB_PATH))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(NLLB_PATH), torch_dtype=torch.float16)
    return tokenizer, model


def translate(text: str, src_lang: str, tgt_lang: str, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM) -> str:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=INPUT_MAX_TOKENS)

    if inputs["input_ids"].shape[1] >= INPUT_MAX_TOKENS:
        print(
            f"[TRANSLATE] Warning: input hit the {INPUT_MAX_TOKENS}-token cap; "
            "the tail of the text was truncated.",
            file=sys.stderr,
        )

    # convert_tokens_to_ids is the version-robust way to get the target
    # language's forced BOS token (tokenizer.lang_code_to_id was removed in
    # some newer transformers releases). Fail loudly rather than generating in
    # an unintended language if the token is missing.
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_token_id is None:
        raise RuntimeError(
            f"Target language code {tgt_lang!r} not found in the NLLB tokenizer vocabulary."
        )

    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=OUTPUT_MAX_TOKENS,
        num_beams=4,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
