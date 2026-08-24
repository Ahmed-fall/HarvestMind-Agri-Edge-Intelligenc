#!/usr/bin/env python3
"""
CLI entry point. This is what judges run directly:

    python src/main.py "According to the FAO manual, what are the exact steps
    to seal a plastic storage container to prevent insect infestations?"

Output is the answer only -- clean and readable, no logging noise on stdout.
Set HARVESTMIND_VERBOSE=1 to see retrieval/RSS diagnostics on stderr.
"""
import os

# Offline + CPU-only hardening, applied before any third-party import can read
# these: models come from local disk, and no code path may see a GPU even on
# machines that have one.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python src/main.py \"<question>\"", file=sys.stderr)
        return 1

    query = " ".join(sys.argv[1:])
    verbose = os.environ.get("HARVESTMIND_VERBOSE", "0") == "1"

    try:
        from pipeline import answer_query
        answer = answer_query(query, verbose=verbose)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # answer_query already degrades internally wherever it can; reaching
        # this line means even retrieval assets were unavailable. Nothing can
        # be salvaged -- fail loudly on stderr, cleanly on stdout.
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
