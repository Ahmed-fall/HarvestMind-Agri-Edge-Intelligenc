#!/usr/bin/env python3
"""
CLI entry point. This is what judges run directly:

    python main.py "According to the FAO manual, what are the exact steps
    to seal a plastic storage container to prevent insect infestations?"

Output is the answer only -- clean and readable, no logging noise, per the plan's
own requirement. Set HARVESTMIND_VERBOSE=1 to see retrieval/RSS diagnostics on
stderr (stdout stays clean either way, so this is safe to leave set during your
own testing without breaking judge-facing output).
"""
import os
import sys

from pipeline import answer_query


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<question>\"", file=sys.stderr)
        return 1

    query = " ".join(sys.argv[1:])
    verbose = os.environ.get("HARVESTMIND_VERBOSE", "0") == "1"

    try:
        answer = answer_query(query, verbose=verbose)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())