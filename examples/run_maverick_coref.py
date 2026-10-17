#!/usr/bin/env python3
"""Run Maverick coreference on a small UTF-8 text file.

This is a POC helper, not a full-book windowing/merge pipeline. Start with the
included short sample before trying longer chapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("booknlp/data/english/pride.short.txt")
DEFAULT_OUTPUT = Path("outputs/maverick-pride-short.json")
DEFAULT_MODEL = "sapienzanlp/maverick-mes-litbank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="UTF-8 plain-text input")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face Maverick model ID or local path")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Maverick documents CPU/CUDA; MPS is not assumed to be supported",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input file does not exist: {args.input}", file=sys.stderr)
        return 2

    try:
        from maverick import Maverick
    except ImportError as exc:
        print(
            "Maverick is not installed. Run uv sync --extra coref with Python 3.12.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    text = args.input.read_text(encoding="utf-8")
    if not text.strip():
        print(f"Input file is empty: {args.input}", file=sys.stderr)
        return 2

    model = Maverick(hf_name_or_path=args.model, device=args.device)
    prediction: dict[str, Any] = model.predict(text)

    payload = {
        "model": args.model,
        "device": args.device,
        "input": str(args.input),
        "tokens": prediction.get("tokens"),
        "clusters_token_offsets": prediction.get("clusters_token_offsets"),
        "clusters_char_offsets": prediction.get("clusters_char_offsets"),
        "clusters_text_mentions": prediction.get("clusters_text_mentions"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Maverick predictions to {args.output}")
    print(f"Clusters: {len(payload['clusters_text_mentions'] or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
