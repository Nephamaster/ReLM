"""Audit formal ReLM JSONL pairs before launching a long training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

try:
    from relm_data import (
        DataFormatError,
        FeatureStats,
        _read_json_pair,
        make_feature,
        resolve_data_files,
    )
except ImportError:  # Supports `python -m tools.audit_relm_data`.
    from tools.relm_data import (
        DataFormatError,
        FeatureStats,
        _read_json_pair,
        make_feature,
        resolve_data_files,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_glob", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--allow_example", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = resolve_data_files(args.data_glob, allow_example=args.allow_example)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    stats = FeatureStats()
    invalid_examples: list[str] = []
    total = 0

    for filename in files:
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(
                tqdm(handle, desc=Path(filename).name, unit="lines"), start=1
            ):
                total += 1
                try:
                    src, tgt = _read_json_pair(line.strip(), line_number)
                except DataFormatError as exc:
                    stats.invalid_records += 1
                    if len(invalid_examples) < 10:
                        invalid_examples.append(str(exc))
                    continue
                make_feature(src, tgt, tokenizer, args.max_seq_length, stats)

    report = {
        "files": files,
        "total_records": total,
        "max_seq_length": args.max_seq_length,
        "stats": stats.as_dict(),
        "invalid_examples": invalid_examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_file:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
