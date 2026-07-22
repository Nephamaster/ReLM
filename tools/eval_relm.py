"""Evaluate a ReLM checkpoint on LEMON/SIGHAN fixed-length pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForMaskedLM

try:
    from relm_data import DataFormatError, encode_tokens, parse_spaced_pair
except ImportError:  # Supports `python -m tools.eval_relm`.
    from tools.relm_data import DataFormatError, encode_tokens, parse_spaced_pair


DEFAULT_CATEGORIES = ["gam", "enc", "cot", "mec", "car", "nov", "new", "sig"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--rsighan_dir", default=None, help="Directory containing rSIGHAN13/14/15 JSONL files.")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def metrics(srcs: list[list[str]], tgts: list[list[str]], preds: list[list[str]]) -> dict[str, float | int]:
    positives = sum(src != tgt for src, tgt in zip(srcs, tgts))
    negatives = sum(src == tgt for src, tgt in zip(srcs, tgts))
    true_positive = sum(src != tgt and pred == tgt for src, tgt, pred in zip(srcs, tgts, preds))
    predicted_positive = sum(src != pred for src, pred in zip(srcs, preds))
    false_positive = sum(src == tgt and pred != tgt for src, tgt, pred in zip(srcs, tgts, preds))
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = false_positive / negatives if negatives else 0.0
    return {
        "samples": len(srcs),
        "positive_samples": positives,
        "negative_samples": negatives,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "fpr": fpr * 100,
    }


def build_records(pairs, tokenizer, max_seq_length: int):
    slots = (max_seq_length - 3) // 2
    cls_id, sep_id, mask_id, pad_id = (
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    )
    records: list[tuple[list[str], list[str], list[int], int, int]] = []
    dropped = {"format": 0, "length_mismatch": 0, "token_mismatch": 0, "seq_length": 0}
    for src_tokens, tgt_tokens in pairs:
        if len(src_tokens) != len(tgt_tokens):
            dropped["length_mismatch"] += 1
            continue
        src_ids = encode_tokens(tokenizer, src_tokens)
        tgt_ids = encode_tokens(tokenizer, tgt_tokens)
        if len(src_ids) != len(tgt_ids):
            dropped["token_mismatch"] += 1
            continue
        if len(src_ids) > slots:
            dropped["seq_length"] += 1
            continue
        input_ids = [cls_id] + src_ids + [sep_id] + [mask_id] * len(tgt_ids) + [sep_id]
        input_ids += [pad_id] * (max_seq_length - len(input_ids))
        records.append((src_tokens, tgt_tokens, input_ids, len(src_ids), len(tgt_ids)))
    return records, dropped


def predict_records(model, tokenizer, records, max_seq_length: int, batch_size: int, device, desc: str):
    pad_id = tokenizer.pad_token_id
    srcs, tgts, preds = [], [], []
    model.eval()
    for start in tqdm(range(0, len(records), batch_size), desc=desc, unit="batch"):
        chunk = records[start : start + batch_size]
        input_ids = torch.tensor([item[2] for item in chunk], dtype=torch.long, device=device)
        attention_mask = input_ids.ne(pad_id).long()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for row, (src_tokens, tgt_tokens, _, source_length, target_length) in zip(logits, chunk):
            target_start = source_length + 2
            predicted_ids = row[target_start : target_start + target_length].argmax(dim=-1).tolist()
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)
            srcs.append(src_tokens)
            tgts.append(tgt_tokens)
            preds.append(predicted_tokens)
    return metrics(srcs, tgts, preds)


def evaluate_file(model, tokenizer, path: Path, max_seq_length: int, batch_size: int, device):
    pairs = []
    dropped = {"format": 0}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                pairs.append(parse_spaced_pair(line, line_number))
            except DataFormatError:
                dropped["format"] += 1
    records, filtered = build_records(pairs, tokenizer, max_seq_length)
    dropped.update(filtered)
    return predict_records(model, tokenizer, records, max_seq_length, batch_size, device, path.stem), dropped


def evaluate_sighan(model, tokenizer, root: Path, max_seq_length: int, batch_size: int, device):
    pairs = []
    dropped = {"format": 0}
    files = [root / f"rSIGHAN{year}.jsonl" for year in (13, 14, 15)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing SIGHAN file: {path}")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    pairs.append((list(record["source"]), list(record["target"])))
                except (json.JSONDecodeError, KeyError, TypeError):
                    dropped["format"] += 1
    records, filtered = build_records(pairs, tokenizer, max_seq_length)
    dropped.update(filtered)
    return predict_records(model, tokenizer, records, max_seq_length, batch_size, device, "SIG"), dropped


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = BertForMaskedLM.from_pretrained(args.model_path).to(device)
    categories = [category.strip().lower() for category in args.categories.split(",") if category.strip()]
    root = Path(args.test_dir)
    rsighan_root = Path(args.rsighan_dir) if args.rsighan_dir else root.parent / "rsighan"
    missing = [
        category
        for category in categories
        if category != "sig" and not (root / f"{category}.txt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing evaluation files under {root}: "
            + ", ".join(f"{category}.txt" for category in missing)
        )
    results = {}
    for category in categories:
        if category == "sig" and not (root / "sig.txt").is_file():
            result, dropped = evaluate_sighan(
                model, tokenizer, rsighan_root, args.max_seq_length, args.batch_size, device
            )
        else:
            result, dropped = evaluate_file(
                model, tokenizer, root / f"{category}.txt", args.max_seq_length, args.batch_size, device
            )
        result["dropped"] = dropped
        results[category.upper()] = result

    valid = [item for item in results.values() if item["samples"]]
    average = {
        key: sum(item[key] for item in valid) / len(valid)
        for key in ["precision", "recall", "f1", "fpr"]
    } if valid else {key: 0.0 for key in ["precision", "recall", "f1", "fpr"]}
    payload = {
        "model_path": str(Path(args.model_path).resolve()),
        "test_dir": str(root.resolve()),
        "rsighan_dir": str(rsighan_root.resolve()) if "sig" in categories else None,
        "max_seq_length": args.max_seq_length,
        "categories": categories,
        "results": results,
        "average": average,
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
