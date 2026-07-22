from __future__ import annotations

import copy
import glob
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class DataFormatError(ValueError):
    """Raised when a formal input record cannot be interpreted as a pair."""


def resolve_data_files(pattern: str, allow_example: bool = False) -> list[str]:
    files = sorted(path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path))
    if not files and os.path.isfile(pattern):
        files = [pattern]
    if not files:
        raise FileNotFoundError(f"No data files matched --data_glob: {pattern}")
    example_files = [path for path in files if ".example." in os.path.basename(path)]
    if example_files and not allow_example:
        raise ValueError(
            "Example files are not accepted as formal training input: "
            + ", ".join(example_files)
        )
    return files


def _read_json_pair(line: str, line_number: int) -> tuple[str, str]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DataFormatError(f"invalid JSON at line {line_number}: {exc}") from exc
    if not isinstance(record, dict) or "src" not in record or "tgt" not in record:
        raise DataFormatError(
            f"line {line_number} must contain JSON fields 'src' and 'tgt'"
        )
    src, tgt = record["src"], record["tgt"]
    if not isinstance(src, str) or not isinstance(tgt, str):
        raise DataFormatError(f"line {line_number} has non-string src/tgt fields")
    return src, tgt


class ConfusionGenerator:
    """The original one-error generator, retained only for optional mono mode."""

    def __init__(self, confus_dir: str, seed: int):
        root = Path(confus_dir)
        with open(root / "stroke.json", encoding="utf-8") as handle:
            self.stroke = json.load(handle)
        with open(root / "pinyin_sim.json", encoding="utf-8") as handle:
            self.pinyin_sim = json.load(handle)
        with open(root / "pinyin_sam.json", encoding="utf-8") as handle:
            self.pinyin_sam = json.load(handle)
        with open(root / "word_freq.txt", encoding="utf-8") as handle:
            self.word_list = [line.strip() for line in handle if line.strip()]
        self.rng = random.Random(seed)

    @staticmethod
    def is_han(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"

    def make_pair(self, sentence: str) -> tuple[str, str]:
        source = list(sentence)
        target = list(sentence)
        positions = [i for i, char in enumerate(source) if self.is_han(char)]
        if not positions:
            return sentence, sentence
        index = self.rng.choice(positions)
        roll = self.rng.random()
        if roll < 0.4:
            candidates = self.pinyin_sam.get(source[index], self.word_list)
        elif roll < 0.7:
            candidates = self.pinyin_sim.get(source[index], self.word_list)
        elif roll < 0.9:
            candidates = self.stroke.get(source[index], self.word_list)
        else:
            candidates = self.word_list
        if not candidates:
            candidates = self.word_list
        source[index] = self.rng.choice(candidates)
        return "".join(source), "".join(target)


def iter_raw_pairs(
    files: Sequence[str],
    data_mode: str,
    confus_dir: str | None,
    seed: int,
    shard_id: int = 0,
    shard_count: int = 1,
) -> Iterator[tuple[int, str, str]]:
    generator = ConfusionGenerator(confus_dir, seed + shard_id) if data_mode == "mono" else None
    raw_index = 0
    for filename in files:
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                current_index = raw_index
                raw_index += 1
                if current_index % shard_count != shard_id:
                    continue
                stripped = line.strip()
                if not stripped:
                    yield current_index, "", ""
                    continue
                try:
                    if data_mode == "pair":
                        src, tgt = _read_json_pair(stripped, line_number)
                    elif data_mode == "mono":
                        if generator is None:
                            raise AssertionError("mono mode requires a confusion generator")
                        src, tgt = generator.make_pair(stripped)
                    else:
                        raise ValueError(f"unsupported data mode: {data_mode}")
                except DataFormatError:
                    yield current_index, None, None
                    continue
                yield current_index, src, tgt


@dataclass
class FeatureStats:
    accepted: int = 0
    invalid_records: int = 0
    empty_records: int = 0
    length_mismatch: int = 0
    token_mismatch: int = 0
    too_long: int = 0
    unknown_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "accepted": self.accepted,
            "invalid_records": self.invalid_records,
            "empty_records": self.empty_records,
            "dropped_length_mismatch": self.length_mismatch,
            "dropped_token_mismatch": self.token_mismatch,
            "dropped_seq_length": self.too_long,
            "unknown_tokens": self.unknown_tokens,
            "total_tokens": self.total_tokens,
            "unk_rate": self.unknown_tokens / self.total_tokens if self.total_tokens else 0.0,
        }


def encode_text(tokenizer, text: str) -> list[int]:
    # bert-base-chinese is character based; passing a character list prevents
    # accidental word-level splitting while retaining tokenizer behavior.
    return encode_tokens(tokenizer, list(text))


def encode_tokens(tokenizer, pieces: Sequence[str]) -> list[int]:
    """Encode an already tokenized, whitespace-separated CSC sentence."""
    encoded = tokenizer(
        list(pieces),
        add_special_tokens=False,
        is_split_into_words=True,
        truncation=False,
    )
    return list(encoded["input_ids"])


def make_feature(
    src: str,
    tgt: str,
    tokenizer,
    max_seq_length: int,
    stats: FeatureStats | None = None,
):
    if stats is None:
        stats = FeatureStats()
    if not src or not tgt:
        stats.empty_records += 1
        return None, "empty"
    if len(src) != len(tgt):
        stats.length_mismatch += 1
        return None, "length_mismatch"

    src_ids = encode_text(tokenizer, src)
    tgt_ids = encode_text(tokenizer, tgt)
    stats.unknown_tokens += sum(
        token_id == tokenizer.unk_token_id for token_id in src_ids + tgt_ids
    )
    stats.total_tokens += len(src_ids) + len(tgt_ids)
    if len(src_ids) != len(tgt_ids):
        stats.token_mismatch += 1
        return None, "token_mismatch"

    slots = (max_seq_length - 3) // 2
    if len(src_ids) > slots:
        stats.too_long += 1
        return None, "seq_length"

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    if None in (cls_id, sep_id, mask_id, pad_id):
        raise ValueError("Tokenizer must define CLS, SEP, MASK and PAD token IDs")

    source_start = 1
    separator_index = source_start + len(src_ids)
    target_start = separator_index + 1
    sequence = (
        [cls_id]
        + src_ids
        + [sep_id]
        + [mask_id] * len(tgt_ids)
        + [sep_id]
    )
    labels = [-100] * len(sequence)
    labels[target_start : target_start + len(tgt_ids)] = tgt_ids
    source_mask = [False] * len(sequence)
    source_mask[source_start:separator_index] = [True] * len(src_ids)
    mft_candidates = [False] * len(sequence)
    mft_candidates[source_start:separator_index] = [
        source_id == target_id for source_id, target_id in zip(src_ids, tgt_ids)
    ]
    mft_targets = [-100] * len(sequence)
    mft_targets[source_start:separator_index] = tgt_ids
    attention_mask = [1] * len(sequence)

    padding = max_seq_length - len(sequence)
    sequence += [pad_id] * padding
    attention_mask += [0] * padding
    labels += [-100] * padding
    source_mask += [False] * padding
    mft_candidates += [False] * padding
    mft_targets += [-100] * padding
    stats.accepted += 1
    return (
        torch.tensor(sequence, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(source_mask, dtype=torch.bool),
        torch.tensor(mft_candidates, dtype=torch.bool),
        torch.tensor(mft_targets, dtype=torch.long),
    ), "accepted"


class ReLMJsonlDataset(IterableDataset):
    """Repeatable, sharded, bounded-memory stream of ReLM training features."""

    def __init__(
        self,
        files: Sequence[str],
        tokenizer,
        max_seq_length: int,
        data_mode: str = "pair",
        confus_dir: str | None = None,
        seed: int = 42,
        shuffle_buffer_size: int = 10000,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.files = list(files)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_mode = data_mode
        self.confus_dir = confus_dir
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rank = rank
        self.world_size = world_size
        self.stats = FeatureStats()

    def _iter_cycle(self, cycle: int) -> Iterator[tuple[torch.Tensor, ...]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        rank = self.rank
        world_size = self.world_size
        shard_count = world_size * worker_count
        shard_id = rank * worker_count + worker_id
        pairs = iter_raw_pairs(
            self.files,
            self.data_mode,
            self.confus_dir,
            self.seed + cycle * 1_000_003,
            shard_id=shard_id,
            shard_count=shard_count,
        )
        rng = np.random.default_rng(self.seed + cycle * 1_000_003 + shard_id)
        buffer: list[tuple[torch.Tensor, ...]] = []

        for _, src, tgt in pairs:
            if src is None or tgt is None:
                self.stats.invalid_records += 1
                continue
            feature, reason = make_feature(
                src, tgt, self.tokenizer, self.max_seq_length, self.stats
            )
            if feature is None:
                continue
            if self.shuffle_buffer_size <= 0:
                yield feature
                continue
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(feature)
                continue
            index = int(rng.integers(0, len(buffer)))
            yield buffer[index]
            buffer[index] = feature

        if buffer:
            rng.shuffle(buffer)
            yield from buffer

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        cycle = 0
        while True:
            yielded = False
            for feature in self._iter_cycle(cycle):
                yielded = True
                yield feature
            if not yielded:
                raise RuntimeError("The data stream produced no valid training examples")
            cycle += 1


def apply_mft(
    batch: dict[str, torch.Tensor],
    tokenizer,
    mask_mode: str,
    mask_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].clone()
    labels = batch["labels"].clone()
    source_mask = batch["source_mask"]
    equal_mask = batch["mft_candidates"]
    if mask_mode == "noerror":
        candidates = source_mask & equal_mask
    elif mask_mode == "error":
        candidates = source_mask & ~equal_mask
    elif mask_mode == "all":
        candidates = source_mask
    else:
        raise ValueError(f"unsupported mask mode: {mask_mode}")
    selected = candidates & (torch.rand_like(input_ids, dtype=torch.float32) < mask_rate)
    input_ids[selected] = tokenizer.mask_token_id
    labels[selected] = batch["mft_targets"][selected]
    return input_ids, labels, selected


def parse_spaced_pair(line: str, line_number: int) -> tuple[list[str], list[str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        raise DataFormatError(f"expected src<TAB>tgt at line {line_number}")
    src = parts[0].split() if " " in parts[0] else list(parts[0])
    tgt = parts[1].split() if " " in parts[1] else list(parts[1])
    return src, tgt
