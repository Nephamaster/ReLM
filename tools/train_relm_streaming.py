"""Train ReLM on the 34M JSONL pairs without loading the corpus in memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    SchedulerType,
    get_scheduler,
    __version__ as transformers_version,
)

try:
    from relm_data import ReLMJsonlDataset, apply_mft, resolve_data_files
except ImportError:  # Supports `python -m tools.train_relm_streaming`.
    from tools.relm_data import ReLMJsonlDataset, apply_mft, resolve_data_files


DEFAULT_DATA = "/share/project/wuhaiming/spaces/ReLM/data/34m_confuse_gen/34m_confuse_gen.jsonl"
DEFAULT_MODEL = "/share/project/wuhaiming/data/models/bert-base-chinese"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--data_glob", default=DEFAULT_DATA)
    parser.add_argument("--data_mode", choices=["pair", "mono"], default="pair")
    parser.add_argument("--confus_dir", default="confus")
    parser.add_argument("--output_dir", default="outputs/relm-34m-paper")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=60000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default=SchedulerType.CONSTANT)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--mask_mode", choices=["noerror", "error", "all"], default="noerror")
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--keep_last_checkpoints", type=int, default=3)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--allow_example", action="store_true")
    parser.add_argument("--expected_world_size", type=int, default=8)
    parser.add_argument(
        "--allow_nonpaper_world_size",
        action="store_true",
        help="Allow smoke tests with a world size other than the paper's 8 processes.",
    )
    parser.add_argument(
        "--allow_nonpaper_global_batch",
        action="store_true",
        help="Allow smoke tests whose global batch is not 4096.",
    )
    parser.add_argument(
        "--skip_hashes",
        action="store_true",
        help="Skip full SHA256 hashes of model and data files.",
    )
    return parser.parse_args()


def sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    root = Path(path)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for item in paths:
        digest.update(str(item.relative_to(root.parent if root.is_file() else root)).encode())
        with open(item, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def infer_step_from_resume(path: str | None) -> int:
    if not path:
        return 0
    match = re.search(r"checkpoint-(\d+)", str(Path(path).resolve()))
    if not match:
        raise ValueError("--resume_from must be inside a checkpoint-N directory")
    return int(match.group(1))


def model_metadata(args, files: list[str], accelerator: Accelerator, tokenizer) -> dict:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_path": str(Path(args.model_path).resolve()),
        "data_files": [str(Path(path).resolve()) for path in files],
        "data_mode": args.data_mode,
        "max_seq_length": args.max_seq_length,
        "mask_mode": args.mask_mode,
        "mask_rate": args.mask_rate,
        "prompt_enabled": False,
        "prompt_length": 0,
        "per_device_batch": args.per_device_train_batch_size,
        "world_size": accelerator.num_processes,
        "expected_world_size": args.expected_world_size,
        "gradient_accumulation": args.gradient_accumulation_steps,
        "global_batch": args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "scheduler": str(args.lr_scheduler_type),
        "warmup_ratio": args.warmup_ratio,
        "max_train_steps": args.max_train_steps,
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(accelerator.device),
        "local_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "accelerate_version": __import__("accelerate").__version__,
        "tokenizer_class": tokenizer.__class__.__name__,
    }
    if not args.skip_hashes and accelerator.is_main_process:
        metadata["model_hash"] = sha256_path(args.model_path)
        metadata["data_hash"] = sha256_path(files[0]) if len(files) == 1 else {
            path: sha256_path(path) for path in files
        }
    else:
        metadata["hashes_skipped"] = True
    return metadata


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def export_hf_model(accelerator: Accelerator, model, tokenizer, destination: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        unwrapped.save_pretrained(
            destination,
            state_dict=state_dict,
            save_function=accelerator.save,
            safe_serialization=False,
        )
        tokenizer.save_pretrained(destination)
    accelerator.wait_for_everyone()


def save_checkpoint(
    accelerator: Accelerator,
    model,
    tokenizer,
    optimizer,
    scheduler,
    output_dir: Path,
    step: int,
    metadata: dict,
    keep_last: int,
) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    state_dir = checkpoint / "accelerate_state"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(state_dir))
    accelerator.wait_for_everyone()
    export_hf_model(accelerator, model, tokenizer, checkpoint / "hf_model")
    if accelerator.is_main_process:
        save_json(checkpoint / "train_config.json", {**metadata, "checkpoint": step})
        checkpoints = sorted(
            (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        for old in checkpoints[:-keep_last]:
            shutil.rmtree(old)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mask_rate <= 1.0:
        raise ValueError("--mask_rate must be between 0 and 1")
    files = resolve_data_files(args.data_glob, allow_example=args.allow_example)
    if args.data_mode == "mono" and not args.confus_dir:
        raise ValueError("--confus_dir is required in mono mode")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
            split_batches=False,
        ),
    )
    set_seed(args.seed, device_specific=True)
    global_batch = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    if accelerator.num_processes != args.expected_world_size and not args.allow_nonpaper_world_size:
        raise ValueError(
            f"Paper protocol expects world size {args.expected_world_size}, "
            f"got {accelerator.num_processes}. Use --allow_nonpaper_world_size only for a smoke test."
        )
    if global_batch != 4096 and not args.allow_nonpaper_global_batch:
        raise ValueError(
            f"Paper protocol requires global batch 4096, got {global_batch}. "
            "Use --allow_nonpaper_global_batch only for a smoke test."
        )

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = BertForMaskedLM.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        raise ValueError("bert-base-chinese tokenizer must define pad_token_id")
    dataset = ReLMJsonlDataset(
        files,
        tokenizer,
        args.max_seq_length,
        data_mode=args.data_mode,
        confus_dir=args.confus_dir,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=args.learning_rate)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=round(args.max_train_steps * args.warmup_ratio),
        num_training_steps=args.max_train_steps,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    metadata = model_metadata(args, files, accelerator, tokenizer)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        save_json(output_dir / "train_config.json", metadata)
    accelerator.wait_for_everyone()

    start_step = infer_step_from_resume(args.resume_from)
    if start_step >= args.max_train_steps:
        raise ValueError("resume checkpoint is already at or beyond max_train_steps")
    if args.resume_from:
        accelerator.load_state(args.resume_from)

    train_iter = iter(loader)
    batches_to_skip = start_step * args.gradient_accumulation_steps
    for _ in range(batches_to_skip):
        next(train_iter)

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=args.max_train_steps,
        initial=start_step,
        disable=not accelerator.is_local_main_process,
        desc="ReLM training",
    )
    global_step = start_step
    running_loss = 0.0
    while global_step < args.max_train_steps:
        batch = next(train_iter)
        batch = {key: value.to(accelerator.device) for key, value in zip(
            ["input_ids", "attention_mask", "labels", "source_mask", "mft_candidates", "mft_targets"],
            batch,
        )}
        with accelerator.accumulate(model):
            input_ids, labels, _ = apply_mft(batch, tokenizer, args.mask_mode, args.mask_rate)
            outputs = model(input_ids=input_ids, attention_mask=batch["attention_mask"], labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            running_loss += loss.detach().float().item()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                if accelerator.is_local_main_process and global_step % 100 == 0:
                    progress.set_postfix(loss=f"{running_loss / 100:.4f}")
                    running_loss = 0.0
                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        output_dir,
                        global_step,
                        metadata,
                        args.keep_last_checkpoints,
                    )
    progress.close()

    if global_step % args.save_steps != 0:
        save_checkpoint(
            accelerator,
            model,
            tokenizer,
            optimizer,
            scheduler,
            output_dir,
            global_step,
            metadata,
            args.keep_last_checkpoints,
        )
    if accelerator.is_main_process:
        save_json(
            output_dir / "runtime_stream_stats_rank0.json",
            dataset.stats.as_dict(),
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
