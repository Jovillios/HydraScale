import os
import sys
import logging
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Any

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule
from torch.utils.data import DataLoader
from transformers import AutoConfig

# Local imports
from model import HydraGPT
from dataloader import HydraDataLoader

# --- Utilities ---


def get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def dist_print(msg: str, rank: int = 0):
    """Helper to print only on a specific rank (default 0)."""
    if get_rank() == rank:
        print(f"[Rank {get_rank()}] {msg}")


def setup_distributed_environment():
    """Validates hardware and initializes DDP."""
    if not torch.cuda.is_available():
        sys.stderr.write("ERROR: CUDA-enabled GPU required.\n")
        sys.exit(1)

    local_rank = get_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_distributed_environment():
    dist.destroy_process_group()


def get_profiler(args: Namespace):
    """Sets up the Torch profiler based on command line args."""
    if not args.profiler:
        return nullcontext()

    def trace_handler(p):
        rank = get_rank()
        trace_dir = Path("hydrascale_traces")
        trace_dir.mkdir(exist_ok=True)
        output = trace_dir / f"trace_rank{rank}_step{p.step_num}.json"
        p.export_chrome_trace(str(output))
        dist_print(f"Exported trace to {output}")

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=args.prof_wait, warmup=args.prof_warmup, active=args.prof_active, repeat=args.prof_repeat),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )


# --- Trainer Class ---


class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, save_every: int, profiler: Optional[Any] = None) -> None:
        self.local_rank = get_rank()
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.profiler = profiler

        # Move model to device and wrap in DDP
        self.model = model.to(self.local_rank)
        self.model = DDP(model, device_ids=[self.local_rank])

    def _run_step(self, input_ids: torch.Tensor, targets: torch.Tensor):
        self.optimizer.zero_grad()
        _, loss = self.model(input_ids, targets=targets)
        loss.backward()
        self.optimizer.step()

        # Step the profiler if it exists
        if self.profiler and not isinstance(self.profiler, nullcontext):
            torch.cuda.synchronize()
            self.profiler.step()

    def _save_checkpoint(self, epoch: int):
        if self.local_rank == 0:
            ckpt_path = Path(f"checkpoint_epoch{epoch}.pt")
            torch.save(self.model.module.state_dict(), ckpt_path)
            dist_print(f"Saved checkpoint: {ckpt_path}")

    def train(self, num_epochs: int):
        dist_print("Starting training...")

        for epoch in range(num_epochs):
            dist_print(f"Epoch {epoch} | Steps: {len(self.train_data)}")

            # Crucial for shuffling in DDP
            if hasattr(self.train_data, "sampler") and hasattr(self.train_data.sampler, "set_epoch"):
                self.train_data.sampler.set_epoch(epoch)

            for batch in self.train_data:
                input_ids = batch["input_ids"].to(self.local_rank)
                targets = batch["targets"].to(self.local_rank)
                self._run_step(input_ids, targets)

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


# --- Configuration ---


def parse_args() -> Namespace:
    parser = ArgumentParser(description="HydraGPT Distributed Training")

    # Model Group
    group_model = parser.add_argument_group("Model Configuration")
    group_model.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    group_model.add_argument("--num_hidden_layers", type=int, default=4)
    group_model.add_argument("--num_attention_heads", type=int, default=8)
    group_model.add_argument("--hidden_size", type=int, default=256)
    group_model.add_argument("--intermediate_size", type=int, default=1024)

    # Data Group
    group_data = parser.add_argument_group("Data Configuration")
    group_data.add_argument("--dataset", type=str, default="ProCreations/Ultra-FineWeb-EDU")
    group_data.add_argument("--subset", type=int, default=1000)
    group_data.add_argument("--split", type=str, default="train")
    group_data.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    group_data.add_argument("--num_workers", type=int, default=0)
    group_data.add_argument("--num_proc", type=int, default=2)

    # Training Group
    group_train = parser.add_argument_group("Training Configuration")
    group_train.add_argument("--seq_len", type=int, default=128)
    group_train.add_argument("--batch_size", type=int, default=4)
    group_train.add_argument("--num_epochs", type=int, default=1)
    group_train.add_argument("--save_every", type=int, default=1)
    group_train.add_argument("--lr", type=float, default=1e-3)

    # Profiler Group
    group_prof = parser.add_argument_group("Profiler Configuration")
    group_prof.add_argument("--profiler", action="store_true", help="Enable PyTorch Profiler")
    group_prof.add_argument("--prof_wait", type=int, default=10, help="Steps to wait before profiling")
    group_prof.add_argument("--prof_warmup", type=int, default=1, help="Warmup steps")
    group_prof.add_argument("--prof_active", type=int, default=3, help="Steps to actively record")
    group_prof.add_argument("--prof_repeat", type=int, default=1, help="Number of profiling cycles")

    return parser.parse_args()


# --- Main Execution ---


def main():
    setup_distributed_environment()
    args = parse_args()

    # Config Setup
    config = AutoConfig.from_pretrained(args.model_name)
    config.update(
        {
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "max_position_embeddings": args.seq_len,
        }
    )

    # Data Loading
    dataloader = HydraDataLoader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer_name,
        num_workers=args.num_workers,
        subset=args.subset,
        split=args.split,
        num_proc=args.num_proc,
        rank=get_rank(),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

    # Model & Optimizer
    model = HydraGPT(config=config)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Training with Dynamic Profiler
    with get_profiler(args) as profiler:
        trainer = Trainer(model=model, train_data=dataloader, optimizer=optimizer, save_every=args.save_every, profiler=profiler)
        trainer.train(args.num_epochs)

    cleanup_distributed_environment()


if __name__ == "__main__":
    main()
