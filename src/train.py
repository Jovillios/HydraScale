import os
import sys

from argparse import ArgumentParser
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity, record_function

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig

from model import HydraGPT
from dataloader import HydraDataLoader


def parse_args():
    parser = ArgumentParser("Training script for GPT model")

    # --- Model Configuration ---
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--intermediate_size", type=int, default=8192)

    # --- Data Configuration ---
    parser.add_argument("--dataset", type=str, default="ProCreations/Ultra-FineWeb-EDU")
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=2)
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")

    # --- Training Configuration ---
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)

    return parser.parse_args()


def trace_handler(p, device, rank):
    output = p.key_averages().table(row_limit=10)
    trace_dir = "/tmp/hydrascale_traces"
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_rank{rank}_step{p.step_num}.json")
    p.export_chrome_trace(trace_path)
    if rank == 0:
        print(f"Exported trace to {trace_path}")


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")


def profiler_setup():
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)

    # Configure profiler with explicit CUDA tracking
    return profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,  # Enable FLOPs counting
    )


class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, save_every: int, profiler=None) -> None:
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(model, device_ids=[self.gpu_id])
        self.profiler = profiler

    def _run_step(self, input_ids, targets):
        self.optimizer.zero_grad()
        _, loss = model(input_ids, targets=targets)
        loss.backward()
        self.optimizer.step()
        if self.profiler:
            torch.cuda.synchronize()
            self.profiler.step()

    def _run_epoch(self, epoch):
        bs = len(next(iter(self.train_data))[0])
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Batchsize: {bs} | Steps: {len(self.train_data)}")
        for input_ids, targets in self.train_data:
            input_ids = input_ids.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_step(input_ids, targets)

    def _save_checkpoint(self, epoch):
        ckpt = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckpt, PATH)
        print(f"Epoch {epoch}: Training checkpoint at {PATH}")

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


if __name__ == "__main__":
    # --- 1. HARDWARE VALIDATION ---
    if not torch.cuda.is_available():
        print("ERROR: This training script requires a CUDA-enabled GPU.", file=sys.stderr)
        print("       Please run on a machine with an NVIDIA GPU.", file=sys.stderr)
        sys.exit(1)  # Exit the script with an error code

    ddp_setup()

    args = parse_args()
    num_steps = args.num_steps
    batch_size = args.batch_size

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.hidden_size = args.hidden_size
    model_config.intermediate_size = args.intermediate_size
    model_config.max_position_embeddings = args.seq_len

    dataloader = HydraDataLoader(
        seq_len=args.seq_len,
        batch_size=batch_size,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer_name,
        num_workers=args.num_workers,
        subset=args.subset,
        split=args.split,
        num_proc=args.num_proc,
        rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

    model = HydraGPT(config=model_config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    if args.profiler:
        with profiler_setup() as profiler:
            trainer = Trainer(model, dataloader, optimizer, args.save_every, profiler)
    else:
        trainer = Trainer(model, dataloader, optimizer, args.save_every)

    trainer.train(args.num_epochs)

    # clean up distributed process group
    dist.destroy_process_group()
