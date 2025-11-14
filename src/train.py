import os
from argparse import ArgumentParser
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity, record_function

from datasets import load_dataset
from transformers import AutoConfig

from model import HydraGPT
from dataloader import HydraDataLoader


def parse_args():
    parser = ArgumentParser("Training script for GPT model")

    # --- Model Configuration ---
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=1024)

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

    return parser.parse_args()


def trace_handler(p, device, rank):
    output = p.key_averages().table(row_limit=10)
    trace_dir = "/tmp/hydrascale_traces"
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_rank{rank}_step{p.step_num}.json")
    p.export_chrome_trace(trace_path)
    if rank == 0:
        print(f"Exported trace to {trace_path}")


def train_step(model, optimizer, batch, device):
    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)

    optimizer.zero_grad()
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_loop(model, dataloader, optimizer, num_steps, device, rank, prof=None):
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        loss = train_step(model, optimizer, batch, device)
        if rank == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {loss:.4f}")

        if prof:
            prof.step()


if __name__ == "__main__":
    # setup distributed process group
    # torchrun provides RANK and WORLD_SIZE env variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Determine backend: use nccl for CUDA, gloo for CPU
    if torch.cuda.is_available():
        backend = "nccl"
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
    else:
        backend = "gloo"
        device = torch.device("cpu")
        device_id = None

    # Initialize distributed process group
    dist.init_process_group(backend=backend)

    # Print device information
    if rank == 0:
        print(f"Using backend: {backend}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"CUDA device name: {torch.cuda.get_device_name(device_id)}")
        else:
            print("CUDA not available, using CPU")

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
        world_size=world_size,
        rank=rank,
    )

    model = HydraGPT(config=model_config)

    # Move model to the appropriate device
    model = model.to(device)

    # Wrap model with DDP
    if torch.cuda.is_available() and device_id is not None:
        ddp_model = DDP(model, device_ids=[device_id], output_device=device_id)
    else:
        ddp_model = DDP(model)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)

    dist.barrier()

    if rank == 0:
        print(f"Starting training loop for {num_steps} steps...")
        print(f"Model device: {next(ddp_model.parameters()).device}")
        print(f"World size: {world_size}, Batch size per rank: {batch_size}")
    
    # Set epoch for DistributedSampler (important for proper data shuffling)
    dataloader.sampler.set_epoch(0)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_ready = partial(trace_handler, device=device, rank=rank)
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)

    with profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        train_loop(ddp_model, dataloader, optimizer, num_steps, device, rank, prof)

    print(f"Rank {rank}: Training loop completed.")

    # clean up distributed process group
    dist.destroy_process_group()
