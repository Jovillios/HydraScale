import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.optim as optim
from datasets import load_dataset
from argparse import ArgumentParser
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
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=2)
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")

    # --- Training Configuration ---
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10)

    return parser.parse_args()


def train_step(model, optimizer, batch):
    input_ids = batch["input_ids"]
    targets = batch["targets"]

    optimizer.zero_grad()
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_loop(model, dataloader, optimizer, num_steps):
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        loss = train_step(model, optimizer, batch)
        if rank == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {loss:.4f}")
    


if __name__ == "__main__":
    acc = torch.accelerator.current_accelerator()

    backend = "gloo"
    if acc is not None:
        backend = dist.get_default_backend_for_device(acc)

    # setup distributed process group
    dist.init_process_group(backend=backend)

    # torchrun provides RANK and WORLD_SIZE env variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device_id = rank % torch.accelerator.device_count()

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
    )

    model = HydraGPT(config=model_config)

    # if cuda is available move the model to GPU with id rank
    if str(acc) == "cuda":
        model.to(device_id)

    if str(acc) == "cuda":
        ddp_model = DDP(model, device_ids=[device_id])
    else:
        ddp_model = DDP(model)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)    

    print(f"Rank {rank}: Starting training loop for {num_steps} steps...")
    train_loop(ddp_model, dataloader, optimizer, num_steps)

    print(f"Rank {rank}: Training loop completed.")

    # clean up distributed process group
    dist.destroy_process_group()
