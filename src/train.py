import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT
import torch.optim as optim
from argparse import ArgumentParser
from transformers import AutoConfig


def parse_args():
    parser = ArgumentParser("Training script for GPT model")

    # --- Model Configuration ---
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=1024)

    # -- Global Configuration
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--vocab_size", type=int, default=1024)

    return parser.parse_args()


def train(ddp_model, config, batch_size, num_steps):
    vocab_size = config.vocab_size
    max_seq_len = config.max_position_embeddings
    for step in range(num_steps):
        # Create some dummy input data
        dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))
        dummy_targets = torch.randint(0, vocab_size, (batch_size, max_seq_len))

        if str(acc) == "cuda":
            dummy_input = dummy_input.to(device_id)
            dummy_targets = dummy_targets.to(device_id)

        optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        _, loss = ddp_model(dummy_input, targets=dummy_targets)

        if rank == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()


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
    model_config.vocab_size = args.vocab_size
    model_config.max_position_embeddings = args.seq_len

    model = GPT(config=model_config)

    # if cuda is available move the model to GPU with id rank
    if str(acc) == "cuda":
        model.to(device_id)

    if str(acc) == "cuda":
        ddp_model = DDP(model, device_ids=[device_id])
    else:
        ddp_model = DDP(model)

    print(f"Rank {rank}: Starting training loop for {num_steps} steps...")
    train(ddp_model, model_config, batch_size, num_steps)

    print(f"Rank {rank}: Training loop completed.")

    # clean up distributed process group
    dist.destroy_process_group()
