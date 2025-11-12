import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT
import torch.optim as optim


def main():
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

    # Hyperparameters for a small test model
    vocab_size = 1024
    n_hidden = 512
    n_layers = 4
    num_heads = 8
    ffn_ratio = 4
    attn_dropout = 0.1
    ffn_dropout = 0.1
    max_seq_len = 256

    # Create the model instance
    model = GPT(vocab_size=vocab_size, n_hidden=n_hidden, n_layers=n_layers, num_heads=num_heads, ffn_ratio=ffn_ratio, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout, max_seq_len=max_seq_len)

    # if cuda is available move the model to GPU with id rank
    if str(acc) == "cuda":
        model.to(device_id)

    if str(acc) == "cuda":
        ddp_model = DDP(model, device_ids=[device_id])
    else:
        ddp_model = DDP(model)
    batch_size = 4
    # Create some dummy input data
    dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    dummy_targets = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    if str(acc) == "cuda":
        dummy_input = dummy_input.to(device_id)
        dummy_targets = dummy_targets.to(device_id)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    _, loss = ddp_model(dummy_input, targets=dummy_targets)

    print(f"Rank {rank}, Loss: {loss.item():.2f}")

    loss.backward()
    optimizer.step()

    print(f"Rank {rank} has completed a backward pass.")

    # clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
