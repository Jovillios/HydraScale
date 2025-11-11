import torch
import torch.distributed as dist
import os


def main():
    """Main function to run in each process."""

    # torchrun will set these environment variables for us.
    dist.init_process_group(backend="gloo")
    # 'gloo' is the backend for CPU. We use it for this simple test.
    # On a real multi-GPU server, you would use "nccl".

    # Get the rank (ID) of the current process and the total number of processes.
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Hello from the HydraScale environment! I am Rank {rank} of {world_size}.")

    # Clean up the process group.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
