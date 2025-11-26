import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn import functional as dist_nn


_TP_STATE: Optional["TensorParallelState"] = None


@dataclass
class TensorParallelState:
    tp_size: int
    tp_rank: int
    dp_size: int
    dp_rank: int
    tp_group: dist.ProcessGroup


def init_tensor_parallel(tp_size: int) -> Optional[TensorParallelState]:
    """
    Initialize tensor parallel process groups. Returns None when tp_size == 1.
    """
    global _TP_STATE

    if tp_size is None or tp_size <= 1:
        _TP_STATE = None
        return None

    if not dist.is_initialized():
        raise RuntimeError("Distributed process group must be initialized before tensor parallel setup.")

    world_size = dist.get_world_size()
    if world_size % tp_size != 0:
        raise ValueError(f"Tensor parallel size {tp_size} must divide world size {world_size}.")

    rank = dist.get_rank()
    dp_size = world_size // tp_size
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size

    tp_ranks = [dp_rank * tp_size + i for i in range(tp_size)]
    tp_group = dist.new_group(ranks=tp_ranks)

    _TP_STATE = TensorParallelState(tp_size=tp_size, tp_rank=tp_rank, dp_size=dp_size, dp_rank=dp_rank, tp_group=tp_group)
    return _TP_STATE


def get_tensor_parallel_state() -> Optional[TensorParallelState]:
    return _TP_STATE


def gather_from_tensor_parallel_region(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    state = get_tensor_parallel_state()
    if state is None:
        return tensor

    gather_list: List[torch.Tensor] = [torch.empty_like(tensor) for _ in range(state.tp_size)]
    dist_nn.all_gather(gather_list, tensor, group=state.tp_group)
    return torch.cat(gather_list, dim=dim)


def reduce_from_tensor_parallel_region(tensor: torch.Tensor) -> torch.Tensor:
    state = get_tensor_parallel_state()
    if state is None:
        return tensor

    dist.all_reduce(tensor, group=state.tp_group)
    return tensor


class ColumnParallelLinear(nn.Module):
    """
    Column parallel linear layer that shards weights across the output dimension.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gather_output: bool = True, tp_state: Optional[TensorParallelState] = None):
        super().__init__()
        self.tp_state = tp_state or get_tensor_parallel_state()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.is_parallel = self.tp_state is not None

        if not self.is_parallel:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            return

        if out_features % self.tp_state.tp_size != 0:
            raise ValueError("out_features must be divisible by tensor parallel size.")

        self.output_size_per_partition = out_features // self.tp_state.tp_size
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        if not self.is_parallel:
            return
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.is_parallel:
            return self.linear(input)

        output_parallel = F.linear(input, self.weight, self.bias)
        if self.gather_output:
            return gather_from_tensor_parallel_region(output_parallel, dim=-1)
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Row parallel linear layer that shards weights across the input dimension.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, input_is_parallel: bool = False, tp_state: Optional[TensorParallelState] = None):
        super().__init__()
        self.tp_state = tp_state or get_tensor_parallel_state()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.is_parallel = self.tp_state is not None

        if not self.is_parallel:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            return

        if in_features % self.tp_state.tp_size != 0:
            raise ValueError("in_features must be divisible by tensor parallel size.")

        self.input_size_per_partition = in_features // self.tp_state.tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        if not self.is_parallel:
            return
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.is_parallel:
            return self.linear(input)

        if self.input_is_parallel:
            input_parallel = input
        else:
            if input.size(-1) % self.tp_state.tp_size != 0:
                raise ValueError("Input feature dimension must be divisible by tensor parallel size.")
            chunks = torch.chunk(input, self.tp_state.tp_size, dim=-1)
            input_parallel = chunks[self.tp_state.tp_rank].contiguous()

        output_parallel = F.linear(input_parallel, self.weight, None)
        output = reduce_from_tensor_parallel_region(output_parallel)

        if self.bias is not None:
            output = output + self.bias

        return output

