import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_parallel import ColumnParallelLinear, RowParallelLinear, get_tensor_parallel_state

# --- Core building blocks of the Transformer ---


class FeedForward(nn.Module):
    """
    A gated two-layer feed-forward network with a non-linearity and dropout.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # sanity check

        # parameters
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        # modules

        # The network consists of an "up-projection" to a higher dimension,
        # followed by a "down-projection" back to the original dimension.
        tp_state = get_tensor_parallel_state()
        self.up_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size, gather_output=False, tp_state=tp_state)
        self.gate_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size, gather_output=False, tp_state=tp_state)
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, input_is_parallel=True, tp_state=tp_state)
        self.act_fn = getattr(F, self.hidden_act)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Apply layers in sequence : up-projection -> activation -> down-projection
        x_gate = F.silu(self.gate_proj(input))
        x_up = self.up_proj(input)
        x = self.down_proj(x_gate * x_up)
        return x


class SelfAttention(nn.Module):
    """
    Tensor-parallel friendly self-attention.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        tp_state = get_tensor_parallel_state()
        tp_size = tp_state.tp_size if tp_state is not None else 1
        if self.num_attention_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size.")

        self.num_heads_per_rank = self.num_attention_heads // tp_size
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.q_proj = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=False, gather_output=False, tp_state=tp_state)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=False, gather_output=False, tp_state=tp_state)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=False, gather_output=False, tp_state=tp_state)
        self.out_proj = RowParallelLinear(self.hidden_size, self.hidden_size, bias=False, input_is_parallel=True, tp_state=tp_state)
        self.dropout = nn.Dropout(self.attention_dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, _ = input.shape

        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        def reshape(x):
            return x.view(B, T, self.num_heads_per_rank, self.head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(attn_output)


class Block(nn.Module):
    """
    A single Transformer Block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # sanity check

        # parameters
        self.hidden_size = config.hidden_size

        # modules
        self.attn = SelfAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input + self.attn(self.ln1(input))
        x = x + self.ffn(self.ln2(x))
        return x


class HydraGPT(nn.Module):
    """
    A minimal Generative Pre-trained Transformer (GPT) model.
    """

    def __init__(self, config):
        super().__init__()
        # sanity check
        assert config.hidden_size % config.num_attention_heads == 0

        # parameters
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.num_hidden_layers = config.num_hidden_layers

        # modules
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_embed = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(self.num_hidden_layers):
            self.layers.append(Block(config))
        self.decoder_head = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input: torch.Tensor, targets=None):
        # input shape: (batch_size, seq_len)
        B, T = input.shape

        # Get token and pos embed
        tok_emb = self.token_embed(input)  # (B, T, C)
        pos_ids = torch.arange(T, device=input.device)
        pos_embed = self.pos_embed(pos_ids)  # (T, C)

        # Add them together (broadcasting pos_emb across the batch dimension)
        x = tok_emb + pos_embed  # (B, T, C)

        # Pass through the Transformer blocks
        for layer in self.layers:
            x = layer(x)  # (B, T, C)

        # Get logits from decoder head
        logits = self.decoder_head(x)

        # --- Loss calculation ----
        loss = None
        if targets is not None:
            (B, T, C) = logits.shape
            logits_for_loss = logits.view(B * T, C)
            targets_for_loss = targets.view(B * T)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        return x, loss


# -- End of Model Definition ---
