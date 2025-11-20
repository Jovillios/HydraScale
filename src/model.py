import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = getattr(F, self.hidden_act)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Apply layers in sequence : up-projection -> activation -> down-projection
        x_gate = F.silu(self.gate_proj(input))
        x_up = self.up_proj(input)
        x = self.down_proj(x_gate * x_up)
        return x


class Block(nn.Module):
    """
    A single Transformer Block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # sanity check

        # parameters
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout

        # modules
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_attention_heads, dropout=self.attention_dropout, batch_first=True)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ln1(input)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, _ = self.mha(q, k, v)
        x = input + attn_output
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
