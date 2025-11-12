import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core building blocks of the Transformer ---


class FeedForward(nn.Module):
    """
    A standard two-layer feed-forward network with a non-linearity and dropout.
    """

    def __init__(self, n_hidden: int, n_ffn: int, dropout: float, act_fn=F.relu) -> None:
        super().__init__()
        # The network consists of an "up-projection" to a higher dimension,
        # followed by a "down-projection" back to the original dimension.
        self.up_proj = nn.Linear(n_hidden, n_ffn)
        self.down_proj = nn.Linear(n_ffn, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = act_fn

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Apply layers in sequence : up-projection -> activation -> down-projection
        x = self.act_fn(self.up_proj(input))
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class Block(nn.Module):
    """
    A single Transformer Block.
    """

    def __init__(self, n_hidden: int, num_heads: int, ffn_ratio: int, attn_dropout: float, ffn_dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(n_hidden, n_hidden)
        self.key = nn.Linear(n_hidden, n_hidden)
        self.value = nn.Linear(n_hidden, n_hidden)
        self.mha = nn.MultiheadAttention(embed_dim=n_hidden, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.ffn = FeedForward(n_hidden=n_hidden, n_ffn=ffn_ratio * n_hidden, dropout=ffn_dropout)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ln1(input)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, _ = self.mha(q, k, v)
        x = input + attn_output
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    A minimal Generative Pre-trained Transformer (GPT) model.
    """

    def __init__(self, vocab_size: int, n_hidden: int, n_layers: int, num_heads: int, ffn_ratio: int, attn_dropout: float, ffn_dropout: float, max_seq_len: int = 512) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_hidden)
        self.pos_embed = nn.Embedding(max_seq_len, n_hidden)
        self.blocks = nn.Sequential(*[Block(n_hidden=n_hidden, num_heads=num_heads, ffn_ratio=ffn_ratio, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout) for _ in range(n_layers)])
        self.decoder_head = nn.Linear(n_hidden, vocab_size)

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
        x = self.blocks(x)  # (B, T, C)

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


# --- Self-Testing Block ---
if __name__ == "__main__":
    # This block of code will only run when the script is executed directly
    # (e.g., `python src/model.py`), not when it's imported by another script.

    print("--- Running Model Self-Test ---")

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

    # --- Test 1: Count Parameters ---
    def count_parameters(model: nn.Module) -> int:
        """Counts the number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters: {count_parameters(model):,d}")

    # --- Test 2: Forward Pass ---
    print("\n--- Testing forward pass ---")
    batch_size = 4
    # Create some dummy input data
    dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    dummy_targets = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    print(f"Input shape: {dummy_input.shape}")

    # Perform a forward pass
    logits, loss = model(dummy_input, targets=dummy_targets)

    print(f"Logits shape: {logits.shape}")
    print(f"Calculated Loss: {loss.item():.2f}")

    # Check if backpropagation works
    loss.backward()
    print("Backward pass successful.")

    print("\n--- Model Self-Test Complete ---")
