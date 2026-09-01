"""NanoLLM — a GPT-2 style decoder-only transformer.

Changes from the original implementation, and why:

* **Batched multi-head attention.** Heads used to be a `nn.ModuleList` of
  independent `SelfAttention` modules run in a Python loop, each with its own
  Q/K/V `nn.Linear`. That launches 3*n_head small GEMMs per block. They are now
  a single fused `c_attn` projection reshaped into heads — one GEMM, and the
  heads become a batch dimension.
* **Scaled dot-product attention.** Uses `F.scaled_dot_product_attention`,
  which dispatches to FlashAttention/memory-efficient kernels. The manual
  softmax path is kept behind a flag for teaching/comparison.
* **Weight tying.** `lm_head.weight` is tied to the token embedding, as in
  real GPT-2. At vocab 50257 x 128 this removes ~6.4M duplicated parameters —
  the majority of the model.
* **KV cache.** `generate()` no longer re-runs the full forward pass over the
  whole context for every single new token.
* **GPT-2 initialisation**, including the 1/sqrt(2*n_layer) scaling on
  residual projections, which matters for stable deep training.
"""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_ln import FusedLayerNorm


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a single fused QKV projection."""

    def __init__(self, context_size, num_embeddings, num_heads, dropout=0.1,
                 use_flash=True):
        super().__init__()
        if num_embeddings % num_heads != 0:
            raise ValueError(
                f"num_embeddings ({num_embeddings}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_size = num_embeddings // num_heads
        self.num_embeddings = num_embeddings
        self.dropout = dropout
        self.use_flash = use_flash

        self.c_attn = nn.Linear(num_embeddings, 3 * num_embeddings)
        self.c_proj = nn.Linear(num_embeddings, num_embeddings)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Only needed by the manual attention path; flash builds its own mask.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_size, context_size)).view(
                1, 1, context_size, context_size
            ),
            persistent=False,
        )

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape

        # (B, T, 3C) -> three (B, num_heads, T, head_size)
        q, k, v = self.c_attn(x).split(self.num_embeddings, dim=2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None

        # With a KV cache the query block is shorter than the key block. A
        # single query attends to every cached key, so no mask applies; only
        # the square case (q_len == k_len) needs causal masking.
        q_len, k_len = q.size(2), k.size(2)
        is_causal = q_len == k_len and q_len > 1

        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            if is_causal:
                att = att.masked_fill(
                    self.tril[:, :, :q_len, :k_len] == 0, float("-inf")
                )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present


class MLP(nn.Module):
    def __init__(self, num_embeddings, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(num_embeddings, 4 * num_embeddings)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * num_embeddings, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, context_size, num_embeddings, num_heads, dropout=0.1,
                 use_flash=True, fused_ln_fallback=False):
        super().__init__()
        self.layernorm_1 = FusedLayerNorm(num_embeddings, force_fallback=fused_ln_fallback)
        self.attention = CausalSelfAttention(
            context_size, num_embeddings, num_heads, dropout, use_flash
        )
        self.layernorm_2 = FusedLayerNorm(num_embeddings, force_fallback=fused_ln_fallback)
        self.mlp = MLP(num_embeddings, dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present = self.attention(
            self.layernorm_1(x), past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.layernorm_2(x))
        return x, present


class GPT2(nn.Module):
    def __init__(self, context_size, vocab_size, num_embeddings, num_heads,
                 num_blocks, dropout=0.1, tie_weights=True, use_flash=True,
                 fused_ln_fallback=False):
        super().__init__()
        self.context_length = context_size
        self.num_blocks = num_blocks

        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.pos_embedding_table = nn.Embedding(context_size, num_embeddings)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(context_size, num_embeddings, num_heads, dropout,
                  use_flash, fused_ln_fallback)
            for _ in range(num_blocks)
        ])
        self.layernorm = FusedLayerNorm(num_embeddings, force_fallback=fused_ln_fallback)
        self.lm_head = nn.Linear(num_embeddings, vocab_size, bias=False)

        if tie_weights:
            # Standard GPT-2 weight tying. Saves vocab_size * n_embd params.
            self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        # Scale residual-path projections so the variance of the residual
        # stream stays ~constant with depth (GPT-2 paper, section 2.3).
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * num_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_embedding_table.weight.numel()
        return n

    def forward(self, inputs, past_kvs=None, use_cache=False):
        B, T = inputs.shape
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        if past_len + T > self.context_length:
            raise ValueError(
                f"sequence length {past_len + T} exceeds context "
                f"length {self.context_length}"
            )

        pos = torch.arange(past_len, past_len + T, device=inputs.device)
        x = self.drop(self.token_embedding_table(inputs) + self.pos_embedding_table(pos))

        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, present = block(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.layernorm(x)
        logits = self.lm_head(x)
        return (logits, presents) if use_cache else logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with the standard GPT-2 parameter grouping.

        Weight decay is applied only to matmul/embedding weights; biases and
        LayerNorm gains are left undecayed. Applying decay to 1D params is a
        well-known way to quietly hurt small-model quality.
        """
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # Fused AdamW is a large win on CUDA and is available in torch >= 2.0.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if (fused_available and device_type == "cuda") else {}
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
        return optimizer, len(decay), len(no_decay)

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens, temperature=1.0, top_k=None,
                 use_cache=True, eos_token=None):
        """Autoregressive sampling with an optional KV cache.

        Without the cache every new token costs a full forward pass over the
        whole context. With it, only the newest token is processed, so
        generation is O(T) rather than O(T^2) in attention work.

        The cache is dropped and rebuilt whenever the sequence would overflow
        the context window: positional embeddings here are absolute and
        learned, so a shifted window changes every cached key's position and
        the cache can no longer be reused. The rebuild deliberately keeps only
        half a window so the cost is amortised over the next ~context_length/2
        tokens, rather than rebuilding on every single step.
        """
        was_training = self.training
        self.eval()
        past_kvs = None
        idx = start_tokens
        rebuild_window = max(1, self.context_length // 2)

        try:
            for _ in range(max_new_tokens):
                if use_cache:
                    if past_kvs is None:
                        # First step: seed the cache with the whole prompt.
                        model_input = idx[:, -self.context_length:]
                    elif past_kvs[0][0].size(2) + 1 > self.context_length:
                        past_kvs = None
                        model_input = idx[:, -rebuild_window:]
                    else:
                        model_input = idx[:, -1:]
                    logits, past_kvs = self(model_input, past_kvs=past_kvs, use_cache=True)
                else:
                    logits = self(idx[:, -self.context_length:])

                logits = logits[:, -1, :]
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))

                probs = F.softmax(logits.float(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)

                if eos_token is not None and (next_token == eos_token).all():
                    break
        finally:
            self.train(was_training)

        return idx
