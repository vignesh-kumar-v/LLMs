import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.cpp_extension import load_inline

_ext_v1v2 = None   # fused_layernorm (V2 Welford) + fused_layernorm_naive (V1)
_ext_v3   = None   # fused_layernorm_v3 (float4 + two-level warp shuffle)

try:
    _dir = os.path.dirname(__file__)
    _ext_v1v2 = load_inline(
        name="fused_ln",
        cpp_sources="",
        cuda_sources=open(os.path.join(_dir, "fused_layernorm.cu")).read(),
        verbose=False,
    )
    _ext_v3 = load_inline(
        name="fused_ln_v3",
        cpp_sources="",
        cuda_sources=open(os.path.join(_dir, "fused_layernorm_v3.cu")).read(),
        verbose=False,
    )
except Exception:
    pass


class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Only custom kernels are float32-only; fall back for bfloat16 / CPU
        if x.is_cuda and x.dtype == torch.float32:
            N     = self.normalized_shape
            shape = x.shape
            x_2d  = x.view(-1, N)
            # Dispatch based on N — learned from benchmarking:
            #   N <= 256 : V2 Welford wins  (~8.4 µs at N=128)
            #   N >  256 : V3 float4  wins  (~18.5 µs at N=768, vs 49 µs V1)
            if N <= 256 and _ext_v1v2 is not None:
                out = _ext_v1v2.fused_layernorm(x_2d, self.weight, self.bias, self.eps)
            elif N > 256 and _ext_v3 is not None:
                out = _ext_v3.fused_layernorm_v3(x_2d, self.weight, self.bias, self.eps)
            else:
                return F.layer_norm(x, (N,), self.weight, self.bias, self.eps)
            return out.view(shape)
        return F.layer_norm(x, (self.normalized_shape,),
                            self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)

class SelfAttention(nn.Module):
    def __init__(self, context_size, num_embeddings, head_size):
        super(SelfAttention, self).__init__()
        self.head_size = head_size
        self.query = nn.Linear(num_embeddings, head_size)
        self.key = nn.Linear(num_embeddings, head_size)
        self.value = nn.Linear(num_embeddings, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # B, 256, 256
        k = self.key(x)   # B, 256, 256
        v = self.value(x) # B, 256, 256
        wei = q @ k.transpose(-2, -1) / (self.head_size**0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = self.dropout(wei @ v)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, context_size, num_embeddings, head_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([SelfAttention(context_size, num_embeddings, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_embeddings)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class MLP(nn.Module):
    def __init__(self, num_embeddings):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.GELU(),
            nn.Linear(num_embeddings * 4, num_embeddings)
        )
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.dropout(self.net(x))

class Block(nn.Module):
    def __init__(self, context_size, num_embeddings, num_heads):
        super(Block, self).__init__()
        head_size = num_embeddings //  num_heads
        self.multihead_attention = MultiHeadAttention(context_size, num_embeddings, head_size, num_heads)
        self.mlp = MLP(num_embeddings)
        self.layernorm_1 = FusedLayerNorm(num_embeddings)
        self.layernorm_2 = FusedLayerNorm(num_embeddings)
    def forward(self, x):
        x = x + self.multihead_attention(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, context_size, vocab_size, num_embeddings, num_heads, num_blocks):
        super(GPT2, self).__init__()
        self.context_length = context_size
        self.num_blocks = num_blocks
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.pos_embedding_table = nn.Embedding(context_size, num_embeddings)
        self.blocks = nn.ModuleList([Block(context_size, num_embeddings, num_heads) for _ in range(num_blocks)])
        self.lm_head = nn.Linear(num_embeddings, vocab_size)
        self.layernorm = FusedLayerNorm(num_embeddings)
    def forward(self, inputs):
        token_embeddings = self.token_embedding_table(inputs)
        pos_embeddings = self.pos_embedding_table(torch.arange(inputs.shape[1], device=inputs.device))
        outputs = token_embeddings + pos_embeddings
        for block in self.blocks:
            outputs = block(outputs)
        outputs = self.layernorm(outputs)
        logits = self.lm_head(outputs)
        return logits
    def generate(self, start_tokens, max_new_tokens):
        for _ in range(max_new_tokens):
            context = start_tokens[:, -self.context_length:]
            logits = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            start_tokens = torch.cat((start_tokens, next_token), dim=1)
        return start_tokens