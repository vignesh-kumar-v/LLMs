import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Head(nn.Module):
    def __init__(self, context_length, embed_dim, head_size, drop_out=0.1, activation='relu'):
        super(Head, self).__init__()
        self.head_size = head_size
        self.queries = weight_norm(nn.Linear(embed_dim, head_size, bias=False))
        self.keys = weight_norm(nn.Linear(embed_dim, head_size, bias=False))
        self.values = weight_norm(nn.Linear(embed_dim, head_size, bias=False))
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(drop_out)
        self.activation = F.relu if activation == 'relu' else F.gelu
    def forward(self, inputs):
        B, T, C = inputs.shape
        queries = self.queries(inputs)
        keys = self.keys(inputs)
        values = self.values(inputs)
        attn_scores = queries @ keys.transpose(1, 2) / (self.head_size ** 0.5)
        attn_scores_masked = attn_scores.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        attn_scores_normalized = F.softmax(attn_scores_masked, dim=-1)
        attn_scores_normalized = self.dropout(attn_scores_normalized)
        context_vectors = attn_scores_normalized @ values
        return context_vectors
    
class MultiHeadAttnention(nn.Module):
    def __init__(self, context_length, embed_dim, head_size, num_heads, drop_out=0.1, activation='relu'):
        super(MultiHeadAttnention, self).__init__()
        self.heads = nn.ModuleList([Head(context_length, embed_dim, head_size, drop_out=drop_out, activation=activation) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embed_dim)
    def forward(self, inputs):
        initial_inputs = inputs
        head_outputs = [head(inputs) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        out = F.relu(self.proj(head_outputs))
        out = initial_inputs + out
        return out

class GPT2(nn.Module):
    def __init__(self, vocab_size, context_length=128, embed_dim=128):
        super(GPT2, self).__init__()
        self.context_length = context_length
        self.token_embedding_table = weight_norm(nn.Embedding(vocab_size, embed_dim))
        self.position_embedding_table = weight_norm(nn.Embedding(context_length, embed_dim))
        self.attention_block = MultiHeadAttnention(context_length, embed_dim, head_size=embed_dim//4, num_heads=4, drop_out=0.3, activation='relu')
        self.lm_head = weight_norm(nn.Linear(embed_dim, vocab_size))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, inputs):
        B, T = inputs.shape
        token_embeddings = self.token_embedding_table(inputs)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=inputs.device))
        token_emb = token_embeddings + pos_embeddings
        x = self.layer_norm(token_emb)
        x = self.attention_block(x)
        logits = self.lm_head(x)
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