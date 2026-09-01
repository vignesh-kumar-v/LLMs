"""Cross-entropy loss, written out rather than calling F.cross_entropy.

Kept hand-rolled because the point of the project is to build the pieces, but
tightened in three ways that matter once it runs under bf16 autocast:

* **float32 reduction.** Logits arrive in bf16 under autocast. bf16 has ~8 bits
  of mantissa, so accumulating a 50k-wide log-softmax in it loses real
  precision. The logits are upcast before the softmax.
* **gather instead of advanced indexing.** `log_probs[arange(n), targets]`
  materialises an int64 index tensor of length B*T every step; `gather` does
  not.
* **ignore_index.** Lets padding positions be excluded from the mean, which
  advanced indexing could not express.
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # (B, T, C) -> (B*T, C); also accepts pre-flattened input.
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1).long()

        # log_softmax is already the numerically stable form (it subtracts the
        # row max internally); the upcast is about the accumulator width.
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        mask = targets != self.ignore_index
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        picked = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

        picked = torch.where(mask, picked, torch.zeros_like(picked))
        denom = mask.sum().clamp(min=1)
        return -picked.sum() / denom

    # Keeps the original call-site style (`criterion(logits, targets)`) working
    # whether or not it is used as an nn.Module.
    __call__ = nn.Module.__call__
