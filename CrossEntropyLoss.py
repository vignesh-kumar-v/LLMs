import torch

class CrossEntropyLoss:
    def __call__(self, logits, targets):
        B, T, C = logits.shape
        logits = logits.reshape(-1, C)
        targets = targets.reshape(-1).long()
        log_probs = torch.log_softmax(logits, dim=1)
        log_probs_true = log_probs[torch.arange(log_probs.size(0)), targets]
        loss = -log_probs_true.mean()
        return loss