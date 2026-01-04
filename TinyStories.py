import torch
from torch.utils.data import Dataset

class TinyStoriesDataset(Dataset):
    def __init__(self, tokens, context_length=128, stride=8):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        self.num_samples = (len(tokens)-context_length) // stride+1
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length
        inputs = self.tokens[start: end]
        targets = self.tokens[start+1: end+1]
        return inputs, targets