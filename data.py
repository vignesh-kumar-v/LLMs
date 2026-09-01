"""Tokenisation and batching for TinyStories.

Replaces the old `dataset.py`, which defined a `TinyStoriesDataset` that
nothing imported — the training loop had already moved to memmap + random
window sampling and the Dataset class was dead code.

Fixes carried in here:

* **Streaming tokenisation.** The original read the entire ~2 GB `train.txt`
  into one Python string and handed it to `tiktoken` in a single call. That
  peaks at many GB of RAM. Tokenisation is now chunked and appended
  incrementally.
* **Staleness detection.** The original skipped tokenisation whenever
  `train.bin` merely existed, so editing `train.txt` silently trained on the
  old tokens. A sidecar `.meta.json` records the source size and mtime.
* **Memmap reuse.** `get_batch` re-opened both memmaps on every call. They are
  now opened once and cached.
* **Rank-aware sampling.** Under DDP each rank must draw *different* windows,
  otherwise all ranks compute identical gradients and the run is just a slower
  single-GPU job. Each rank seeds its own generator.
"""

import json
import os

import numpy as np
import torch

#: uint16 storage is only valid while the vocabulary fits in 16 bits.
_MAX_UINT16_VOCAB = 2**16


def _source_meta(txt_path):
    st = os.stat(txt_path)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def prepare_bin(txt_path, bin_path, tokenizer, chunk_lines=100_000, force=False,
                verbose=True):
    """Tokenise ``txt_path`` into a flat uint16 ``bin_path``.

    Skips the work when the sidecar metadata shows the source is unchanged.
    """
    if tokenizer.n_vocab >= _MAX_UINT16_VOCAB:
        raise ValueError(
            f"vocab size {tokenizer.n_vocab} does not fit in uint16; "
            "widen the dtype in prepare_bin/get_batch"
        )
    if not os.path.exists(txt_path):
        raise FileNotFoundError(
            f"{txt_path} not found — run `python TinyStories.py` first"
        )

    meta_path = bin_path + ".meta.json"
    current = _source_meta(txt_path)

    if not force and os.path.exists(bin_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                cached = json.load(f)
            if cached.get("source") == current:
                if verbose:
                    n = os.path.getsize(bin_path) // 2
                    print(f"[data] {bin_path} up to date ({n:,} tokens), skipping.")
                return
            if verbose:
                print(f"[data] {txt_path} changed since {bin_path} was built; re-tokenising.")
        except (json.JSONDecodeError, OSError):
            pass  # unreadable metadata -> just rebuild

    if verbose:
        print(f"[data] tokenising {txt_path} -> {bin_path} ...")

    total = 0
    tmp_path = bin_path + ".tmp"
    with open(txt_path, "r", encoding="utf-8") as src, open(tmp_path, "wb") as dst:
        batch = []
        for line in src:
            batch.append(line)
            if len(batch) >= chunk_lines:
                total += _encode_and_write(batch, tokenizer, dst)
                batch = []
                if verbose:
                    print(f"[data]   {total:,} tokens ...", flush=True)
        if batch:
            total += _encode_and_write(batch, tokenizer, dst)

    os.replace(tmp_path, bin_path)
    with open(meta_path, "w") as f:
        json.dump({"source": current, "tokens": total}, f)
    if verbose:
        print(f"[data] wrote {total:,} tokens to {bin_path}")


def _encode_and_write(lines, tokenizer, handle):
    ids = tokenizer.encode("".join(lines), allowed_special={"<|endoftext|>"})
    np.asarray(ids, dtype=np.uint16).tofile(handle)
    return len(ids)


class TokenStream:
    """Random fixed-length windows over a flat token file.

    Sampling windows at random (rather than iterating a sliding window) means
    every step sees a fresh, decorrelated batch without needing to shuffle a
    2 GB array or track epoch boundaries.
    """

    def __init__(self, bin_path, context_length, device, seed=1337, rank=0):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"{bin_path} not found — run prepare_bin first")
        self.bin_path = bin_path
        self.context_length = context_length
        self.device = device
        # Opened once; np.memmap does not read the file into RAM.
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if len(self.data) <= context_length + 1:
            raise ValueError(
                f"{bin_path} holds only {len(self.data)} tokens, need more than "
                f"{context_length + 1}"
            )
        # Distinct stream per rank so DDP workers see different data.
        self.generator = torch.Generator().manual_seed(seed + 1000 * rank)
        self.high = len(self.data) - context_length - 1

    def batch(self, batch_size, generator=None):
        gen = generator or self.generator
        idx = torch.randint(0, self.high, (batch_size,), generator=gen)
        ctx = self.context_length

        # Build one contiguous numpy block, then a single host->device copy,
        # instead of stacking batch_size separate small tensors.
        x_np = np.stack([self.data[i : i + ctx] for i in idx.tolist()])
        y_np = np.stack([self.data[i + 1 : i + 1 + ctx] for i in idx.tolist()])

        x = torch.from_numpy(x_np.astype(np.int64))
        y = torch.from_numpy(y_np.astype(np.int64))

        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def fixed_batches(self, batch_size, num_batches, seed=0):
        """Deterministic batches, so validation loss is comparable across epochs."""
        gen = torch.Generator().manual_seed(seed)
        for _ in range(num_batches):
            yield self.batch(batch_size, generator=gen)

    def __len__(self):
        return len(self.data)
