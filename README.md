# NanoLLM — Tiny Language Model Trainer

A GPT-2 style transformer trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, built from scratch in PyTorch. The project goes beyond a basic implementation — it includes **custom fused CUDA kernels** for LayerNorm with three progressively optimised versions, an ncu profiling pipeline, and full GPU memory/utilisation tracking.

---

## Project Structure

```
LLMs/
├── NanoLLM.py              # Model architecture + FusedLayerNorm with kernel dispatch
├── main.py                 # Training loop (get_batch memmap, grad clip, GPU monitoring)
├── config.py               # All hyperparameters
├── dataset.py              # TinyStoriesDataset (sliding window)
├── CrossEntropyLoss.py     # Custom cross-entropy loss
├── TinyStories.py          # Dataset downloader from HuggingFace
│
├── fused_layernorm.cu      # CUDA kernels: V1 (naive) + V2 (Welford + warp shuffle)
├── fused_layernorm_v3.cu   # CUDA kernel:  V3 (float4 + two-level warp shuffle + multi-row blocks)
├── test_layernorm.py       # Correctness checks + benchmark (V1 vs V2 vs V3 vs PyTorch)
├── profile_run.py          # Minimal ncu profiling script (torch.compile disabled)
│
└── requirements.txt
```

---

## Model Architecture

| Component | Detail |
|---|---|
| Type | GPT-2 style decoder-only transformer |
| Vocabulary | GPT-2 BPE via tiktoken (50,257 tokens) |
| Context length | 128 tokens |
| Embedding dim | 128 |
| Attention heads | 4 |
| Transformer blocks | 4 |
| Activation | GELU |
| Dropout | 0.1 |
| LayerNorm | Custom fused CUDA kernel (see below) |

---

## Custom CUDA LayerNorm Kernels

Three kernels are implemented in `fused_layernorm.cu` and `fused_layernorm_v3.cu`, with automatic dispatch inside `FusedLayerNorm`:

| Version | Technique | Best at |
|---|---|---|
| **V1** | Shared-memory tree reduction (2 passes: mean then variance) | baseline |
| **V2** | Welford online algorithm + `__shfl_down_sync` warp shuffle | N ≤ 256 |
| **V3** | float4 vectorised loads + two-level warp shuffle + multi-row blocks | N > 256 |

**Benchmark results (B=512):**

| Kernel | N=128 | N=768 |
|---|---|---|
| V1 Naive | 8.38 µs | 49.15 µs |
| V2 Welford | 8.37 µs | 67.64 µs |
| V3 float4 | 16.42 µs | **18.46 µs** |
| PyTorch LN | 14.33 µs | 14.15 µs |

`FusedLayerNorm` dispatches automatically:
- `float32`, `N ≤ 256` → **V2** (~1.7× faster than PyTorch at N=128)
- `float32`, `N > 256` → **V3** (2.66× faster than V1 at N=768)
- `bfloat16` / CPU → `F.layer_norm` fallback

---

## Training

### 1. Install dependencies
```bash
python -m venv .
source bin/activate
pip install -r requirements.txt
sudo apt install ninja-build   # required for JIT-compiling CUDA extensions
```

### 2. Download the dataset
```bash
python TinyStories.py   # saves train.txt and val.txt
```

### 3. Run training
```bash
python main.py
```

On first run, `train.txt` and `val.txt` are tokenised and saved as `train.bin` / `val.bin` (memmap). Subsequent runs skip tokenisation.

Training prints per-epoch:
```
Epoch 1/50, Train Loss: 7.9899, Val Loss: 6.2418, Mem: 1.23GB (peak 1.45GB), GPU util: 87%
```

After training, `training_stats.png` is saved with three plots: loss curves, GPU memory (allocated + peak), and GPU utilisation.

### Configuration (`config.py`)

```python
# Dataset
context_length  = 128     # token sequence length
batch_size      = 8       # training batch size
steps_per_epoch = 1000    # training steps per epoch
val_steps       = 200     # validation steps per epoch

# Model
use_compile     = True    # set False when profiling with ncu
num_embeddings  = 128     # embedding / hidden dimension
num_heads       = 4       # attention heads
num_blocks      = 4       # transformer blocks
learning_rate   = 3e-4
num_epochs      = 50
```

### Training details

- **Optimiser**: AdamW (`weight_decay=1e-5`)
- **LR schedule**: CosineAnnealingLR (`eta_min=1e-7`)
- **Precision**: BF16 autocast (forward + validation) — no GradScaler needed
- **Gradient clipping**: `max_norm=1.0`
- **Compilation**: `torch.compile()` (toggle via `config.use_compile`)
- **Data loading**: `np.memmap` + random sampling via `get_batch()`
- **Checkpointing**: `best_model.pt` saved on validation loss improvement

---

## Monitoring

```bash
tensorboard --logdir=runs
```

Logs: train/val loss, gradient norms, weight norms, GPU memory (allocated + peak), GPU utilisation.

---

## CUDA Kernel Benchmarking

```bash
python test_layernorm.py
```

Runs correctness checks (max error vs PyTorch) and latency benchmarks for all three kernel versions at N=128 and N=768.

---

## Profiling with Nsight Compute

```bash
# One-time setup
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
sudo apt install ninja-build

# Profile only the fused LayerNorm kernel
sudo -E $(which ncu) --set full --kernel-name fused_layernorm_kernel \
    -o my_profile $(which python) profile_run.py

# Open in GUI
ncu-ui my_profile.ncu-rep
```

`profile_run.py` has `torch.compile` disabled and calls the kernel directly (bypassing the model) to ensure the custom kernel is captured rather than falling back to PyTorch's implementation. NVTX range markers (`"fused_layernorm_forward"`) label the profiled region in the timeline view.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- `ninja-build` (for JIT CUDA kernel compilation)
- See `requirements.txt` for full Python dependencies

---

## License

MIT License

## Acknowledgments

- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) by Eldan & Li
- GPT-2 architecture by OpenAI
- tiktoken tokeniser by OpenAI
