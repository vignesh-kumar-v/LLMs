"""
Minimal profiling script for ncu.

Usage:
    sudo $(which ncu) --set full --kernel-name fused_layernorm_kernel -o my_profile_v2 $(which python) profile_run.py

Notes:
    - torch.compile is intentionally disabled (breaks ncu kernel replay)
    - Extension is loaded explicitly here so it compiles under whichever user
      (including root when invoked via sudo) rather than relying on NanoLLM's
      cached import path.
    - Kernel is called directly — no model wrapper — so nothing can fall back.
    - NVTX ranges mark the profiled region in the ncu timeline view.
"""
import os
import sys
import torch
from torch.utils.cpp_extension import load_inline

assert torch.cuda.is_available(), "CUDA not available"
print(f"Profiling on: {torch.cuda.get_device_name(0)}")

# ── Load the extension explicitly ─────────────────────────────────────────────
# Bypasses NanoLLM's cached _fused_ln_ext (which may be None under sudo due
# to a different cache directory). Compiles fresh if needed.
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "fused_layernorm.cu")) as f:
    cuda_src = f.read()

print("Compiling fused_layernorm extension...")
ext = load_inline(
    name="fused_ln_profile",
    cpp_sources="",
    cuda_sources=cuda_src,
    verbose=True,
)
print("Extension loaded.\n")

# ── Inputs — float32, no autocast ────────────────────────────────────────────
# The kernel only handles float32. autocast would cast to bfloat16 and trigger
# the F.layer_norm fallback, meaning the kernel would never be called.
B, N = 512, 128          # 512 rows, hidden dim 128 (matches model config)
x     = torch.randn(B, N, device="cuda", dtype=torch.float32)
gamma = torch.ones(N,     device="cuda", dtype=torch.float32)
beta  = torch.zeros(N,    device="cuda", dtype=torch.float32)

# ── Warmup ────────────────────────────────────────────────────────────────────
# Ensures the kernel is compiled, loaded into the GPU instruction cache,
# and the CUDA context is fully initialised before ncu starts capturing.
print("Warming up...")
for _ in range(10):
    out = ext.fused_layernorm(x, gamma, beta, 1e-5)
torch.cuda.synchronize()
print("Warmup done. Starting profiled region...\n")

# ── Profiled region ───────────────────────────────────────────────────────────
torch.cuda.nvtx.range_push("fused_layernorm_forward")
out = ext.fused_layernorm(x, gamma, beta, 1e-5)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("Profiled region complete.")
print(f"Output shape : {out.shape}")
print(f"Output mean  : {out.mean().item():.6f}  (expect ~0)")
print(f"Output std   : {out.std().item():.6f}   (expect ~1)")
