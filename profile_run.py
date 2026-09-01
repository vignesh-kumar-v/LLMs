"""Minimal profiling harness for Nsight Compute.

    sudo -E $(which ncu) --set full --kernel-name regex:ln_fwd_kernel \
        -o nanollm_fwd $(which python) profile_run.py

    # backward instead
    sudo -E $(which ncu) --set full --kernel-name regex:ln_bwd \
        -o nanollm_bwd $(which python) profile_run.py --backward

Why it is written this way:
  * `torch.compile` is never used here — it breaks ncu's kernel replay.
  * The extension is loaded directly rather than through the model, so nothing
    can quietly fall back to `F.layer_norm` and leave you profiling the wrong
    kernel.
  * NVTX ranges label each region in the ncu timeline.

The original version profiled in float32 only, because the kernels were
float32-only. They now run in bf16 too, which is the dtype training actually
uses — `--dtype` picks which one you profile.
"""

import argparse

import torch

import fused_ln


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192,
                        help="rows = batch * context length")
    parser.add_argument("--hidden", type=int, default=384, help="normalised dim")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--backward", action="store_true",
                        help="also run the backward pass inside the profiled range")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"Profiling on: {torch.cuda.get_device_name(0)}")

    if fused_ln.load_extension(verbose=True) is None:
        raise SystemExit(f"extension unavailable: {fused_ln.kernel_status()}")
    print(f"Loaded: {fused_ln.kernel_status()}\n")

    dtype = getattr(torch, args.dtype)
    M, N = args.rows, args.hidden

    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=args.backward)
    gamma = torch.ones(N, device="cuda", dtype=torch.float32)
    beta = torch.zeros(N, device="cuda", dtype=torch.float32)

    module = fused_ln.FusedLayerNorm(N).cuda()

    # Warmup: compile/load the kernel and settle the CUDA context before ncu
    # starts capturing, so the measured launch is representative.
    print("Warming up ...")
    for _ in range(10):
        out = module(x)
        if args.backward:
            out.backward(torch.ones_like(out))
            x.grad = None
            module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print("Warmup done. Starting profiled region.\n")

    torch.cuda.nvtx.range_push("fused_layernorm_forward")
    out = module(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    if args.backward:
        torch.cuda.nvtx.range_push("fused_layernorm_backward")
        out.backward(torch.ones_like(out))
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    print(f"Shape       : {tuple(out.shape)}  ({args.dtype})")
    print(f"Output mean : {out.float().mean().item():+.6f}  (expect ~0)")
    print(f"Output std  : {out.float().std().item():.6f}   (expect ~1)")
    if args.backward:
        print(f"dx norm     : {x.grad.float().norm().item():.4f}")
        print(f"dgamma norm : {module.weight.grad.norm().item():.4f}")


if __name__ == "__main__":
    main()
