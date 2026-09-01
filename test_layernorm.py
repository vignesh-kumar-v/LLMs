"""Correctness + benchmark suite for the fused LayerNorm kernels.

Covers two things that the original version could not:

* the **production** kernel (`fused_layernorm_train.cu`) in fp32 *and* bf16 —
  bf16 is the dtype training actually runs in, so an fp32-only test told us
  nothing about the path used in anger;
* the **backward** pass — dx, dgamma and dbeta checked against PyTorch
  autograd, plus a `gradcheck` in float64-ish tolerance on a small case.

The V1/V2/V3 comparison from the original file is kept, since that progression
is the point of the exercise.

    python test_layernorm.py            # correctness + benchmarks
    python test_layernorm.py --quick    # correctness only
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

import fused_ln


def _load_benchmark_kernels():
    """V1/V2/V3 — forward-only, float32, kept for the optimisation story."""
    from torch.utils.cpp_extension import load

    # Note the name: it must NOT be "fused_ln", which is this repo's Python
    # module — load() registers the extension in sys.modules under this name
    # and would shadow it.
    v1v2 = load(name="fused_ln_v1v2", sources=["fused_layernorm.cu"],
                extra_cuda_cflags=["-O3"], verbose=False)
    v3 = load(name="fused_ln_bench_v3", sources=["fused_layernorm_v3.cu"],
              extra_cuda_cflags=["-O3"], verbose=False)
    return v1v2, v3


def benchmark(fn, *args, warmup=25, iters=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────

def check_forward_backward(B, N, dtype, failures):
    """Compare the fused module against an fp32 reference, forward and backward.

    The reference runs entirely in float32 rather than using nn.LayerNorm at
    `dtype`: the kernel deliberately keeps gamma/beta and all accumulators in
    fp32 while the activations are bf16, so an fp32 reference is what it should
    actually be judged against. It also sidesteps nn.LayerNorm refusing a bf16
    input against fp32 weights outside autocast.
    """
    torch.manual_seed(0)
    label = f"B={B} N={N} {str(dtype).split('.')[-1]}"

    x = torch.randn(B, N, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().float().clone().requires_grad_(True)

    w = torch.randn(N, device="cuda") * 0.3 + 1.0
    b = torch.randn(N, device="cuda") * 0.1

    fused = fused_ln.FusedLayerNorm(N).cuda()
    with torch.no_grad():
        fused.weight.copy_(w)
        fused.bias.copy_(b)
    w_ref = w.clone().requires_grad_(True)
    b_ref = b.clone().requires_grad_(True)

    out = fused(x)
    out_ref = F.layer_norm(x_ref, (N,), w_ref, b_ref, fused.eps)

    grad = torch.randn(B, N, device="cuda", dtype=dtype)
    out.backward(grad)
    out_ref.backward(grad.float())

    # bf16 carries ~3 decimal digits, so tolerances scale with the dtype.
    # dgamma/dbeta accumulate over B rows, so their tolerance scales with B too.
    tol = {torch.float32: 2e-5, torch.bfloat16: 3e-2, torch.float16: 1e-2}[dtype]
    red_tol = tol * max(1.0, B / 128)

    results = {
        "forward": ((out.float() - out_ref).abs().max().item(), tol),
        "dx": ((x.grad.float() - x_ref.grad).abs().max().item(), tol),
        "dgamma": ((fused.weight.grad - w_ref.grad).abs().max().item(), red_tol),
        "dbeta": ((fused.bias.grad - b_ref.grad).abs().max().item(), red_tol),
    }

    print(f"  {label}")
    for name, (err, limit) in results.items():
        ok = err <= limit
        print(f"    {name:8s} max err {err:.3e}   {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{label} {name} err={err:.3e} > tol={limit:.1e}")


def check_shapes(failures):
    """Non-multiple-of-vector-width N must take the scalar fallback path."""
    print("  odd shapes (scalar fallback path)")
    for N in (17, 100, 129, 1000):
        x = torch.randn(8, N, device="cuda", dtype=torch.float32, requires_grad=True)
        fused = fused_ln.FusedLayerNorm(N).cuda()
        out = fused(x)
        ref = F.layer_norm(x, (N,), fused.weight, fused.bias, fused.eps)
        err = (out - ref).abs().max().item()
        ok = err < 2e-5
        print(f"    N={N:<5d} max err {err:.3e}   {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"N={N} fallback err={err:.3e}")


def check_3d_and_large(failures):
    """The module reshapes (B,T,C) -> (B*T,C); and N > 1024 must work."""
    print("  3D input and large N")
    for shape, N in (((4, 128, 384), 384), ((2, 64, 2048), 2048)):
        x = torch.randn(*shape, device="cuda", dtype=torch.float32, requires_grad=True)
        fused = fused_ln.FusedLayerNorm(N).cuda()
        out = fused(x)
        ref = F.layer_norm(x, (N,), fused.weight, fused.bias, fused.eps)
        err = (out - ref).abs().max().item()
        ok = err < 2e-5
        print(f"    {str(shape):<18s} max err {err:.3e}   {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"shape {shape} err={err:.3e}")


def check_gradcheck(failures):
    """Analytic gradients vs. finite differences on a small float64-free case."""
    print("  autograd gradcheck (float32, loose eps)")
    torch.manual_seed(0)
    N = 16
    x = torch.randn(4, N, device="cuda", dtype=torch.float32, requires_grad=True)
    fused = fused_ln.FusedLayerNorm(N).cuda()

    ok = torch.autograd.gradcheck(
        lambda inp: fused(inp), (x,), eps=1e-3, atol=1e-2, rtol=1e-2,
        raise_exception=False,
    )
    print(f"    gradcheck {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append("gradcheck failed")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmarks(v1v2, v3, low_dtype):
    B = 512
    for N in (128, 768):
        print(f"\n── Benchmark  (B={B}, N={N}) " + "─" * 34)
        x = torch.randn(B, N, device="cuda", dtype=torch.float32)
        gamma = torch.ones(N, device="cuda")
        beta = torch.zeros(N, device="cuda")
        ln = torch.nn.LayerNorm(N).cuda()

        rows = []
        if N <= 1024:
            rows.append(("V1 naive", benchmark(v1v2.fused_layernorm_naive, x, gamma, beta, 1e-5)))
            rows.append(("V2 Welford", benchmark(v1v2.fused_layernorm, x, gamma, beta, 1e-5)))
        if N % 4 == 0:
            rows.append(("V3 float4", benchmark(v3.fused_layernorm_v3, x, gamma, beta, 1e-5)))
        rows.append(("Production fp32", benchmark(
            lambda: torch.ops.nanollm.fused_ln_fwd(x, gamma, beta, 1e-5))))
        rows.append(("PyTorch LN", benchmark(ln, x)))

        short = str(low_dtype).split(".")[-1]
        xb = x.to(low_dtype)
        rows.append((f"Production {short}", benchmark(
            lambda: torch.ops.nanollm.fused_ln_fwd(xb, gamma, beta, 1e-5))))
        lnb = torch.nn.LayerNorm(N).cuda().to(low_dtype)
        rows.append((f"PyTorch LN {short}", benchmark(lnb, xb)))

        best = min(t for _, t in rows)
        for name, t in rows:
            print(f"  {name:18s} {t:8.2f} us   ({t/best:.2f}x vs best)")

        # Forward+backward is what training actually pays for.
        print(f"\n── Fwd+Bwd    (B={B}, N={N}) " + "─" * 34)
        for name, module, dt in (
            (f"Fused {short}", fused_ln.FusedLayerNorm(N).cuda(), low_dtype),
            (f"PyTorch {short}", torch.nn.LayerNorm(N).cuda().to(low_dtype), low_dtype),
        ):
            xi = torch.randn(B, N, device="cuda", dtype=dt, requires_grad=True)

            def step():
                o = module(xi)
                o.backward(torch.ones_like(o))
                xi.grad = None

            print(f"  {name:18s} {benchmark(step):8.2f} us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip benchmarks")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — these kernels require an NVIDIA GPU.")
        return 0

    major, minor = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)} (sm_{major}{minor})")

    # bf16 needs sm_80+. On Turing (T4, sm_75) test fp16 instead — that is the
    # dtype training actually falls back to there, so it is the one that
    # matters on this hardware.
    if major >= 8:
        test_dtypes = (torch.float32, torch.bfloat16)
    else:
        test_dtypes = (torch.float32, torch.float16)
        print("  sm_75 or older: no hardware bf16, testing float16 instead")
    print("Compiling production kernel ...")
    if fused_ln.load_extension(verbose=False) is None:
        print(f"FAILED: {fused_ln.kernel_status()}")
        return 1
    print(f"  {fused_ln.kernel_status()}\n")

    failures = []
    print("── Correctness " + "─" * 48)
    for dtype in test_dtypes:
        for B, N in ((512, 128), (512, 768), (1000, 384)):
            check_forward_backward(B, N, dtype, failures)
    check_shapes(failures)
    check_3d_and_large(failures)
    check_gradcheck(failures)

    if not args.quick:
        print("\nCompiling V1/V2/V3 benchmark kernels ...")
        v1v2, v3 = _load_benchmark_kernels()
        run_benchmarks(v1v2, v3, test_dtypes[-1])

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All correctness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
