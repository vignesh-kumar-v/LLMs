import torch
import time
from torch.utils.cpp_extension import load_inline

# ── Compile all three kernels ─────────────────────────────────────────────────
with open("fused_layernorm.cu")    as f: cuda_src_v1v2 = f.read()
with open("fused_layernorm_v3.cu") as f: cuda_src_v3   = f.read()

print("Compiling V1 + V2...")
ext    = load_inline(name="fused_ln",    cpp_sources="", cuda_sources=cuda_src_v1v2, verbose=False)
print("Compiling V3...")
ext_v3 = load_inline(name="fused_ln_v3", cpp_sources="", cuda_sources=cuda_src_v3,   verbose=False)
print("Done.\n")


def benchmark(fn, *args, warmup=50, iters=500):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6   # µs


B = 512
for N in [128, 768]:
    x     = torch.randn(B, N, device="cuda", dtype=torch.float32)
    gamma = torch.ones(N,     device="cuda", dtype=torch.float32)
    beta  = torch.zeros(N,    device="cuda", dtype=torch.float32)
    ln    = torch.nn.LayerNorm(N).cuda()
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0.0)

    # ── Correctness ───────────────────────────────────────────────────────────
    with torch.no_grad():
        ref    = ln(x)
        out_v1 = ext.fused_layernorm_naive(x, gamma, beta, 1e-5)
        out_v2 = ext.fused_layernorm(x, gamma, beta, 1e-5)
        out_v3 = ext_v3.fused_layernorm_v3(x, gamma, beta, 1e-5)

    print(f"── Correctness  (B={B}, N={N}) ──────────────────────────────")
    print(f"  V1 max error vs PyTorch : {(out_v1 - ref).abs().max().item():.2e}")
    print(f"  V2 max error vs PyTorch : {(out_v2 - ref).abs().max().item():.2e}")
    print(f"  V3 max error vs PyTorch : {(out_v3 - ref).abs().max().item():.2e}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    t_v1 = benchmark(ext.fused_layernorm_naive,    x, gamma, beta, 1e-5)
    t_v2 = benchmark(ext.fused_layernorm,          x, gamma, beta, 1e-5)
    t_v3 = benchmark(ext_v3.fused_layernorm_v3,    x, gamma, beta, 1e-5)
    t_pt = benchmark(ln,                           x)

    fastest = min(t_v1, t_v2, t_v3, t_pt)

    print(f"\n── Benchmark    (B={B}, N={N}) ──────────────────────────────")
    print(f"  V1 naive   : {t_v1:7.2f} µs   ({t_v1/fastest:.2f}×  vs best)")
    print(f"  V2 Welford : {t_v2:7.2f} µs   ({t_v2/fastest:.2f}×  vs best)")
    print(f"  V3 float4  : {t_v3:7.2f} µs   ({t_v3/fastest:.2f}×  vs best)")
    print(f"  PyTorch    : {t_pt:7.2f} µs   ({t_pt/fastest:.2f}×  vs best)")
    print(f"\n  V2 vs V1   : {'faster' if t_v2 < t_v1 else 'slower'} by {max(t_v1,t_v2)/min(t_v1,t_v2):.2f}×")
    print(f"  V3 vs V1   : {'faster' if t_v3 < t_v1 else 'slower'} by {max(t_v1,t_v3)/min(t_v1,t_v3):.2f}×")
    print(f"  V3 vs V2   : {'faster' if t_v3 < t_v2 else 'slower'} by {max(t_v2,t_v3)/min(t_v2,t_v3):.2f}×")
    print()
