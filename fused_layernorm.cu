#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ─────────────────────────────────────────────────────────────────────────────
// V1 — Naive kernel (kept for benchmarking comparison)
// Two separate tree-reductions via shared memory: one for mean, one for variance.
// Problems: 2 passes over data, 6x __syncthreads, partial[1024] wastes 4 KB/block.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_layernorm_kernel_naive(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ out,
    int N, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* row_data = smem;               // N floats

    float val = 0.0f;
    if (tid < N) { val = x[row * N + tid]; row_data[tid] = val; }
    __syncthreads();

    __shared__ float partial[1024];
    partial[tid] = (tid < N) ? row_data[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    float mean = partial[0] / N;

    partial[tid] = (tid < N) ? (row_data[tid] - mean) * (row_data[tid] - mean) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(partial[0] / N + eps);

    if (tid < N)
        out[row * N + tid] = gamma[tid] * (row_data[tid] - mean) * inv_std + beta[tid];
}

torch::Tensor fused_layernorm_naive_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps = 1e-5)
{
    int B = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    int threads = 1;
    while (threads < N) threads <<= 1;
    threads = min(threads, 1024);
    size_t smem = N * sizeof(float);
    fused_layernorm_kernel_naive<<<B, threads, smem>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), out.data_ptr<float>(), N, eps);
    return out;
}


// ─────────────────────────────────────────────────────────────────────────────
// V2 — Welford + warp-shuffle kernel
//
// Key improvements over V1:
//   1. Single pass  — Welford computes mean and variance simultaneously.
//   2. Warp shuffles — __shfl_down_sync replaces shared-memory tree reduction
//                      inside each warp (zero __syncthreads within a warp).
//   3. Tiny smem    — only 3 * nwarps floats (48 B for N=128) vs 4608 B before.
//   4. val in register — x[row*N+tid] loaded once, reused for normalisation.
// ─────────────────────────────────────────────────────────────────────────────

// Merge two Welford accumulators (a) += (b)
__device__ __forceinline__ void welford_merge(
    float& mean_a, float& M2_a, int& cnt_a,
    float  mean_b, float  M2_b, int  cnt_b)
{
    if (cnt_b == 0) return;
    int   new_cnt   = cnt_a + cnt_b;
    float delta     = mean_b - mean_a;
    mean_a = mean_a + delta * (float)cnt_b  / (float)new_cnt;
    M2_a   = M2_a + M2_b + delta * delta * (float)cnt_a * (float)cnt_b / (float)new_cnt;
    cnt_a  = new_cnt;
}

__global__ void fused_layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ out,
    int N, float eps)
{
    int row    = blockIdx.x;
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warp   = tid >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    // ── Phase 1: each thread loads one element, seeds its Welford state ──────
    float val  = (tid < N) ? x[row * N + tid] : 0.0f;
    float mean = (tid < N) ? val  : 0.0f;
    float M2   = 0.0f;
    int   cnt  = (tid < N) ? 1   : 0;

    // ── Phase 2: warp-level Welford reduction via __shfl_down_sync ───────────
    // No shared memory, no __syncthreads — pure register communication.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float b_mean = __shfl_down_sync(0xffffffff, mean, offset);
        float b_M2   = __shfl_down_sync(0xffffffff, M2,   offset);
        int   b_cnt  = __shfl_down_sync(0xffffffff, cnt,  offset);
        if (lane < offset)
            welford_merge(mean, M2, cnt, b_mean, b_M2, b_cnt);
    }
    // lane 0 of each warp now holds that warp's accumulated (mean, M2, cnt)

    // ── Phase 3: inter-warp reduction via shared memory ──────────────────────
    // smem layout: [mean×nwarps | M2×nwarps | cnt×nwarps]
    extern __shared__ float smem[];
    float* s_mean = smem;
    float* s_M2   = smem + nwarps;
    float* s_cnt  = smem + nwarps * 2;

    if (lane == 0) {
        s_mean[warp] = mean;
        s_M2[warp]   = M2;
        s_cnt[warp]  = (float)cnt;
    }
    __syncthreads();

    // Thread 0 merges all warp results sequentially (nwarps <= 32, so very fast)
    if (tid == 0) {
        for (int w = 1; w < nwarps; w++) {
            float wm = s_mean[w], wM2 = s_M2[w];
            int   wc = (int)s_cnt[w];
            welford_merge(mean, M2, cnt, wm, wM2, wc);
        }
        s_mean[0] = mean;
        s_M2[0]   = M2;
    }
    __syncthreads();

    // ── Phase 4: normalise using register-cached val (no global re-read) ─────
    float final_mean = s_mean[0];
    float inv_std    = rsqrtf(s_M2[0] / (float)N + eps);

    if (tid < N)
        out[row * N + tid] = gamma[tid] * (val - final_mean) * inv_std + beta[tid];
}

torch::Tensor fused_layernorm_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps = 1e-5)
{
    int B = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    int threads = 1;
    while (threads < N) threads <<= 1;
    threads = min(threads, 1024);
    int    nwarps  = (threads + 31) / 32;
    size_t smem_sz = 3 * nwarps * sizeof(float);   // 48 B for N=128
    fused_layernorm_kernel<<<B, threads, smem_sz>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), out.data_ptr<float>(), N, eps);
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm",       &fused_layernorm_cuda,       "Fused LayerNorm — Welford + warp shuffle");
    m.def("fused_layernorm_naive", &fused_layernorm_naive_cuda, "Fused LayerNorm — naive tree reduction");
}
