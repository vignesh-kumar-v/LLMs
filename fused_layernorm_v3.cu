// fused_layernorm_v3.cu — two-level warp shuffle, float4 loads, multi-row blocks

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void welford_merge(
    float& mean_a, float& M2_a, int& cnt_a,
    float b_mean,  float b_M2,  int b_cnt)
{
    int new_cnt = cnt_a + b_cnt;
    if (new_cnt == 0) return;
    float delta = b_mean - mean_a;
    mean_a = mean_a + delta * b_cnt / (float)new_cnt;
    M2_a   = M2_a + b_M2 + delta * delta * cnt_a * b_cnt / (float)new_cnt;
    cnt_a  = new_cnt;
}

__global__ void fused_layernorm_v3_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ out,
    int N, float eps)
{
    // Block handles ROWS_PER_BLOCK rows simultaneously → fixes occupancy
    // threadIdx.y = which row within block, threadIdx.x = position within row
    int row    = blockIdx.x * blockDim.y + threadIdx.y;
    int tid    = threadIdx.x;
    int lane   = tid % 32;
    int warp   = tid / 32;
    int nwarps = blockDim.x / 32;

    // Phase 1: Vectorized float4 load + online Welford (single pass)
    float mean = 0.f, M2 = 0.f;
    int   cnt  = 0;

    // float4: process 4 elements per thread per iteration → 4× bandwidth efficiency
    const float4* x4 = reinterpret_cast<const float4*>(x + row * N);
    int vec_iters = N / 4;

    for (int i = tid; i < vec_iters; i += blockDim.x) {
        float4 v = x4[i];
        // Welford update for each of the 4 elements
        auto update = [&](float val) {
            cnt++;
            float delta = val - mean;
            mean += delta / cnt;
            M2   += delta * (val - mean);
        };
        update(v.x); update(v.y); update(v.z); update(v.w);
    }
    // Handle remainder (when N % 4 != 0)
    for (int i = vec_iters * 4 + tid; i < N; i += blockDim.x) {
        cnt++;
        float val   = x[row * N + i];
        float delta = val - mean;
        mean += delta / cnt;
        M2   += delta * (val - mean);
    }

    // Phase 2: Warp-level reduction (fully parallel, no smem)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float b_mean = __shfl_down_sync(0xffffffff, mean, offset);
        float b_M2   = __shfl_down_sync(0xffffffff, M2,   offset);
        int   b_cnt  = __shfl_down_sync(0xffffffff, cnt,  offset);
        if (lane < offset)
            welford_merge(mean, M2, cnt, b_mean, b_M2, b_cnt);
    }

    // Phase 3: Cross-warp — warp leaders write to smem
    // smem layout: [nwarps means | nwarps M2s | nwarps cnts] per row-slot
    extern __shared__ float smem[];
    int    row_slot = threadIdx.y * nwarps * 3;
    float* s_mean   = smem + row_slot;
    float* s_M2     = s_mean + nwarps;
    float* s_cnt    = s_M2   + nwarps;

    if (lane == 0) {
        s_mean[warp] = mean;
        s_M2[warp]   = M2;
        s_cnt[warp]  = (float)cnt;
    }
    __syncthreads();  // barrier 1: wait for all warp leaders

    // Phase 4: First warp reduces across all warp leaders — PARALLEL shuffle
    // (fixes V2's serial thread-0 loop bottleneck)
    if (warp == 0) {
        mean = (lane < nwarps) ? s_mean[lane] : 0.f;
        M2   = (lane < nwarps) ? s_M2[lane]   : 0.f;
        cnt  = (lane < nwarps) ? (int)s_cnt[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float b_mean = __shfl_down_sync(0xffffffff, mean, offset);
            float b_M2   = __shfl_down_sync(0xffffffff, M2,   offset);
            int   b_cnt  = __shfl_down_sync(0xffffffff, cnt,  offset);
            if (lane < offset)
                welford_merge(mean, M2, cnt, b_mean, b_M2, b_cnt);
        }

        if (lane == 0) {
            s_mean[0] = mean;
            s_M2[0]   = M2 / N;  // store variance
        }
    }
    __syncthreads();  // barrier 2: broadcast final stats

    float final_mean    = s_mean[0];
    float final_inv_std = rsqrtf(s_M2[0] + eps);

    // Phase 5: Vectorized float4 write (re-reads x via x4 — already in L1/L2)
    float4*       out4 = reinterpret_cast<float4*>(out + row * N);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);
    const float4* b4   = reinterpret_cast<const float4*>(beta);

    for (int i = tid; i < vec_iters; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = g4[i];
        float4 bv = b4[i];
        float4 ov;
        ov.x = gv.x * (xv.x - final_mean) * final_inv_std + bv.x;
        ov.y = gv.y * (xv.y - final_mean) * final_inv_std + bv.y;
        ov.z = gv.z * (xv.z - final_mean) * final_inv_std + bv.z;
        ov.w = gv.w * (xv.w - final_mean) * final_inv_std + bv.w;
        out4[i] = ov;
    }
    // Handle remainder
    for (int i = vec_iters * 4 + tid; i < N; i += blockDim.x)
        out[row * N + i] = gamma[i] * (x[row * N + i] - final_mean) * final_inv_std + beta[i];
}

torch::Tensor fused_layernorm_v3_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps = 1e-5)
{
    int B = x.size(0);
    int N = x.size(1);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::Float,
                "V3 kernel requires a float32 CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "V3 kernel requires a contiguous tensor");
    // The float4 path reinterprets the row pointer as a 16-byte type. PyTorch's
    // allocator returns aligned storage, but a view with a non-zero offset does
    // not have to be — so assert instead of assuming.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0,
                "V3 kernel requires 16-byte aligned input");
    TORCH_CHECK(N % 4 == 0,
                "V3 kernel requires N divisible by 4 for aligned row starts, got N=", N);

    auto out = torch::empty_like(x);

    const int ROWS_PER_BLOCK = 4;
    const int THREADS_X      = 256;
    int nwarps = THREADS_X / 32;   // 8

    dim3 block(THREADS_X, ROWS_PER_BLOCK);
    dim3 grid((B + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    // smem: [mean + M2 + cnt] × nwarps × ROWS_PER_BLOCK
    size_t smem = ROWS_PER_BLOCK * nwarps * 3 * sizeof(float);

    fused_layernorm_v3_kernel<<<grid, block, smem>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), out.data_ptr<float>(), N, eps);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm_v3", &fused_layernorm_v3_cuda,
          "Fused LayerNorm v3 — float4 + two-level warp shuffle + multi-row blocks");
}
