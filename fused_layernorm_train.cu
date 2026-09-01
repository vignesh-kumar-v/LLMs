// fused_layernorm_train.cu — production LayerNorm kernels (forward + backward)
//
// Why this file exists separately from fused_layernorm{,_v3}.cu:
//   Those two files hold the V1/V2/V3 *benchmark* progression (float32, forward
//   only). They demonstrate the optimisation story but cannot be used in real
//   training because (a) they are float32-only, so under bf16 autocast the
//   module silently fell back to F.layer_norm, and (b) they have no backward
//   pass, so gradients could not flow through them.
//
// This file fixes both:
//   * templated on scalar type  -> float32, bfloat16 and float16 all supported
//   * saves mean/rstd in forward -> backward can recompute xhat without a
//     second pass over global memory for statistics
//   * full backward: dx (row-parallel) + dgamma/dbeta (column-parallel)
//
// Numerics: x/out/dy are stored in scalar_t, but every accumulator, and gamma/
// beta/dgamma/dbeta, are float32. This matches what PyTorch/Apex do — reducing
// a 128..1024 element row in bf16 would lose far too much precision.
//
// Algorithm carried over from V3: single-pass Welford, vectorised 16-byte
// loads, multiple rows per block, two-level warp-shuffle reduction.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#define WARP_SIZE 32

// ─────────────────────────────────────────────────────────────────────────────
// 16-byte vector type: 4 floats, or 8 bf16/fp16. Lets one instruction move a
// full 128-bit transaction regardless of element width.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T, int W>
struct alignas(sizeof(T) * W) Vec {
    T v[W];
};

template <typename T>
struct VecWidth {
    static constexpr int value = 16 / sizeof(T);
};

// ─────────────────────────────────────────────────────────────────────────────
// Welford merge — combines two (mean, M2, count) accumulators.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void welford_merge(
    float& mean_a, float& M2_a, int& cnt_a,
    float b_mean, float b_M2, int b_cnt)
{
    int new_cnt = cnt_a + b_cnt;
    if (new_cnt == 0) return;
    float delta = b_mean - mean_a;
    float nb = (float)b_cnt, na = (float)cnt_a, nn = (float)new_cnt;
    mean_a = mean_a + delta * nb / nn;
    M2_a   = M2_a + b_M2 + delta * delta * na * nb / nn;
    cnt_a  = new_cnt;
}

__device__ __forceinline__ void warp_welford_reduce(
    float& mean, float& M2, int& cnt, int lane)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float b_mean = __shfl_down_sync(0xffffffff, mean, offset);
        float b_M2   = __shfl_down_sync(0xffffffff, M2,   offset);
        int   b_cnt  = __shfl_down_sync(0xffffffff, cnt,  offset);
        if (lane < offset)
            welford_merge(mean, M2, cnt, b_mean, b_M2, b_cnt);
    }
}

// Plain sum reduction within a warp (used by the backward pass).
__device__ __forceinline__ float warp_sum(float v)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// FORWARD
//   out[row,i] = gamma[i] * (x[row,i] - mean[row]) * rstd[row] + beta[i]
//   also writes mean[row], rstd[row] for the backward pass.
//
// Block layout: blockDim = (threads_x, rows_per_block), chosen at launch by
//   pick_launch_config() to match the row width — see the note there.
//   threadIdx.y selects the row this thread works on, threadIdx.x the position
//   within that row. threads_x is always a multiple of 32, so no warp ever
//   spans two rows — warp-level shuffles stay row-local and `active` is
//   uniform across every warp (important: threads must not early-return, or
//   the __syncthreads below would deadlock).
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, int VEC>
__global__ void ln_fwd_kernel(
    const scalar_t* __restrict__ x,
    const float*    __restrict__ gamma,
    const float*    __restrict__ beta,
    scalar_t*       __restrict__ out,
    float*          __restrict__ mean_out,
    float*          __restrict__ rstd_out,
    int M, int N, float eps)
{
    const int row    = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid    = threadIdx.x;
    const int lane   = tid % WARP_SIZE;
    const int warp   = tid / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    const bool active = (row < M);

    using VecT = Vec<scalar_t, VEC>;

    float mean = 0.f, M2 = 0.f;
    int   cnt  = 0;

    // ── Phase 1: single-pass Welford over the row, 16B vector loads ──────────
    if (active) {
        if (VEC > 1) {
            const VecT* xv = reinterpret_cast<const VecT*>(x + (size_t)row * N);
            const int   nvec = N / VEC;
            for (int i = tid; i < nvec; i += blockDim.x) {
                VecT chunk = xv[i];
                #pragma unroll
                for (int k = 0; k < VEC; ++k) {
                    float val = static_cast<float>(chunk.v[k]);
                    cnt++;
                    float delta = val - mean;
                    mean += delta / (float)cnt;
                    M2   += delta * (val - mean);
                }
            }
        } else {
            for (int i = tid; i < N; i += blockDim.x) {
                float val = static_cast<float>(x[(size_t)row * N + i]);
                cnt++;
                float delta = val - mean;
                mean += delta / (float)cnt;
                M2   += delta * (val - mean);
            }
        }
    }

    // ── Phase 2: intra-warp Welford reduction (registers only) ───────────────
    warp_welford_reduce(mean, M2, cnt, lane);

    // ── Phase 3: warp leaders publish to shared memory ───────────────────────
    // Layout per row-slot: [mean x nwarps | M2 x nwarps | cnt x nwarps]
    extern __shared__ float smem[];
    float* s_mean = smem + threadIdx.y * nwarps * 3;
    float* s_M2   = s_mean + nwarps;
    float* s_cnt  = s_M2 + nwarps;

    if (lane == 0) {
        s_mean[warp] = mean;
        s_M2[warp]   = M2;
        s_cnt[warp]  = (float)cnt;
    }
    __syncthreads();

    // ── Phase 4: warp 0 reduces the per-warp partials, in parallel ───────────
    if (warp == 0) {
        mean = (lane < nwarps) ? s_mean[lane] : 0.f;
        M2   = (lane < nwarps) ? s_M2[lane]   : 0.f;
        cnt  = (lane < nwarps) ? (int)s_cnt[lane] : 0;

        warp_welford_reduce(mean, M2, cnt, lane);

        if (lane == 0) {
            float var  = M2 / (float)N;
            float rstd = rsqrtf(var + eps);
            s_mean[0] = mean;
            s_M2[0]   = rstd;
            if (active) {
                mean_out[row] = mean;
                rstd_out[row] = rstd;
            }
        }
    }
    __syncthreads();

    const float row_mean = s_mean[0];
    const float row_rstd = s_M2[0];

    // ── Phase 5: normalise + affine, vectorised store ────────────────────────
    if (active) {
        if (VEC > 1) {
            const VecT* xv = reinterpret_cast<const VecT*>(x + (size_t)row * N);
            VecT*       ov = reinterpret_cast<VecT*>(out + (size_t)row * N);
            const int   nvec = N / VEC;
            for (int i = tid; i < nvec; i += blockDim.x) {
                VecT chunk = xv[i];
                VecT res;
                #pragma unroll
                for (int k = 0; k < VEC; ++k) {
                    int idx = i * VEC + k;
                    float val = static_cast<float>(chunk.v[k]);
                    float nrm = (val - row_mean) * row_rstd;
                    res.v[k] = static_cast<scalar_t>(gamma[idx] * nrm + beta[idx]);
                }
                ov[i] = res;
            }
        } else {
            for (int i = tid; i < N; i += blockDim.x) {
                float val = static_cast<float>(x[(size_t)row * N + i]);
                float nrm = (val - row_mean) * row_rstd;
                out[(size_t)row * N + i] = static_cast<scalar_t>(gamma[i] * nrm + beta[i]);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BACKWARD — dx
//   xhat = (x - mean) * rstd
//   dxhat = dy * gamma
//   dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
//
// Row-parallel: each row needs two scalar reductions (c1, c2), so this mirrors
// the forward kernel's block layout and reduction structure.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, int VEC>
__global__ void ln_bwd_dx_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    const float*    __restrict__ gamma,
    const float*    __restrict__ mean,
    const float*    __restrict__ rstd,
    scalar_t*       __restrict__ dx,
    int M, int N)
{
    const int row    = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid    = threadIdx.x;
    const int lane   = tid % WARP_SIZE;
    const int warp   = tid / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    const bool active = (row < M);

    using VecT = Vec<scalar_t, VEC>;

    const float row_mean = active ? mean[row] : 0.f;
    const float row_rstd = active ? rstd[row] : 0.f;

    float c1 = 0.f, c2 = 0.f;   // sum(dxhat), sum(dxhat * xhat)

    if (active) {
        if (VEC > 1) {
            const VecT* dyv = reinterpret_cast<const VecT*>(dy + (size_t)row * N);
            const VecT* xv  = reinterpret_cast<const VecT*>(x  + (size_t)row * N);
            const int   nvec = N / VEC;
            for (int i = tid; i < nvec; i += blockDim.x) {
                VecT dchunk = dyv[i];
                VecT xchunk = xv[i];
                #pragma unroll
                for (int k = 0; k < VEC; ++k) {
                    int idx = i * VEC + k;
                    float dxhat = static_cast<float>(dchunk.v[k]) * gamma[idx];
                    float xhat  = (static_cast<float>(xchunk.v[k]) - row_mean) * row_rstd;
                    c1 += dxhat;
                    c2 += dxhat * xhat;
                }
            }
        } else {
            for (int i = tid; i < N; i += blockDim.x) {
                float dxhat = static_cast<float>(dy[(size_t)row * N + i]) * gamma[i];
                float xhat  = (static_cast<float>(x[(size_t)row * N + i]) - row_mean) * row_rstd;
                c1 += dxhat;
                c2 += dxhat * xhat;
            }
        }
    }

    c1 = warp_sum(c1);
    c2 = warp_sum(c2);

    extern __shared__ float smem[];
    float* s_c1 = smem + threadIdx.y * nwarps * 2;
    float* s_c2 = s_c1 + nwarps;

    if (lane == 0) { s_c1[warp] = c1; s_c2[warp] = c2; }
    __syncthreads();

    if (warp == 0) {
        c1 = (lane < nwarps) ? s_c1[lane] : 0.f;
        c2 = (lane < nwarps) ? s_c2[lane] : 0.f;
        c1 = warp_sum(c1);
        c2 = warp_sum(c2);
        if (lane == 0) {
            s_c1[0] = c1 / (float)N;
            s_c2[0] = c2 / (float)N;
        }
    }
    __syncthreads();

    const float m1 = s_c1[0];
    const float m2 = s_c2[0];

    if (active) {
        if (VEC > 1) {
            const VecT* dyv = reinterpret_cast<const VecT*>(dy + (size_t)row * N);
            const VecT* xv  = reinterpret_cast<const VecT*>(x  + (size_t)row * N);
            VecT*       dxv = reinterpret_cast<VecT*>(dx + (size_t)row * N);
            const int   nvec = N / VEC;
            for (int i = tid; i < nvec; i += blockDim.x) {
                VecT dchunk = dyv[i];
                VecT xchunk = xv[i];
                VecT res;
                #pragma unroll
                for (int k = 0; k < VEC; ++k) {
                    int idx = i * VEC + k;
                    float dxhat = static_cast<float>(dchunk.v[k]) * gamma[idx];
                    float xhat  = (static_cast<float>(xchunk.v[k]) - row_mean) * row_rstd;
                    res.v[k] = static_cast<scalar_t>(row_rstd * (dxhat - m1 - xhat * m2));
                }
                dxv[i] = res;
            }
        } else {
            for (int i = tid; i < N; i += blockDim.x) {
                float dxhat = static_cast<float>(dy[(size_t)row * N + i]) * gamma[i];
                float xhat  = (static_cast<float>(x[(size_t)row * N + i]) - row_mean) * row_rstd;
                dx[(size_t)row * N + i] =
                    static_cast<scalar_t>(row_rstd * (dxhat - m1 - xhat * m2));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BACKWARD — dgamma / dbeta
//   dgamma[i] = sum_rows dy[row,i] * xhat[row,i]
//   dbeta[i]  = sum_rows dy[row,i]
//
// These are *column* reductions, so the row-parallel layout above is wrong for
// them. Instead: a 2D grid where blockIdx.x tiles columns and blockIdx.y tiles
// rows; each block reduces its (32 cols x row-chunk) tile into a partial row.
// The (few) partials are then summed with a torch reduction on the host side,
// which avoids a second hand-written kernel and any atomics.
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void ln_bwd_dwdb_partial_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    const float*    __restrict__ mean,
    const float*    __restrict__ rstd,
    float*          __restrict__ dgamma_part,
    float*          __restrict__ dbeta_part,
    int M, int N, int rows_per_chunk)
{
    // +1 padding on the inner dimension avoids shared-memory bank conflicts
    // when the reduction below walks down a column.
    __shared__ float s_dg[WARP_SIZE][WARP_SIZE + 1];
    __shared__ float s_db[WARP_SIZE][WARP_SIZE + 1];

    const int col = blockIdx.x * WARP_SIZE + threadIdx.x;

    float dg = 0.f, db = 0.f;

    if (col < N) {
        const int row_start = blockIdx.y * rows_per_chunk;
        int row_end = row_start + rows_per_chunk;
        if (row_end > M) row_end = M;

        for (int row = row_start + threadIdx.y; row < row_end; row += WARP_SIZE) {
            float d    = static_cast<float>(dy[(size_t)row * N + col]);
            float xhat = (static_cast<float>(x[(size_t)row * N + col]) - mean[row]) * rstd[row];
            dg += d * xhat;
            db += d;
        }
    }

    s_dg[threadIdx.y][threadIdx.x] = dg;
    s_db[threadIdx.y][threadIdx.x] = db;
    __syncthreads();

    // Tree-reduce down the y axis (across row groups) for this column.
    for (int s = WARP_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.y < s) {
            s_dg[threadIdx.y][threadIdx.x] += s_dg[threadIdx.y + s][threadIdx.x];
            s_db[threadIdx.y][threadIdx.x] += s_db[threadIdx.y + s][threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0 && col < N) {
        dgamma_part[(size_t)blockIdx.y * N + col] = s_dg[0][threadIdx.x];
        dbeta_part [(size_t)blockIdx.y * N + col] = s_db[0][threadIdx.x];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side launchers
// ─────────────────────────────────────────────────────────────────────────────

// Target threads per block. 256 keeps occupancy high without starving the
// per-SM register budget.
static constexpr int TARGET_BLOCK = 256;

// Size the block to the row's actual work.
//
// The first version hard-coded 256 threads x 4 rows regardless of N. At N=128
// with 4-wide vectors a row is only 32 vector elements, so 224 of the 256
// threads had nothing to do, and the cross-warp reduction still ran over all 8
// warps — 7 of them empty. That made the kernel 3.6x slower than PyTorch on a
// T4. Now threads_x covers one row (rounded up to a whole warp so no warp ever
// straddles two rows) and the leftover block capacity is spent on more rows.
static void pick_launch_config(int N, int vec_width, int& threads_x, int& rows_per_block)
{
    const int nvec = (N + vec_width - 1) / vec_width;
    threads_x = ((nvec + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    if (threads_x < WARP_SIZE)   threads_x = WARP_SIZE;
    if (threads_x > TARGET_BLOCK) threads_x = TARGET_BLOCK;  // wide rows loop
    rows_per_block = TARGET_BLOCK / threads_x;
    if (rows_per_block < 1) rows_per_block = 1;
}

// The vectorised path reinterprets the row pointer as a 16-byte type, so it is
// only legal when the row stride keeps every row 16B-aligned and the base
// pointer itself is aligned. PyTorch's caching allocator returns 512B-aligned
// storage, but a non-contiguous or offset view could still break this — so we
// check rather than assume (the original V3 kernel assumed).
template <typename scalar_t>
static bool can_vectorize(const void* ptr, int N)
{
    constexpr int W = VecWidth<scalar_t>::value;
    if (N % W != 0) return false;
    return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

#define LN_DISPATCH(TYPE, NAME, ...)                                          \
    [&] {                                                                     \
        switch (TYPE) {                                                       \
            case at::ScalarType::Float: {                                     \
                using scalar_t = float;      return __VA_ARGS__();            \
            }                                                                 \
            case at::ScalarType::BFloat16: {                                  \
                using scalar_t = at::BFloat16; return __VA_ARGS__();          \
            }                                                                 \
            case at::ScalarType::Half: {                                      \
                using scalar_t = at::Half;   return __VA_ARGS__();            \
            }                                                                 \
            default:                                                          \
                TORCH_CHECK(false, NAME " unsupported dtype: ", TYPE);        \
                return 0;                                                     \
        }                                                                     \
        return 0;                                                             \
    }()

std::vector<torch::Tensor> fused_layernorm_forward(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, N)");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Float &&
                beta.scalar_type()  == at::ScalarType::Float,
                "gamma/beta must be float32");
    x     = x.contiguous();
    gamma = gamma.contiguous();
    beta  = beta.contiguous();

    const int M = x.size(0);
    const int N = x.size(1);

    auto out  = torch::empty_like(x);
    auto opts = x.options().dtype(torch::kFloat32);
    auto mean = torch::empty({M}, opts);
    auto rstd = torch::empty({M}, opts);

    if (M == 0) return {out, mean, rstd};

    auto stream = at::cuda::getCurrentCUDAStream();

    LN_DISPATCH(x.scalar_type(), "fused_layernorm_forward", [&] {
        const bool vec = can_vectorize<scalar_t>(x.data_ptr(), N) &&
                         can_vectorize<scalar_t>(out.data_ptr(), N);
        const int vw = vec ? VecWidth<scalar_t>::value : 1;

        int threads_x, rows_per_block;
        pick_launch_config(N, vw, threads_x, rows_per_block);
        const int nwarps = threads_x / WARP_SIZE;
        dim3 block(threads_x, rows_per_block);
        dim3 grid((M + rows_per_block - 1) / rows_per_block);
        size_t smem = (size_t)rows_per_block * nwarps * 3 * sizeof(float);

        if (vec) {
            ln_fwd_kernel<scalar_t, VecWidth<scalar_t>::value>
                <<<grid, block, smem, stream>>>(
                    x.data_ptr<scalar_t>(), gamma.data_ptr<float>(),
                    beta.data_ptr<float>(), out.data_ptr<scalar_t>(),
                    mean.data_ptr<float>(), rstd.data_ptr<float>(), M, N, (float)eps);
        } else {
            ln_fwd_kernel<scalar_t, 1><<<grid, block, smem, stream>>>(
                x.data_ptr<scalar_t>(), gamma.data_ptr<float>(),
                beta.data_ptr<float>(), out.data_ptr<scalar_t>(),
                mean.data_ptr<float>(), rstd.data_ptr<float>(), M, N, (float)eps);
        }
        return 0;
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out, mean, rstd};
}

std::vector<torch::Tensor> fused_layernorm_backward(
    torch::Tensor dy, torch::Tensor x, torch::Tensor gamma,
    torch::Tensor mean, torch::Tensor rstd)
{
    TORCH_CHECK(dy.is_cuda() && x.is_cuda(), "dy/x must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, N)");
    TORCH_CHECK(gamma.scalar_type() == at::ScalarType::Float,
                "gamma must be float32");
    // Autograd can hand back a grad whose dtype differs from the activation
    // (e.g. an fp32 grad into a bf16 layer); the kernel indexes both with the
    // same scalar_t, so normalise here rather than reinterpreting memory.
    if (dy.scalar_type() != x.scalar_type()) dy = dy.to(x.scalar_type());
    dy    = dy.contiguous();
    x     = x.contiguous();
    gamma = gamma.contiguous();

    const int M = x.size(0);
    const int N = x.size(1);

    auto dx     = torch::empty_like(x);
    auto fopts  = x.options().dtype(torch::kFloat32);
    auto dgamma = torch::zeros({N}, fopts);
    auto dbeta  = torch::zeros({N}, fopts);

    if (M == 0) return {dx, dgamma, dbeta};

    auto stream = at::cuda::getCurrentCUDAStream();

    // ── dx ───────────────────────────────────────────────────────────────────
    {
        LN_DISPATCH(x.scalar_type(), "fused_layernorm_backward", [&] {
            const bool vec = can_vectorize<scalar_t>(x.data_ptr(), N) &&
                             can_vectorize<scalar_t>(dy.data_ptr(), N) &&
                             can_vectorize<scalar_t>(dx.data_ptr(), N);
            const int vw = vec ? VecWidth<scalar_t>::value : 1;

            int threads_x, rows_per_block;
            pick_launch_config(N, vw, threads_x, rows_per_block);
            const int nwarps = threads_x / WARP_SIZE;
            dim3 block(threads_x, rows_per_block);
            dim3 grid((M + rows_per_block - 1) / rows_per_block);
            size_t smem = (size_t)rows_per_block * nwarps * 2 * sizeof(float);

            if (vec) {
                ln_bwd_dx_kernel<scalar_t, VecWidth<scalar_t>::value>
                    <<<grid, block, smem, stream>>>(
                        dy.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                        gamma.data_ptr<float>(), mean.data_ptr<float>(),
                        rstd.data_ptr<float>(), dx.data_ptr<scalar_t>(), M, N);
            } else {
                ln_bwd_dx_kernel<scalar_t, 1><<<grid, block, smem, stream>>>(
                    dy.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                    gamma.data_ptr<float>(), mean.data_ptr<float>(),
                    rstd.data_ptr<float>(), dx.data_ptr<scalar_t>(), M, N);
            }
            return 0;
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ── dgamma / dbeta ───────────────────────────────────────────────────────
    {
        // Enough chunks to fill the GPU, but few enough that the final torch
        // sum over partials stays trivial.
        int num_chunks = (M + WARP_SIZE - 1) / WARP_SIZE;
        if (num_chunks > 256) num_chunks = 256;
        if (num_chunks < 1)   num_chunks = 1;
        const int rows_per_chunk = (M + num_chunks - 1) / num_chunks;

        auto dgamma_part = torch::empty({num_chunks, N}, fopts);
        auto dbeta_part  = torch::empty({num_chunks, N}, fopts);

        dim3 block(WARP_SIZE, WARP_SIZE);
        dim3 grid((N + WARP_SIZE - 1) / WARP_SIZE, num_chunks);

        LN_DISPATCH(x.scalar_type(), "fused_layernorm_backward_dwdb", [&] {
            ln_bwd_dwdb_partial_kernel<scalar_t><<<grid, block, 0, stream>>>(
                dy.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                mean.data_ptr<float>(), rstd.data_ptr<float>(),
                dgamma_part.data_ptr<float>(), dbeta_part.data_ptr<float>(),
                M, N, rows_per_chunk);
            return 0;
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        dgamma = dgamma_part.sum(0);
        dbeta  = dbeta_part.sum(0);
    }

    return {dx, dgamma, dbeta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &fused_layernorm_forward,
          "Fused LayerNorm forward (fp32/bf16/fp16, returns out/mean/rstd)");
    m.def("backward", &fused_layernorm_backward,
          "Fused LayerNorm backward (returns dx/dgamma/dbeta)");
}
