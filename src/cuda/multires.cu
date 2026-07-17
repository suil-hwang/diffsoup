#include "multires.cuh"

#include <math.h>

#include "cuda_common.cuh"

namespace diffsoup {
namespace cuda {

__device__ void multires_triangle_interp_d(
    float b0, float b1,            // barycentric coordinates
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    const float* features,         // [S, feature_dim], where S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* out                     // [feature_dim]
) {
    uint32_t offset = 0;

    // For each subdivision level from min_level to max_level
    for (uint32_t level = min_level; level <= max_level; ++level) {
        const uint32_t res = 1 << level;
        const float res_f = static_cast<float>(res);

        float b0_level = b0 * res_f;
        float b1_level = b1 * res_f;

        const uint32_t x = MIN(static_cast<uint32_t>(floorf(b0_level)), res - 1);
        const uint32_t y = MIN(static_cast<uint32_t>(floorf(b1_level)), res - 1 - x);  // x + y <= res - 1

        b0_level = b0_level - static_cast<float>(x);
        b1_level = b1_level - static_cast<float>(y);

        const bool flip = b0_level + b1_level > 1.f;
        const uint32_t flip_u = static_cast<uint32_t>(flip);
        const float flip_f = static_cast<float>(flip);

        const uint32_t x0 = x + 1;
        const uint32_t y0 = y;
        const uint32_t x1 = x;
        const uint32_t y1 = y + 1;
        const uint32_t x2 = x + flip_u;
        const uint32_t y2 = MIN(y + flip_u, res - x2);  // x2 + y2 <= res

        uint32_t index[3];
        index[0] = (x0 + y0) * (x0 + y0 + 1) / 2 + y0;
        index[1] = (x1 + y1) * (x1 + y1 + 1) / 2 + y1;
        index[2] = (x2 + y2) * (x2 + y2 + 1) / 2 + y2;

        float weight[3];
        weight[0] = (1.f - flip_f) * b0_level + flip_f * (1.f - b1_level);
        weight[1] = (1.f - flip_f) * b1_level + flip_f * (1.f - b0_level);
        weight[2] = 1.f - weight[0] - weight[1];

        const float* level_features = features + offset * feature_dim;

        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            #pragma unroll 4
            for (uint32_t j = 0; j < feature_dim; ++j) {
                out[j] += weight[i] * level_features[index[i] * feature_dim + j];
            }
        }

        if (level == 0) offset += 3;
        else offset += ((1 << (level - 1)) + 1) * ((1 << level) + 1);
    }
}

__device__ void backward_multires_triangle_interp_d(
    float b0, float b1,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    float* grad_features,          // [S, feature_dim], where S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* grad_out          // [feature_dim]
) {
    uint32_t offset = 0;

    // For each subdivision level from min_level to max_level
    for (uint32_t level = min_level; level <= max_level; ++level) {
        const uint32_t res = 1 << level;
        const float res_f = static_cast<float>(res);

        float b0_level = b0 * res_f;
        float b1_level = b1 * res_f;

        const uint32_t x = MIN(static_cast<uint32_t>(floorf(b0_level)), res - 1);
        const uint32_t y = MIN(static_cast<uint32_t>(floorf(b1_level)), res - 1 - x);  // x + y <= res - 1

        b0_level = b0_level - static_cast<float>(x);
        b1_level = b1_level - static_cast<float>(y);

        const bool flip = b0_level + b1_level > 1.f;
        const uint32_t flip_u = static_cast<uint32_t>(flip);
        const float flip_f = static_cast<float>(flip);

        const uint32_t x0 = x + 1;
        const uint32_t y0 = y;
        const uint32_t x1 = x;
        const uint32_t y1 = y + 1;
        const uint32_t x2 = x + flip_u;
        const uint32_t y2 = MIN(y + flip_u, res - x2);  // x2 + y2 <= res

        uint32_t index[3];
        index[0] = (x0 + y0) * (x0 + y0 + 1) / 2 + y0;
        index[1] = (x1 + y1) * (x1 + y1 + 1) / 2 + y1;
        index[2] = (x2 + y2) * (x2 + y2 + 1) / 2 + y2;

        float weight[3];
        weight[0] = (1.f - flip_f) * b0_level + flip_f * (1.f - b1_level);
        weight[1] = (1.f - flip_f) * b1_level + flip_f * (1.f - b0_level);
        weight[2] = 1.f - weight[0] - weight[1];

        float* level_grad_features = grad_features + offset * feature_dim;

        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            #pragma unroll 4
            for (uint32_t j = 0; j < feature_dim; ++j) {
                atomicAdd(&level_grad_features[index[i] * feature_dim + j], grad_out[j] * weight[i]);
            }
        }

        if (level == 0) offset += 3;
        else offset += ((1 << (level - 1)) + 1) * ((1 << level) + 1);
    }
}

__global__ void multires_triangle_alpha_kernel(
    int num_frags,
    const float* __restrict__ frag_attrs,   // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    const float* alpha_src,                 // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* __restrict__ frag_alpha          // [num_frags]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int triangle_id = static_cast<int>(frag_attrs[idx * 4 + 3]) - 1;
    if (triangle_id < 0) return;

    const float b0 = frag_attrs[idx * 4 + 0];
    const float b1 = frag_attrs[idx * 4 + 1];

    float alpha = 0.f;

    multires_triangle_interp_d(
        b0, b1, min_level, max_level, /*feature_dim=*/1,
        &alpha_src[triangle_id * S], &alpha
    );

    frag_alpha[idx] = alpha;
}

void multires_triangle_alpha(
    int num_frags,
    const float* __restrict__ frag_attrs,   // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const float* alpha_src,                 // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* __restrict__ frag_alpha,         // [num_frags]
    cudaStream_t stream
) {
    if (num_frags == 0) return;

    const uint32_t S = total_feats_from_levels(min_level, max_level);

    multires_triangle_alpha_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS, 0, stream>>>(
        num_frags, frag_attrs, min_level, max_level, S,
        alpha_src, frag_alpha
    );

    CUDA_CHECK(cudaGetLastError());
}

__global__ void backward_multires_triangle_alpha_kernel(
    int num_frags,
    const float* __restrict__ frag_attrs,        // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    float* grad_alpha_src,                       // [T, S]
    const float* __restrict__ grad_frag_alpha    // [num_frags]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int triangle_id = static_cast<int>(frag_attrs[idx * 4 + 3]) - 1;
    if (triangle_id < 0) return;

    const float b0 = frag_attrs[idx * 4 + 0];
    const float b1 = frag_attrs[idx * 4 + 1];
    const float grad_alpha = grad_frag_alpha[idx];

    backward_multires_triangle_interp_d(
        b0, b1, min_level, max_level, /*feature_dim=*/1,
        &grad_alpha_src[triangle_id * S],
        &grad_alpha
    );
}

void backward_multires_triangle_alpha(
    int num_frags,
    const float* __restrict__ frag_attrs,      // [num_frags, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    float* grad_alpha_src,                      // [T, S], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_frag_alpha,  // [num_frags]
    cudaStream_t stream
) {
    if (num_frags == 0) return;

    const uint32_t S = total_feats_from_levels(min_level, max_level);

    backward_multires_triangle_alpha_kernel<<<CUDA_BLOCKS(num_frags), CUDA_THREADS, 0, stream>>>(
        num_frags, frag_attrs, min_level, max_level, S,
        grad_alpha_src, grad_frag_alpha
    );

    CUDA_CHECK(cudaGetLastError());
}

__global__ void multires_triangle_color_kernel(
    int B, int H, int W,
    const float* __restrict__ rast,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    const uint32_t feature_dim,
    const float* features,               // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    float* __restrict__ out              // [B, H, W, feature_dim]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const int tri_id = static_cast<int>(rast[idx * 4 + 3]) - 1;
    if (tri_id < 0) {
        #pragma unroll 4
        for (uint32_t c = 0; c < feature_dim; ++c) {
            out[idx * feature_dim + c] = 0.f;
        }
        return;
    }

    const float b0 = rast[idx * 4 + 0];
    const float b1 = rast[idx * 4 + 1];

    #pragma unroll 4
    for (uint32_t c = 0; c < feature_dim; ++c) {
        out[idx * feature_dim + c] = 0.f;
    }

    multires_triangle_interp_d(
        b0, b1, min_level, max_level, feature_dim,
        &features[tri_id * S * feature_dim], &out[idx * feature_dim]
    );
}

void multires_triangle_color(
    int B, int H, int W,
    const float* __restrict__ rast,          // [B, H, W, 4]
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    const float* features,                   // [B, S, feature_dim]
    float* out,                              // [B, H, W, feature_dim]
    cudaStream_t stream
) {
    const uint32_t S = total_feats_from_levels(min_level, max_level);

    multires_triangle_color_kernel<<<CUDA_BLOCKS(B * H * W), CUDA_THREADS, 0, stream>>>(
        B, H, W, rast, min_level, max_level,
        S, feature_dim, features, out
    );

    CUDA_CHECK(cudaGetLastError());
}

__global__ void backward_multires_triangle_color_kernel(
    int B, int H, int W,
    const float* __restrict__ rast,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t S,
    const uint32_t feature_dim,
    float* grad_features,                // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_out   // [B, H, W, feature_dim]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const int tri_id = static_cast<int>(rast[idx * 4 + 3]) - 1;
    if (tri_id < 0) return;

    const float b0 = rast[idx * 4 + 0];
    const float b1 = rast[idx * 4 + 1];

    backward_multires_triangle_interp_d(
        b0, b1, min_level, max_level, feature_dim,
        &grad_features[tri_id * S * feature_dim],
        &grad_out[idx * feature_dim]
    );
}

void backward_multires_triangle_color(
    int B, int H, int W,
    const float* __restrict__ rast,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t feature_dim,
    float* grad_features,                // [T, S, feature_dim], where T is triangle count and S = Σ (2^(level - 1) + 1) * (2^level + 1)
    const float* __restrict__ grad_out,  // [B, H, W, feature_dim]
    cudaStream_t stream
) {
    const uint32_t S = total_feats_from_levels(min_level, max_level);

    backward_multires_triangle_color_kernel<<<CUDA_BLOCKS(B * H * W), CUDA_THREADS, 0, stream>>>(
        B, H, W, rast, min_level, max_level,
        S, feature_dim, grad_features, grad_out
    );

    CUDA_CHECK(cudaGetLastError());
}

// inverse of: idx = T_n + y, where n = x+y, T_n = n(n+1)/2, 0<=y<=n, x = n-y
inline __device__ void index_to_xy_on_level(uint32_t L, uint32_t idx, uint32_t& x, uint32_t& y) {
    // solve n from triangular number: T_n <= idx < T_{n+1}
    // n = floor((sqrt(8*idx+1)-1)/2)
    const float fi   = static_cast<float>(idx);
    const uint32_t n = static_cast<uint32_t>(floorf((sqrtf(8.f * fi + 1.f) - 1.f) * 0.5f));
    const uint32_t Tn = (n * (n + 1u)) >> 1; // n(n+1)/2
    y = idx - Tn;
    x = n - y;

    // (x,y) are coordinates on the level-L vertex lattice with constraint x+y<=2^L
    // nothing else to do here.
}

// The target lattice is identical for every triangle and feature channel.
// Build its sparse interpolation plan once per autograd invocation, then use a
// scalar thread mapping over (triangle, target sample, channel).  This changes
// strided C=7 accesses into scalar, coalesced accesses and avoids repeating the
// lattice decode in both forward and backward.

__global__ void build_accumulation_plan_kernel(
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t target_level,
    const uint32_t S_T,
    const uint32_t num_levels,
    int* __restrict__ plan_indices,
    float* __restrict__ plan_weights
) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= S_T) return;

    uint32_t x_target, y_target;
    index_to_xy_on_level(target_level, k, x_target, y_target);

    const float target_res = static_cast<float>(1u << target_level);
    const float b0 = static_cast<float>(x_target) / target_res;
    const float b1 = static_cast<float>(y_target) / target_res;

    uint32_t feature_offset = 0;
    for (uint32_t slot = 0; slot < num_levels; ++slot) {
        const uint32_t level = min_level + slot;
        const uint32_t res = 1u << level;
        const float res_f = static_cast<float>(res);

        float b0_level = b0 * res_f;
        float b1_level = b1 * res_f;

        const uint32_t x = MIN(static_cast<uint32_t>(floorf(b0_level)), res - 1u);
        const uint32_t y = MIN(static_cast<uint32_t>(floorf(b1_level)), res - 1u - x);

        b0_level -= static_cast<float>(x);
        b1_level -= static_cast<float>(y);

        const bool flip = b0_level + b1_level > 1.f;
        const uint32_t flip_u = static_cast<uint32_t>(flip);
        const float flip_f = static_cast<float>(flip);

        const uint32_t px[3] = {x + 1u, x, x + flip_u};
        const uint32_t py[3] = {
            y,
            y + 1u,
            MIN(y + flip_u, res - (x + flip_u)),
        };

        const float weights[3] = {
            (1.f - flip_f) * b0_level + flip_f * (1.f - b1_level),
            (1.f - flip_f) * b1_level + flip_f * (1.f - b0_level),
            0.f,
        };

        const uint32_t base = (k * num_levels + slot) * 3u;
        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t n = px[i] + py[i];
            plan_indices[base + i] = static_cast<int>(
                feature_offset + n * (n + 1u) / 2u + py[i]
            );
        }
        plan_weights[base + 0] = weights[0];
        plan_weights[base + 1] = weights[1];
        plan_weights[base + 2] = 1.f - weights[0] - weights[1];

        feature_offset += level == 0u
            ? 3u
            : (((1u << (level - 1u)) + 1u) * ((1u << level) + 1u));
    }
}

void build_accumulation_plan(
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t target_level,
    int* __restrict__ plan_indices,
    float* __restrict__ plan_weights,
    cudaStream_t stream
) {
    const uint32_t S_T = feats_at_level(target_level);
    const uint32_t num_levels = max_level - min_level + 1u;
    build_accumulation_plan_kernel<<<CUDA_BLOCKS(S_T), CUDA_THREADS, 0, stream>>>(
        min_level, max_level, target_level, S_T, num_levels,
        plan_indices, plan_weights
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void accumulate_to_level_forward_planned_kernel(
    int T,
    const uint32_t S_stride_total,
    const uint32_t S_T,
    const uint32_t num_levels,
    const uint32_t feature_dim,
    const float* __restrict__ features,
    const int* __restrict__ plan_indices,
    const float* __restrict__ plan_weights,
    float* __restrict__ f_target
) {
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t N = static_cast<uint64_t>(T) * S_T * feature_dim;
    if (tid >= N) return;

    const uint32_t c = static_cast<uint32_t>(tid % feature_dim);
    const uint64_t sample = tid / feature_dim;
    const uint32_t k = static_cast<uint32_t>(sample % S_T);
    const uint32_t t = static_cast<uint32_t>(sample / S_T);

    const uint32_t plan_base = k * num_levels * 3u;
    float value = 0.f;
    for (uint32_t slot = 0; slot < num_levels; ++slot) {
        const uint32_t base = plan_base + slot * 3u;
        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            const uint64_t src = (
                static_cast<uint64_t>(t) * S_stride_total
                + static_cast<uint32_t>(plan_indices[base + i])
            ) * feature_dim + c;
            value += plan_weights[base + i] * features[src];
        }
    }
    f_target[tid] = value;
}

void accumulate_to_level_forward(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t target_level,
    const uint32_t feature_dim,
    const float* __restrict__ features,
    const int* __restrict__ plan_indices,
    const float* __restrict__ plan_weights,
    float* __restrict__ f_target,
    cudaStream_t stream
) {
    if (T == 0 || feature_dim == 0) return;
    const uint32_t S_stride_total = total_feats_from_levels(min_level, max_level);
    const uint32_t S_T = feats_at_level(target_level);
    const uint32_t num_levels = max_level - min_level + 1u;

    const uint64_t N = static_cast<uint64_t>(T) * S_T * feature_dim;
    accumulate_to_level_forward_planned_kernel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, stream>>>(
        T, S_stride_total, S_T, num_levels, feature_dim,
        features, plan_indices, plan_weights, f_target
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void accumulate_to_level_backward_planned_kernel(
    int T,
    const uint32_t S_stride_total,
    const uint32_t S_T,
    const uint32_t num_levels,
    const uint32_t first_direct_slot,
    const uint32_t feature_dim,
    float* __restrict__ grad_features,
    const float* __restrict__ grad_f_target,
    const int* __restrict__ plan_indices,
    const float* __restrict__ plan_weights
) {
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t N = static_cast<uint64_t>(T) * S_T * feature_dim;
    if (tid >= N) return;

    const uint32_t c = static_cast<uint32_t>(tid % feature_dim);
    const uint64_t sample = tid / feature_dim;
    const uint32_t k = static_cast<uint32_t>(sample % S_T);
    const uint32_t t = static_cast<uint32_t>(sample / S_T);
    const float grad = grad_f_target[tid];

    const uint32_t plan_base = k * num_levels * 3u;
    for (uint32_t slot = 0; slot < num_levels; ++slot) {
        const uint32_t base = plan_base + slot * 3u;
        #pragma unroll
        for (uint32_t i = 0; i < 3; ++i) {
            const float weight = plan_weights[base + i];
            // Target lattice vertices produce many exact zero weights.  They
            // carry no gradient and need not contend on the atomic address.
            if (weight == 0.f) continue;
            const uint64_t dst = (
                static_cast<uint64_t>(t) * S_stride_total
                + static_cast<uint32_t>(plan_indices[base + i])
            ) * feature_dim + c;
            // At levels at least as fine as the target lattice, every target
            // vertex maps injectively to one stored vertex.  Those writes
            // cannot collide, so only coarser levels need atomic accumulation.
            if (slot >= first_direct_slot) {
                grad_features[dst] = grad * weight;
            }
            else {
                atomicAdd(&grad_features[dst], grad * weight);
            }
        }
    }
}

void accumulate_to_level_backward(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t target_level,
    const uint32_t feature_dim,
    float* __restrict__ grad_features,
    const float* __restrict__ grad_f_target,
    const int* __restrict__ plan_indices,
    const float* __restrict__ plan_weights,
    cudaStream_t stream
) {
    if (T == 0 || feature_dim == 0) return;
    const uint32_t S_stride_total = total_feats_from_levels(min_level, max_level);
    const uint32_t S_T = feats_at_level(target_level);
    const uint32_t num_levels = max_level - min_level + 1u;
    const uint32_t first_direct_slot = target_level <= min_level
        ? 0u
        : (target_level > max_level ? num_levels : target_level - min_level);

    const uint64_t N = static_cast<uint64_t>(T) * S_T * feature_dim;
    accumulate_to_level_backward_planned_kernel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, stream>>>(
        T, S_stride_total, S_T, num_levels, first_direct_slot, feature_dim,
        grad_features, grad_f_target, plan_indices, plan_weights
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void accumulate_to_level_backward_gather_kernel(
    int T,
    const uint32_t S_source,
    const uint32_t S_gather,
    const uint32_t S_target,
    const uint32_t feature_dim,
    float* __restrict__ grad_features,
    const float* __restrict__ grad_f_target,
    const int* __restrict__ reverse_offsets,
    const int* __restrict__ reverse_target_indices,
    const float* __restrict__ reverse_weights
) {
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t N = static_cast<uint64_t>(T) * S_gather * feature_dim;
    if (tid >= N) return;

    const uint32_t c = static_cast<uint32_t>(tid % feature_dim);
    const uint64_t source_row = tid / feature_dim;
    const uint32_t s = static_cast<uint32_t>(source_row % S_gather);
    const uint32_t t = static_cast<uint32_t>(source_row / S_gather);

    float value = 0.f;
    const int begin = reverse_offsets[s];
    const int end = reverse_offsets[s + 1u];
    for (int edge = begin; edge < end; ++edge) {
        const uint32_t k = static_cast<uint32_t>(reverse_target_indices[edge]);
        const uint64_t src = (
            static_cast<uint64_t>(t) * S_target + k
        ) * feature_dim + c;
        value += reverse_weights[edge] * grad_f_target[src];
    }

    const uint64_t dst = (
        static_cast<uint64_t>(t) * S_source + s
    ) * feature_dim + c;
    grad_features[dst] = value;
}

void accumulate_to_level_backward_gather(
    int T,
    const uint32_t min_level,
    const uint32_t max_level,
    const uint32_t target_level,
    const uint32_t feature_dim,
    float* __restrict__ grad_features,
    const float* __restrict__ grad_f_target,
    const int* __restrict__ reverse_offsets,
    const int* __restrict__ reverse_target_indices,
    const float* __restrict__ reverse_weights,
    cudaStream_t stream
) {
    if (T == 0 || feature_dim == 0) return;

    const uint32_t S_source = total_feats_from_levels(min_level, max_level);
    const uint32_t S_target = feats_at_level(target_level);
    uint32_t S_gather = S_source;

    if (target_level == max_level) {
        const uint32_t direct_offset = S_source - S_target;
        const size_t source_pitch = static_cast<size_t>(S_source)
            * feature_dim * sizeof(float);
        const size_t target_pitch = static_cast<size_t>(S_target)
            * feature_dim * sizeof(float);
        CUDA_CHECK(cudaMemcpy2DAsync(
            grad_features + static_cast<size_t>(direct_offset) * feature_dim,
            source_pitch,
            grad_f_target,
            target_pitch,
            target_pitch,
            static_cast<size_t>(T),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        S_gather = direct_offset;
    }

    if (S_gather == 0) return;

    const uint64_t N = static_cast<uint64_t>(T) * S_gather * feature_dim;
    accumulate_to_level_backward_gather_kernel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, stream>>>(
        T, S_source, S_gather, S_target, feature_dim,
        grad_features, grad_f_target,
        reverse_offsets, reverse_target_indices, reverse_weights
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace diffsoup
