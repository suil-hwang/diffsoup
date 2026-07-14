#pragma once

#include <cuda_runtime.h>

namespace diffsoup {
namespace cuda {

void encode_view_dir_sh2(
    int B, int H, int W,
    const float* __restrict__ rast, // [B, H, W, 4]
    const float* inv_mvp,           // [B, 4, 4]
    float* __restrict__ encoding,   // [B, H, W, 9]
    cudaStream_t stream
);

} // namespace cuda
} // namespace diffsoup
