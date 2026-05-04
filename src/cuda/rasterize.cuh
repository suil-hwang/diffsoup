// src/cuda/rasterize.cuh
// CUDA declarations for software rasterisation, depth testing, edge gradients,
// and the stochastic opacity masking auxiliary loss backward pass.

#pragma once

namespace diffsoup {
namespace cuda {

int compute_triangle_rects(
    int H, int W, int B,
    int V, const float* pos,     // [B * V][4]
    int T, const int* tri,       // [T][3]
    int* triangle_rects,         // [B * T][4]: h0, h_len, w0, w_len
    int* frag_prefix_sum         // [B * T]
);

void compute_fragments(
    int H, int W,
    int V, const float* pos,
    int T, const int* tri,
    int num_tris,
    int num_frags,
    const int* frag_prefix_sum,
    const int* triangle_rects,
    int* frag_pix,
    float* frag_attrs
);

void depth_test(
    int B, int H, int W,
    int num_frags,
    const int* frag_pix,       // [num_frags, 3]
    const float* frag_attrs,   // [num_frags, 4]
    const float* frag_alpha,   // [num_frags]
    const float* alpha_thresh, // [num_frags]
    float* rast_out            // [B, H, W, 4]
);

int filter_valid_fragments(
    int num_frags,
    const int* frag_pix,         // [num_frags, 3]
    const float* frag_attrs,     // [num_frags, 4]
    int* frag_pix_out,           // [num_frags, 3]
    float* frag_attrs_out        // [num_frags, 4]
);

void backward_edge_grad(
    int B, int H, int W, int C,
    const float* color,                      // [B, H, W, C]
    const float* grad_color,                 // [B, H, W, C]
    const float* rast,                       // [B, H, W, 4]
    int V,
    const float* pos,                        // [B, V, 4]
    float* grad_pos,                         // [B, V, 4]
    const int* tri                           // [T, 3]
);

void backward_opacity_aux_loss(
    int B, int H, int W, int C,
    const float* color,                      // [B, H, W, C]
    const float* target,                     // [B, H, W, C]
    const float* rast,                       // [B, H, W, 4]
    int num_frags,
    const int* frag_pix,                     // [num_frags, 3]
    const float* frag_attrs,                 // [num_frags, 4]
    const float* frag_alpha,                 // [num_frags]
    float* grad_frag_alpha                   // [num_frags]
);

} // namespace cuda
} // namespace diffsoup
