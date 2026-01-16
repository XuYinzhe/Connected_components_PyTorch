#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

namespace cc_torch {

// --------------------------
// Device helper functions
// --------------------------

template <typename T>
__device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}

__device__ int32_t find(const int32_t *s_buf, int32_t n);
__device__ int32_t find_n_compress(int32_t *s_buf, int32_t n);
__device__ void union_(int32_t *s_buf, int32_t a, int32_t b);

// --------------------------
// 2D Kernels
// --------------------------

__global__ void init_labeling_2d(int32_t *label, uint32_t W, uint32_t H);
__global__ void merge_2d(uint8_t *img, int32_t *label, uint32_t W, uint32_t H);
__global__ void compression_2d(int32_t *label, int32_t W, int32_t H);
__global__ void final_labeling_2d(const uint8_t *img, int32_t *label, int32_t W, int32_t H);

// --------------------------
// 3D Kernels
// --------------------------

__global__ void init_labeling_3d(int32_t *label, uint32_t W, uint32_t H, uint32_t D);
__global__ void merge_3d(uint8_t *img, int32_t *label, uint8_t *last_cube_fg,
                         uint32_t W, uint32_t H, uint32_t D);
__global__ void compression_3d(int32_t *label, uint32_t W, uint32_t H, uint32_t D);
__global__ void final_labeling_3d(int32_t *label, uint8_t *last_cube_fg,
                                  uint32_t W, uint32_t H, uint32_t D);

// --------------------------
// Host callable functions
// --------------------------

torch::Tensor connected_components_labeling_2d(const torch::Tensor &input);
torch::Tensor connected_components_labeling_3d(const torch::Tensor &input);

} // namespace cc_torch