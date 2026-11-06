#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel execution layout
__global__ void fused_bias_gelu_kernel(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, int size, int bias_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] + bias[idx % bias_size];
        // Fast GELU approximmation
        output[idx] = 0.5f * val * (1.0f + tanhf(0.7978845608028654f * (val + 0.044715f * val * val * val)));
    }
}

// C++ launcher wrapper for pytorch integration
torch::Tensor launch_fused_bias_gelu(torch::Tensor input, torch::Tensor bias) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int bias_size = bias.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fused_bias_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        bias_size
    );

    return output;
}