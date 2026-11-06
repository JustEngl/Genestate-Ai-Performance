#include <torch/extension.h>

// Forward declaration of the CUDA launcher function
torch::Tensor launch_fused_bias_gelu(torch::Tensor input, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_gelu", &launch_fused_bias_gelu, "Fused Bias Add and GELU activation (CUDA native)");
}