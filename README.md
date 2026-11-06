


# GenEstate AI Performance Engine

A modular high-performance inference runtime designed for low-latency Stable Diffusion XL (SDXL) image generation on consumer NVIDIA GPUs (RTX 30 Series / Ampere architecture). 

This project bridges high-level PyTorch diffusion pipelines with custom native C++ and CUDA extension kernels to minimize execution latency and memory overhead.

---

## Architecture & Request Lifecycle

The diagram below illustrates how an incoming generation request flows through the engine, leveraging PyTorch execution primitives alongside native CUDA kernels for accelerated image synthesis.

```text
+-----------------------------------------------------------------------+
|                           CLIENT / USER                               |
+-----------------------------------------------------------------------+
                                   |
                                   v  (Prompt & Seed)
+-----------------------------------------------------------------------+
|                    GENESTATE ENGINE INTERFACE                         |
|                    (genestate/engine/sdxlengine.py)                   |
+-----------------------------------------------------------------------+
                                   |
               +-------------------+-------------------+
               |                                       |
               v                                       v
    [Baseline PyTorch Path]                 [CUDA Accelerated Path]
               |                                       |
               v                                       v
    Standard FP16 Execution                 Channels Last (NHWC) Layout
    Default Diffusers Kernels               TF32 Tensor Core Acceleration
               |                                       |
               |                                       v
               |                        +-------------------------------+
               |                        |   NATIVE C++/CUDA EXTENSION   |
               |                        |     (cppcuda/binding.cpp)     |
               |                        +-------------------------------+
               |                                       |
               |                                       v
               |                        +-------------------------------+
               |                        |  KERNEL 01: Fused Bias GELU   |
               |                        | (cppcuda/activation_kernel.cu)|
               |                        +-------------------------------+
               |                                       |
               +-------------------+-------------------+
                                   |
                                   v  (Generated Image & Latency Metrics)
+-----------------------------------------------------------------------+
|                             OUTPUT                                    |
|              (Rendered Image + Latency Telemetry)                     |
+-----------------------------------------------------------------------+

```

---

## Detailed System Design

### 1. Request Dispatch & Pipeline Configuration

When a generation request arrives, `SDXLEngine` configures the underlying SDXL pipeline based on the selected execution mode (`baseline` vs `cuda_optimized`). It initializes half-precision FP16 weights and configures PyTorch backends for TensorFloat-32 (TF32) matrix multiplications.

### 2. Native C++/CUDA JIT Integration

The C++ bridge is dynamically loaded at runtime using PyTorch C++ Extensions backed by Ninja and MSVC (`cl.exe`).

* **`cppcuda/binding.cpp`**: Exposes native C++ functions to the Python environment using PyBind11.
* **`cppcuda/activation_kernel.cu`**: Custom CUDA kernel executing fused Bias-GELU activation directly on GPU registers to lower memory bandwidth pressure.

### 3. VRAM & Layout Optimization

To prevent VRAM swapping on 8 GB cards (such as the RTX 3070 Ti), the engine applies smart CPU offloading alongside `channels_last` (NHWC) memory representation. This layout aligns memory access patterns directly with NVIDIA Tensor Core architectures.

---

## Project Structure

```text
Genestateai/
├── cppcuda/
│   ├── activation_kernel.cu    # Native CUDA fused activation kernel
│   └── binding.cpp             # PyBind11 C++ module definitions
├── genestate/
│   ├── config.py               # Global engine parameters
│   └── engine/
│       └── sdxlengine.py       # Core inference runner & JIT loader
├── benchmarks/
│   └── bench_inference.py      # Latency measurement & graphing suite
├── scripts/
│   └── run_benchmark.py        # Terminal execution entrypoint
└── setup.py                    # C++/CUDA extension build setup

```

---

## Getting Started

### Prerequisites

* Windows 10 / 11
* Python 3.11
* NVIDIA CUDA Toolkit 12.x
* Visual Studio 2022 Build Tools (C++ Desktop Workload)

### Installation & Execution

1. **Activate Environment & Install Dependencies**

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ninja wheel setuptools

```

2. **Run Benchmarks**

```powershell
python -m scripts.run_benchmark

```

The benchmark script evaluates execution modes, outputs latency performance tables, and generates a visual performance breakdown saved as `benchmark_results.png`.

```

