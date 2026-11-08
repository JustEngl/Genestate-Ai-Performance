import logging
import os
import random
import sys
import time
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from torch.utils.cpp_extension import load
from genestate.config import Config
from genestate.builder.trt_wrapper import TensorRTUNetWrapper

logger = logging.getLogger(__name__)

# msvc compiler setup for windows JIT
if sys.platform == "win32":
    msvc_bin = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
    if os.path.exists(msvc_bin) and msvc_bin not in os.environ["PATH"]:
        os.environ["PATH"] = msvc_bin + os.path.pathsep + os.environ["PATH"]

# native cpp/cuda jit loader
CUDA_EXT_AVAILABLE = False
genestate_cuda_ops = None

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    genestate_cuda_ops = load(
        name="genestate_cuda_ops",
        sources=[
            os.path.join(root_dir, "cppcuda", "binding.cpp"),
            os.path.join(root_dir, "cppcuda", "activation_kernel.cu"),
        ],
        verbose=False
    )
    CUDA_EXT_AVAILABLE = True
    print(" Native C++/CUDA JIT Extension loaded successfully via MSVC & Ninja")
except Exception as e:
    print(f" c++/cuda jit fallback: {e} ")


class SDXLEngine:
    def __init__(self, mode: str = "baseline"):
        self.pipe = None
        self.mode = mode

    def load_model(self):
        logger.info(f"loading model in mode: {self.mode}...")
        
        # TF32 execution on RTX 3070 Ti Ampere Tensor Cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.pipe = DiffusionPipeline.from_pretrained(
            Config.BASE_MODEL,
            torch_dtype=Config.DTYPE,
            variant="fp16",
            use_safetensors=True,
        )

        # Fast DPM++ 2M Karras Scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )

        if self.mode == "tensorrt":
            engine_path = os.path.join("models", "sdxl_unet_fp16.engine")
            if os.path.exists(engine_path):
                logger.info(f"binding compiled tensorrt c++ engine from {engine_path}")
                self.pipe.unet = TensorRTUNetWrapper(engine_path, self.pipe.unet)
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_vae_slicing()
            else:
                logger.warning(f"engine binary '{engine_path}' not found, falling back to pytorch")
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_vae_slicing()

        elif self.mode == "cuda_optimized":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.unet.to(memory_format=torch.channels_last)

        else:  # baseline
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

        self._warmup()

    def _warmup(self):
        generator = torch.Generator(device="cpu").manual_seed(42)
        with torch.inference_mode():
            _ = self.pipe(
                "modern house design",
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                width=Config.WIDTH,
                height=Config.HEIGHT,
                generator=generator
            ).images[0]

    def generate(self, prompt: str, negative_prompt: str, seed: int = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                width=Config.WIDTH,
                height=Config.HEIGHT,
                generator=generator,
            ).images[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        return image, latency