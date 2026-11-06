import logging
import random
import time
import torch
from diffusers import DiffusionPipeline
from genestate.config import Config
from kernels.fused_ops import TritonEngine

logger = logging.getLogger(__name__)

class SDXLEngine:
    def __init__(self, use_triton: bool = False):
        self.pipe = None
        self.use_triton = use_triton
        self.triton_engine = TritonEngine(enabled=use_triton)

    def load_model(self):
        logger.info(f"loading model with triton={self.use_triton}...")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.pipe = DiffusionPipeline.from_pretrained(
            Config.BASE_MODEL,
            torch_dtype=Config.DTYPE,
            variant="fp16",
            use_safetensors=True,
        )

        self.pipe.enable_model_cpu_offload()
        self.pipe.unet.to(memory_format=torch.channels_last)
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
        
        # cuda sync timer
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