import logging
import random
import time
import torch
from diffusers import DiffusionPipeline
from genestate.config import Config

logger = logging.getLogger(__name__)

class SDXLEngine:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        logger.info(f"Loading {Config.BASE_MODEL} with FP16 & CPU Offloading...")
        
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
        logger.info("Warming up GPU execution pipeline...")
        generator = torch.Generator(device="cpu").manual_seed(42)
        with torch.inference_mode():
            _ = self.pipe(
                "modern house architecture design",
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                width=Config.WIDTH,
                height=Config.HEIGHT,
                generator=generator
            ).images[0]
        logger.info("--> PIPELINE WARMUP COMPLETE <--")

    def generate(self, prompt: str, negative_prompt: str, seed: int = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        t0 = time.time()
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
        
        latency = time.time() - t0
        logger.info(f"Inference execution took: {latency:.2f}s")
        return image, latency