import os
import torch

class Config:
    BASE_MODEL: str = "Lykon/dreamshaper-xl-v2-turbo"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Inference Defaults
    NUM_INFERENCE_STEPS: int = 12
    GUIDANCE_SCALE: float = 3.0
    HEIGHT: int = 864
    WIDTH: int = 864

    