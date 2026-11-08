import os
import sys
import torch
from diffusers import DiffusionPipeline
from genestate.config import Config
from genestate.builder.trt_builder import build_tensorrt_engine

def main():
    onnx_dir = "models/onnx"
    onnx_path = os.path.join(onnx_dir, "sdxl_unet.onnx")
    engine_path = os.path.join("models", "sdxl_unet_fp16.engine")

    os.makedirs(onnx_dir, exist_ok=True)

    if not os.path.exists(onnx_path):
        pipe = DiffusionPipeline.from_pretrained(
            Config.BASE_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        unet = pipe.unet.to("cuda")

        sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
        timestep = torch.tensor([1], dtype=torch.float16, device="cuda")
        encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float16, device="cuda")
        
        text_embeds = torch.randn(1, 1280, dtype=torch.float16, device="cuda")
        time_ids = torch.zeros((1, 6), dtype=torch.float16, device="cuda")
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

        torch.onnx.export(
            unet,
            (sample, timestep, encoder_hidden_states),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
            output_names=["out_sample"],
            dynamic_axes={
                "sample": {0: "batch_size", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch_size"},
                "text_embeds": {0: "batch_size"},
                "time_ids": {0: "batch_size"},
            },
            kwargs={"added_cond_kwargs": added_cond_kwargs}
        )
        del unet, pipe
        torch.cuda.empty_cache()

    build_tensorrt_engine(onnx_path, engine_path)

if __name__ == "__main__":
    main()