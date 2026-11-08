import torch
import torch.nn as nn
import tensorrt as trt

class TensorRTUNetWrapper(nn.Module):
    def __init__(self, engine_path: str, original_unet):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        
        # Copy original unet properties required by diffusers pipeline
        self.config = original_unet.config
        self.dtype = original_unet.dtype
        self.device = original_unet.device
        self.add_embedding = getattr(original_unet, "add_embedding", None)

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, **kwargs):
        if added_cond_kwargs is None:
            added_cond_kwargs = {}

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")

        # Set input tensor dynamic shapes
        self.context.set_input_shape("sample", tuple(sample.shape))
        self.context.set_input_shape("timestep", (1,))
        self.context.set_input_shape("encoder_hidden_states", tuple(encoder_hidden_states.shape))
        self.context.set_input_shape("text_embeds", tuple(text_embeds.shape))
        self.context.set_input_shape("time_ids", tuple(time_ids.shape))

        out_sample = torch.empty_like(sample)

        # Bind GPU VRAM data pointers directly
        self.context.set_tensor_address("sample", sample.data_ptr())
        self.context.set_tensor_address("timestep", timestep.data_ptr())
        self.context.set_tensor_address("encoder_hidden_states", encoder_hidden_states.data_ptr())
        self.context.set_tensor_address("text_embeds", text_embeds.data_ptr())
        self.context.set_tensor_address("time_ids", time_ids.data_ptr())
        self.context.set_tensor_address("out_sample", out_sample.data_ptr())

        # Execute TensorRT C++ Execution Graph
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        if not return_dict:
            return (out_sample,)
        
        class UNetOutput:
            def __init__(self, sample):
                self.sample = sample
                
        return UNetOutput(sample=out_sample)