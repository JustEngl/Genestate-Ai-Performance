import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_tensorrt_engine(onnx_path: str, engine_path: str) -> bool:
    builder = trt.Builder(TRT_LOGGER)
    
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    else:
        network = builder.create_network()

    config = builder.create_builder_config()

    if hasattr(trt.BuilderFlag, "TF32"):
        config.set_flag(trt.BuilderFlag.TF32)
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    abs_onnx_path = os.path.abspath(onnx_path)
    if not parser.parse_from_file(abs_onnx_path):
        for error in range(parser.num_errors):
            print(f"trt parser error: {parser.get_error(error)}")
        return False

    profile = builder.create_optimization_profile()
    
    # Profile ranges matching ONNX static shapes & dynamic batch sizes
    # min=(1,4,64,64), opt=(2,4,64,64), max=(2,4,128,128)
    profile.set_shape("sample", (1, 4, 64, 64), (2, 4, 64, 64), (2, 4, 128, 128))
    profile.set_shape("timestep", (1,), (1,), (1,))
    profile.set_shape("encoder_hidden_states", (1, 77, 2048), (2, 77, 2048), (2, 77, 2048))
    profile.set_shape("text_embeds", (1, 1280), (2, 1280), (2, 1280))
    profile.set_shape("time_ids", (1, 6), (2, 6), (2, 6))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return False

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    return True