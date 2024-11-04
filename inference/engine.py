from collections import OrderedDict
import tensorrt as trt
import torch
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine
)
from polygraphy.logger import G_LOGGER
from polygraphy.backend.common import bytes_from_path
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt_to_torch_dtype_dict = {
    trt.DataType.BOOL     : torch.bool,
    trt.DataType.UINT8    : torch.uint8,
    trt.DataType.INT8     : torch.int8,
    trt.DataType.INT32    : torch.int32,
    trt.DataType.INT64    : torch.int64,
    trt.DataType.HALF     : torch.float16,
    trt.DataType.FLOAT    : torch.float32,
    trt.DataType.BF16     : torch.bfloat16
}

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class Engine():    
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, refit_weights, is_fp16):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)

        refitted_weights = set()
        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            if trt_weight_name not in refit_weights:
                continue

            # get weight from state dict
            trt_datatype = trt.DataType.FLOAT
            if is_fp16:
                refit_weights[trt_weight_name] = refit_weights[trt_weight_name].half()
                trt_datatype = trt.DataType.HALF

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(trt_datatype, refit_weights[trt_weight_name].data_ptr(), torch.numel(refit_weights[trt_weight_name]))
            trt_wt_location = trt.TensorLocation.DEVICE if refit_weights[trt_weight_name].is_cuda else trt.TensorLocation.HOST

            # apply refit
            refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        assert set(refitted_weights) == set(refit_weights.keys())
        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(self,
        onnx_path,
        strongly_typed=False,
        fp16=True,
        bf16=False,
        tf32=False,
        int8=False,
        fp8=False,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
        native_instancenorm=True,
        verbose=False,
        **extra_build_args
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        if not enable_all_tactics:
            extra_build_args['tactic_sources'] = []

        flags = []
        if native_instancenorm:
            flags.append(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
        network = network_from_onnx_path(
            onnx_path,
            flags=flags,
            strongly_typed=strongly_typed
        )
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        with G_LOGGER.verbosity(G_LOGGER.EXTRA_VERBOSE if verbose else G_LOGGER.ERROR):
            engine = engine_from_network(
                network,
                config=CreateConfig(fp16=fp16,
                    bf16=bf16,
                    tf32=tf32,
                    int8=int8,
                    fp8=fp8,
                    refittable=enable_refit,
                    profiles=[p],
                    load_timing_cache=timing_cache,
                    **extra_build_args
                ),
                save_timing_cache=timing_cache
            )
            save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, device_memory=None):
        if device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = device_memory
        else:
            self.context = self.engine.create_execution_context()

    def reactivate(self, device_memory):
        assert self.context
        self.context.device_memory = device_memory

    def deactivate(self):
        del self.context
        self.context = None

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            dtype=trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            tensor = torch.empty(tuple(shape), dtype=dtype).to(device=device)
            self.tensors[name] = tensor


    def deallocate_buffers(self):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            del self.tensors[binding]

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        #print(f"feed_dict:{feed_dict.keys()}")
        for name, buf in feed_dict.items():
            #if name == "sample":
                #print(f"sample={buf.shape}")
            #try:
            self.tensors[name].copy_(buf)
            #except:
                #print(f"ERROR: {name} {buf.shape} {self.tensors[name].shape} {feed_dict.keys()},{self.engine_path}")
                #raise ValueError(f"ERROR: {name} {buf.shape} {self.tensors[name].shape}")
                #os._exit(0)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                CUASSERT(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                self.context.execute_async_v3(stream)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            #print(f"stream:{stream.__class__}")
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors