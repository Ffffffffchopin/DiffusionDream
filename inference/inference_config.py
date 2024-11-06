#import os
import torch


class InferenceConfig:

    def __init__(self):

        self.input_image_path = "input.jpg"
        self.output_image_path = "output.jpg"
        self.prompt = "Zoom into the image"
        self.action = "1,0,0.0,0.0"
        self.input_image_url = "https://www.helloimg.com/i/2024/10/20/6714d43670bd9.png"
        self.seed = 0
        self.num_inference_steps = 4
        self.image_guidance_scale = 1.5
        self.guidance_scale = 7.5
        self.torch_dtype = torch.float16
        self.model_path = "fffffchopin/Instruct_pix2pix"
        self.scheduler_class = "EulerAncestralDiscreteScheduler"
        self.vae_class = "AutoencoderKL"
        self.inference_with_TensorRT = False
        self.onnx_dir_path = "onnx_path"
        self.engine_dir_path = "engine_path"
        self.onnx_opt_dir_path = "onnx_opt_path"
        self.opset_version = 19
        self.static_shape = True
        self.image_height = 512
        self.image_width = 512
        self.int8 = False
        self.static_batch = False
        self.use_cuda_graph = False