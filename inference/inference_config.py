#import os
import torch


class InferenceConfig:

    def __init__(self):

        self.input_image_path = "car.jpg"
        self.output_image_path = "output.jpg"

        self.prompt = "1,-1,-0.0294,0.0629"
        #self.prompt = "make the river glow"
        #self.prompt = "Portrait shot of a woman, yellow shirt, photograph"

        self.action = "1,0,0.0,0.0"
        self.input_image_url = "https://www.helloimg.com/i/2024/10/20/6714d43670bd9.png"
        self.seed = 0

        self.num_inference_steps = 1

        self.image_guidance_scale = 0.0
        self.guidance_scale = 0.0
        self.torch_dtype = torch.float16
        self.model_path = "E:\\models\\fffchopin_instruct_pix2pix"
        #self.model_path = "E:\\models\\instruct_pix2pix"
        self.scheduler_class = "EulerAncestralDiscreteScheduler"
        self.vae_class = "AutoencoderKL"

        self.inference_with_TensorRT = True
        #self.inference_with_TensorRT = False

        self.onnx_dir_path = "onnx_path"
        self.engine_dir_path = "engine_path"
        self.onnx_opt_dir_path = "onnx_opt_path"
        self.opset_version = 19
        self.static_shape = True
        self.image_height = 512
        self.image_width = 512
        self.int8 = True
        self.static_batch = True
        self.use_cuda_graph = True

        self.calibration_prompts_path = "calibration-actions.txt"
        self.pipeline_class = "StableDiffusionInstructPix2PixPipeline"
        #self.inference_with_onnxruntime = True