from inference_config import InferenceConfig
import torch
from utils import (
    download_image,
    get_generator,
    #get_pipeline_config,
    #get_feature_extractor,
    encode_prompt,
    prepare_image_latents,
    prepare_latents,
    )
from models import (
    get_unet_model,
    get_scheduler,
    get_vae,
    get_tokenizer,
    #get_feature_extractor,
    get_text_encoder,
    )
from onnx_utils import (
    get_clip_onnx,
    get_unet_onnx,
    get_vae_encoder_onnx,
    get_vae_decoder_onnx, 
 )
from engine_utils import (
    get_unet_engine,
    get_vae_encoder_engine,
    get_vae_decoder_engine,
    get_clip_engine,
)
from cuda import cudart


from diffusers.image_processor import VaeImageProcessor

import time

import os
import torchvision.transforms as transforms




#设置允许使用tf32以提高性能
torch.backends.cuda.matmul.allow_tf32 = True

def run_inference():

    
    with torch.no_grad():
        inference_config = InferenceConfig()
    

        do_classifier_free_guidance = inference_config.guidance_scale > 1.0 and inference_config.image_guidance_scale >= 1.0

        print(f"使用分类器自由引导: {do_classifier_free_guidance}")
    
        input_image = download_image(inference_config.input_image_url,inference_config.input_image_path)
        generator = get_generator(inference_config.seed)
    #config_dict = get_pipeline_config(inference_config.model_path)
    
    #加载模型组件

        unet = get_unet_model(inference_config.model_path,inference_config.torch_dtype)

        scheduler = get_scheduler(inference_config.model_path,inference_config.scheduler_class)

        vae = get_vae(inference_config.model_path,inference_config.vae_class)

        tokenizer = get_tokenizer(inference_config.model_path)

        text_encoder = get_text_encoder(inference_config.model_path)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor,do_normalize=True)

        if inference_config.inference_with_TensorRT:

            print("使用TensorRT推理")

            #获取onnx模型
            get_clip_onnx(text_encoder,inference_config.onnx_dir_path,inference_config.onnx_opt_dir_path,inference_config.opset_version)

            get_unet_onnx(unet,inference_config.onnx_dir_path,inference_config.onnx_opt_dir_path,inference_config.opset_version,inference_config.static_shape,inference_config.image_height,inference_config.image_width,do_classifier_free_guidance,inference_config.int8)

            get_vae_encoder_onnx(vae,inference_config.onnx_dir_path,inference_config.onnx_opt_dir_path,inference_config.opset_version,inference_config.image_height,inference_config.image_width,inference_config.int8)

            vae_decoder = vae

            vae_decoder.forward = vae.decode

            torch.cuda.empty_cache()

            get_vae_decoder_onnx(vae_decoder,inference_config.onnx_dir_path,inference_config.onnx_opt_dir_path,inference_config.opset_version,inference_config.image_height,inference_config.image_width,inference_config.int8)

            #构建TensorRT引擎

            unet_engine = get_unet_engine(inference_config.engine_dir_path,inference_config.onnx_opt_dir_path,inference_config.int8,inference_config.static_batch,inference_config.image_height,inference_config.image_width,inference_config.static_shape,do_classifier_free_guidance)

            vae_encoder_engine = get_vae_encoder_engine(inference_config.engine_dir_path,inference_config.onnx_opt_dir_path,inference_config.int8,inference_config.image_height,inference_config.image_width,inference_config.static_batch)

            vae_decoder_engine = get_vae_decoder_engine(inference_config.engine_dir_path,inference_config.onnx_opt_dir_path,inference_config.int8,inference_config.image_height,inference_config.image_width,inference_config.static_batch,vae)

            clip_engine = get_clip_engine(inference_config.engine_dir_path,inference_config.onnx_opt_dir_path,inference_config.int8,inference_config.static_batch)

            engines = [vae_encoder_engine,clip_engine,unet_engine,vae_decoder_engine,]

            max_device_memory = 0

            for engine_name in engines:
                engine_name.load()
                max_device_memory = max(max_device_memory, engine_name.engine.device_memory_size)
            
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)

            for engine_name in engines:
                engine_name.activate(device_memory=shared_device_memory)
            
            events = {} 
            

            for stage in ['clip', 'denoise', 'vae', 'vae_encoder', 'vqgan']:
                events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]

            stream = cudart.cudaStreamCreate()[1] 

            clip_engine.allocate_buffers(shape_dict={'input_ids': (1, 77),'text_embeddings': (1, 77, 768)},device='cuda')

            xB =  2 if inference_config.do_classifier_free_guidance else 1 

            unet_engine.allocate_buffers(shape_dict={'sample': (xB, 8, inference_config.image_height, inference_config.image_width),'encoder_hidden_states':(xB,77,768),'latent': (xB,4,inference_config.image_height,inference_config.image_width)},device='cuda')

            vae_encoder_engine.allocate_buffers(shape_dict={'image': (1, 3, inference_config.image_height, inference_config.image_width),'latent': (1,4,inference_config.image_height, inference_config.image_width)},device='cuda')

            vae_decoder_engine.allocate_buffers(shape_dict={'latent': (1, vae.config['latent_channels'], inference_config.image_height, inference_config.image_width),'images':(1,3,inference_config.image_height, inference_config.image_width)},device='cuda')

        else:
            stream = None
            clip_engine = None
            unet_engine = None
            vae_encoder_engine = None
            vae_decoder_engine = None


        #开始执行推理
        print("开始执行推理")

        start_time = time.perf_counter()

        prompt_embeds = encode_prompt(inference_config.action,tokenizer,text_encoder,do_classifier_free_guidance,inference_config.inference_with_TensorRT,clip_engine,stream,inference_config.use_cuda_graph)

        #image = image_processor.preprocess(input_image,height=inference_config.image_height,width=inference_config.image_width)

        print(f"输入图像形状: {input_image.size}")

        image = image_processor.preprocess(input_image,width=inference_config.image_width,height=inference_config.image_height)

        '''

        tmp_image = image.squeeze(0)

        to_pil = transforms.ToPILImage()

        tmp_image = to_pil(tmp_image)

        print(f"图像类型: {image.__class__}")

        tmp_image.save("debug.jpg")

        #os._exit(0)

        '''

        scheduler.set_timesteps(inference_config.num_inference_steps,device="cuda")

        timesteps = scheduler.timesteps

        image_latents = prepare_image_latents(image,vae,do_classifier_free_guidance,inference_config.inference_with_TensorRT,vae_encoder_engine,stream,inference_config.use_cuda_graph)

        height, width = image_latents.shape[-2:]
        height = height * vae_scale_factor
        width = width * vae_scale_factor
        num_channels_latents = vae.config.latent_channels

    #初始化噪声
        latents = prepare_latents(height, width,num_channels_latents,generator,inference_config.torch_dtype,vae_scale_factor,scheduler)

    #num_timesteps = len(timesteps)

        for i, t in enumerate(timesteps):
            #latent_model_input = latents
            latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
            scaled_latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)
            if inference_config.inference_with_TensorRT:
                noise_pred = unet_engine.infer({'sample': scaled_latent_model_input,'encoder_hidden_states': prompt_embeds,'timestep': t},stream,inference_config.use_cuda_graph)['latent']
            else:
                noise_pred = unet(scaled_latent_model_input,t,encoder_hidden_states=prompt_embeds,return_dict=False,)[0]
            if do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (noise_pred_uncond + inference_config.guidance_scale * (noise_pred_text - noise_pred_image) + inference_config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
)
            latents = scheduler.step(noise_pred, t, latents,return_dict=False)[0]
        if inference_config.inference_with_TensorRT:
            latents = vae_decoder_engine.infer({'latent': latents},stream,inference_config.use_cuda_graph)['images']
        else:
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            #image = vae.decode(latents , return_dict=False)[0] 
        do_denormalize = [True] * image.shape[0]

        image = image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)

        print(f"推理{inference_config.num_inference_steps}步用时: ", time.perf_counter() - start_time)
        return image[0]
    
        

if __name__ == "__main__":

    ret = run_inference()
    ret.save("output.jpg")
