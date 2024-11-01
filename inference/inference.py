from inference_config import InferenceConfig
import torch
from utils import (
    download_image,
    get_generator,
    #get_pipeline_config,
    get_unet_model,
    get_scheduler,
    get_vae,
    get_tokenizer,
    #get_feature_extractor,
    encode_prompt,
    prepare_image_latents,
    prepare_latents,
    get_text_encoder
    )

from diffusers.image_processor import VaeImageProcessor

import time




#设置允许使用tf32以提高性能
torch.backends.cuda.matmul.allow_tf32 = True

def run_inference():

    
    with torch.no_grad():
        inference_config = InferenceConfig()
    

        do_classifier_free_guidance = inference_config.guidance_scale > 1.0 and inference_config.image_guidance_scale >= 1.0
    
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

        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        if inference_config.inference_with_TensorRT:
            print("使用TensorRT推理")
            

        #开始执行推理
        print("开始执行推理")

        start_time = time.perf_counter()

        prompt_embeds = encode_prompt(inference_config.prompt,tokenizer,text_encoder,do_classifier_free_guidance)

        image = image_processor.preprocess(input_image)

        scheduler.set_timesteps(inference_config.num_inference_steps,device="cuda")

        timesteps = scheduler.timesteps

        image_latents = prepare_image_latents(image,vae,do_classifier_free_guidance)

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
            noise_pred = unet(scaled_latent_model_input,t,encoder_hidden_states=prompt_embeds,return_dict=False,)[0]
            if do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (noise_pred_uncond + inference_config.guidance_scale * (noise_pred_text - noise_pred_image) + inference_config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
)
            latents = scheduler.step(noise_pred, t, latents,return_dict=False)[0]
    
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]

        image = image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)

        print(f"推理{inference_config.num_inference_steps}步用时: ", time.perf_counter() - start_time)
        return image[0]
    
        

if __name__ == "__main__":

    ret = run_inference()
    ret.save("output.jpg")
