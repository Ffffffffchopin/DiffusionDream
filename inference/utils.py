import io
import requests
import PIL
#from inference_config import InferenceConfig
import torch
import os
import json
from typing import Union
from diffusers import UNet2DConditionModel
import importlib
from transformers import CLIPTokenizer,CLIPTextModel,CLIPImageProcessor



def download_image(url,save_path):
    #image = requests.get(url, stream=True).raw
    #image = io.BytesIO(image)
    response = requests.get(url)
    byte = io.BytesIO(response.content)
    image = PIL.Image.open(byte)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.save(save_path)
    return image

def get_generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed)

def _dict_from_json_file(json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

def get_pipeline_config(model_path):
    config_path = os.path.join(model_path,"model_index.json")
    config = _dict_from_json_file(config_path)
    config.pop("_ignore_files", None)
    return config

def get_unet_model(model_path,torch_dtype):
    unet_path = os.path.join(model_path,"unet")
    model = UNet2DConditionModel.from_pretrained(unet_path,use_safetensors=True,torch_dtype=torch_dtype,low_cpu_mem_usage=True).to("cuda")
    model.eval()

    return model

def get_scheduler(model_path,scheduler_class):
    scheduler_path = os.path.join(model_path,"scheduler")
    module = importlib.import_module("diffusers")
    scheduler = getattr(module,scheduler_class).from_pretrained(scheduler_path)
    #scheduler = scheduler.to("cuda")

    return scheduler

def get_vae(model_path,vae_class):
    vae_path = os.path.join(model_path,"vae")
    module = importlib.import_module("diffusers")
    vae = getattr(module,vae_class).from_pretrained(vae_path,torch_dtype=torch.float16,use_safetensors=True,low_cpu_mem_usage=False).to("cuda")
    #vae = AutoencoderKL.from_pretrained(vae_path,torch_dtype=torch.float16,use_safetensors=True,low_cpu_mem_usage=True).to("cuda")
    vae.eval()
    return vae

def get_tokenizer(model_path):
    tokenizer_path = os.path.join(model_path,"tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    #tokenizer.eval()
    #tokenizer = tokenizer.to("cuda")
    return tokenizer

def get_text_encoder(model_path):
    text_encoder_path = os.path.join(model_path,"text_encoder")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path,torch_dtype=torch.float16,use_safetensors=True,low_cpu_mem_usage=True).to("cuda")
    text_encoder.eval()
    return text_encoder

def get_feature_extractor(model_path):
    feature_extractor_path = os.path.join(model_path,"feature_extractor")
    feature_extractor = CLIPImageProcessor.from_pretrained(feature_extractor_path)
    return feature_extractor

def encode_prompt(prompt,tokenizer,text_encoder,do_classifier_free_guidance):
    text_inputs = tokenizer(prompt,return_tensors="pt",padding="max_length",max_length=tokenizer.model_max_length,truncation=True)
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to("cuda"),)[0]
    prompt_embeds_dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device="cuda")
    if do_classifier_free_guidance:
        uncond_tokens = [""]
        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(uncond_tokens,padding="max_length",max_length=max_length,truncation=True,return_tensors="pt",)
        negative_prompt_embeds = text_encoder(uncond_input.input_ids.to("cuda"),)[0]
        #seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device="cuda")
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])

    return prompt_embeds

def prepare_image_latents(image,vae,do_classifier_free_guidance):
    image = image.to(device="cuda",dtype=torch.float16)
    image = vae.encode(image)
    '''
    try:
        image_latents = image.latents
    except:
        print(hasattr(image, "latents"))
        os._exit(0)
    '''
    if hasattr(image,"latent_dist"):
        image_latents = image.latent_dist.mode()
    else:
        image_latents = image.latents
    image_latents = torch.cat([image_latents], dim=0)
    if do_classifier_free_guidance:
        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
    return image_latents

def randn_tensor(shape,generator,dtype):
    #batch_size = shape[0]
    layout = torch.strided
    latents = torch.randn(shape, generator=generator, device="cuda", dtype=dtype, layout=layout).to("cuda")
    return latents

def prepare_latents(height, width,num_channels_latents,generator,dtype,vae_scale_factor,scheduler):
    shape = (1,num_channels_latents,int(height) // vae_scale_factor, int(width) // vae_scale_factor,)
    latents = randn_tensor(shape,generator,dtype)
    latents = latents * scheduler.init_noise_sigma
    return latents

def get_clip_onnx(model,onnx_path,onnx_opt_path,onnx_opset,static_shape):
    if not os.path.exists(os.path.join(onnx_opt_path,"clip.onnx")):
        if not os.path.exists(os.path.join(onnx_path,"clip.onnx")):
            with torch.inference_mode(), torch.autocast("cuda"):
                input = torch.zeros(1,77,dtype=torch.float32,device="cuda")
                
        
    



