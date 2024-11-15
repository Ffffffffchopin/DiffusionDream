import io
import requests
import PIL
#from inference_config import InferenceConfig
import torch
import os
import json
from typing import Union






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



def encode_prompt(prompt,tokenizer,text_encoder,do_classifier_free_guidance,inference_with_TensorRT,engine_name,stream,use_cuda_graph):
    text_inputs = tokenizer(prompt,return_tensors="pt",padding="max_length",max_length=tokenizer.model_max_length,truncation=True)
    text_input_ids = text_inputs.input_ids
    if inference_with_TensorRT:
        text_input_ids = text_input_ids.to("cuda")
        prompt_embeds = runEngine(engine_name,{'input_ids': text_input_ids},stream,use_cuda_graph)
        prompt_embeds = prompt_embeds['text_embeddings'].clone()
    else:
        prompt_embeds = text_encoder(text_input_ids.to("cuda"),)[0]
    prompt_embeds_dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device="cuda")
    if do_classifier_free_guidance:
        uncond_tokens = [""]
        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(uncond_tokens,padding="max_length",max_length=max_length,truncation=True,return_tensors="pt",)
        if inference_with_TensorRT:
            negative_prompt_embeds = runEngine(engine_name,{'input_ids': uncond_input.input_ids.to("cuda")},stream,use_cuda_graph)
            negative_prompt_embeds = negative_prompt_embeds['text_embeddings'].clone()
        else:
            negative_prompt_embeds = text_encoder(uncond_input.input_ids.to("cuda"),)[0]
        #seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device="cuda")
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])

    return prompt_embeds

def prepare_image_latents(image,vae,do_classifier_free_guidance,inference_with_TensorRT,engine_name,stream,use_cuda_graph):
    image = image.to(device="cuda",dtype=torch.float16)
    if inference_with_TensorRT:
        image_latents = runEngine(engine_name,{'images': image},stream,use_cuda_graph)
        #print(f"image_latents:{image_latents.keys()}")
        image_latents = image_latents['latent']
        #print(f"image_latents:{image_latents.shape}")
    else:
        image_latents = vae.encode(image)
    '''
    try:
        image_latents = image.latents
    except:
        print(hasattr(image, "latents"))
        os._exit(0)
    '''
    if hasattr(image_latents,"latent_dist"):
        image_latents = image_latents.latent_dist.mode()
    elif hasattr(image,"latents"):
        image_latents = image_latents.latents
    else:
        image_latents = image_latents
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

def runEngine(engine_name, feed_dict, stream, use_cuda_graph):
        return engine_name.infer(feed_dict, stream, use_cuda_graph=use_cuda_graph)


       
        

       
       
                



        
                
        
    




