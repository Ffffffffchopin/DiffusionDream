import io
import requests
import PIL
#from inference_config import InferenceConfig
import torch
import os
import json
from typing import Union
import re
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from percentile_calibrator import PercentileCalibrator
from diffusers.models.attention_processor import Attention
#import onnxruntime






def get_image(url,save_path,download=True):
    #image = requests.get(url, stream=True).raw
    #image = io.BytesIO(image)
    if download:
        response = requests.get(url)
        byte = io.BytesIO(response.content)
        image = PIL.Image.open(byte)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.save(save_path)
    else:
        image = PIL.Image.open(save_path)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")
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
        text_input_ids = text_input_ids.to(device="cuda",dtype=torch.int32)
        prompt_embeds = runEngine(engine_name,{'input_ids': text_input_ids},stream,use_cuda_graph)
        prompt_embeds = prompt_embeds['text_embeddings'].clone()
    else:
        prompt_embeds = text_encoder(text_input_ids.to(device="cuda",dtype=torch.int32),)[0]
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
        image_latents = image_latents.latent_dist.sample()
        #print(f"image_latents_dist_sample:{image_latents.shape}")

    
    elif hasattr(image,"latents"):
        #print(f"image_latents:{image_latents.shape}")
        image_latents = image_latents.latents
    else:
        image_latents = image_latents
        #print(f"image:{image_latents.shape}")
        
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


def load_calib_prompts(batch_size, calib_data_path):
    with open(calib_data_path, "r", encoding="utf-8") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|proj_out).*"
    )
    return pattern.match(name) is not None


def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="min-mean",
):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
        elif isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config

def quantize_lvl(unet, quant_level=2.5, linear_only=False):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            if linear_only:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
            else:
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level >= 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, Attention):
            # TRT only supports FP8 MHA with head_size % 16 == 0.
            head_size = int(module.inner_dim / module.heads)
            if quant_level >= 4 and head_size % 16 == 0:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()


if __name__ == "__main__":
    ret = load_calib_prompts(2,"calibration-prompts.txt")
    print(ret[0])
