import os
import importlib
from diffusers import UNet2DConditionModel
from transformers import CLIPTokenizer,CLIPTextModel,CLIPImageProcessor
import torch

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