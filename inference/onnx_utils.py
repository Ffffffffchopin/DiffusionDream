import torch
import os
import onnx
from optimizer import Optimizer
import importlib
from utils import load_calib_prompts,get_image,get_int8_config,quantize_lvl,filter_func
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

def get_clip_onnx(model,onnx_path,onnx_opt_path,onnx_opset):
    if not os.path.exists(os.path.join(onnx_opt_path,"clip.onnx")):
        if not os.path.exists(os.path.join(onnx_path,"clip.onnx")):
            with torch.inference_mode(), torch.autocast("cuda"):
                input = torch.zeros(1,77,dtype=torch.float16,device="cuda")
                torch.onnx.export(model,input,os.path.join(onnx_path,"clip.onnx"),export_params=True,opset_version=onnx_opset,do_constant_folding=True,input_names=['input_ids'],output_names=['text_embeddings'],)

        opt = Optimizer(onnx.load(os.path.join(onnx_path,"clip.onnx")),verbose=False)
        keep_outputs = [0]
        opt.select_outputs(keep_outputs)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.select_outputs(keep_outputs, names=["text_embeddings"])
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        if onnx_opt_graph.ByteSize() > 2147483648:
            onnx.save_model(onnx_opt_graph,os.path.join(onnx_opt_path,"clip.onnx"),save_as_external_data=True,all_tensors_to_one_file=True,convert_attribute=False)
        else:
            onnx.save(onnx_opt_graph,os.path.join(onnx_opt_path,"clip.onnx") )

def get_unet_onnx(model,onnx_path,onnx_opt_path,onnx_opset,static_shape,image_height,image_width,int8,model_path,pipeline_class,calibration_prompts_path,input_image_url,save_path,denoising_steps,do_classifier_free_guidance,generator):
    if not os.path.exists(os.path.join(onnx_opt_path,"unet.onnx")):
        if not os.path.exists(os.path.join(onnx_path,"unet.onnx")):
            if int8:
                if not os.path.exists("state_dict.pt"):
                    #calibration_size=32,
                    calib_batch_size=2
                    module = importlib.import_module("diffusers")
                    pipeline = getattr(module,pipeline_class).from_pretrained(model_path,torch_dtype=torch.float16,use_safetensors=True,low_cpu_mem_usage=True,local_files_only=True).to("cuda")
                    calibration_prompts = load_calib_prompts(calib_batch_size,calibration_prompts_path )
                    input_image = get_image(input_image_url,save_path,download=False)
                    def do_calibrate(pipeline, calibration_prompts, **kwargs):
                        for i_th, prompts in enumerate(calibration_prompts):
                            if i_th >= kwargs["calib_size"]:
                                return 
                            pipeline(prompt=prompts,num_inference_steps=kwargs["n_steps"],image=input_image,generator=generator).images
                    def forward_loop(model):
                        pipeline.unet = model
                        do_calibrate(pipeline=pipeline,calibration_prompts=calibration_prompts,calib_size=100 // calib_batch_size,n_steps=denoising_steps,)
                    quant_config = get_int8_config(
                                model,
                                quant_level=3.0,
                                alpha=0.8,
                                percentile=1.0,
                                num_inference_steps=denoising_steps
                            )
                    print("进行Unet量化")
                    mtq.quantize(model, quant_config, forward_loop)
                    mto.save(model, "state_dict.pt")
                else:
                    mto.restore(model, "state_dict.pt")
                quantize_lvl(model, quant_level=3.0)
                mtq.disable_quantizer(model, filter_func)

            with torch.inference_mode(), torch.autocast("cuda"):
                if not static_shape:
                    image_height = image_height - 8 if image_height % 16 == 0 else image_height
                    image_width = image_width - 8 if image_width % 16 == 0 else image_width
                barch_dim = 3 if do_classifier_free_guidance else 1
                input = (torch.randn(barch_dim, 8, image_height//8, image_width//8, dtype=torch.float16, device="cuda"),torch.tensor([1.], dtype=torch.float16, device="cuda"),torch.randn(barch_dim, 77, 768, dtype=torch.float16, device="cuda"))
                #xB = '2B' if do_classifier_free_guidance  else 'B'
                torch.onnx.export(model,input,os.path.join(onnx_path,"unet.onnx"),export_params=True,opset_version=onnx_opset,do_constant_folding=True,input_names=['sample', 'timestep', 'encoder_hidden_states', 'images', 'controlnet_scales'],output_names=['latent'],
                #dynamic_axes={
                #'sample': {0: xB, 2: 'H', 3: 'W'},
                #'encoder_hidden_states': {0: xB},
                #'latent': {0: xB, 2: 'H', 3: 'W'}}
                )
        opt = Optimizer(onnx.load(os.path.join(onnx_path,"unet.onnx")),verbose=False)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        if int8:
           opt.fuse_mha_qkv_int8_sq()
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        if onnx_opt_graph.ByteSize() > 2147483648:
            onnx.save_model(onnx_opt_graph,os.path.join(onnx_opt_path,"unet.onnx"),save_as_external_data=True,all_tensors_to_one_file=True,convert_attribute=False)
        else:
            onnx.save(onnx_opt_graph,os.path.join(onnx_opt_path,"unet.onnx") ) 

def get_vae_encoder_onnx(model,onnx_path,onnx_opt_path,onnx_opset,image_height,image_width):
    if not os.path.exists(os.path.join(onnx_opt_path,"vae_encoder.onnx")): 
        if not os.path.exists(os.path.join(onnx_path,"vae_encoder.onnx")): 
          with torch.inference_mode(), torch.autocast("cuda"):
            input = torch.randn(1,3,image_height,image_width,dtype=torch.float16,device="cuda")
            torch.onnx.export(model,input,os.path.join(onnx_path,"vae_encoder.onnx"),export_params=True,opset_version=onnx_opset,do_constant_folding=True,input_names=['images'],output_names=['latent'],#dynamic_axes={
            #'images': {0: 'B', 2: '8H', 3: '8W'},
            #'latent': {0: 'B', 2: 'H', 3: 'W'}}
            )
        opt = Optimizer(onnx.load(os.path.join(onnx_path,"vae_encoder.onnx")), verbose=False)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        #if int8:
        #    opt.fuse_mha_qkv_int8_sq()
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        if onnx_opt_graph.ByteSize() > 2147483648:
            onnx.save_model(onnx_opt_graph,os.path.join(onnx_opt_path,"vae_encoder.onnx"),save_as_external_data=True,all_tensors_to_one_file=True,convert_attribute=False)
        else:
            onnx.save(onnx_opt_graph,os.path.join(onnx_opt_path,"vae_encoder.onnx") )

def get_vae_decoder_onnx(model,onnx_path,onnx_opt_path,onnx_opset,image_height,image_width,):
    if not os.path.exists(os.path.join(onnx_opt_path,"vae_decoder.onnx")):
        if not os.path.exists(os.path.join(onnx_path,"vae_decoder.onnx")):
            '''
            if int8:
                if not os.path.exists("vae_state_dict.pt"):
                    #calibration_size=32,
                    calib_batch_size=2
                    module = importlib.import_module("diffusers")
                    pipeline = getattr(module,pipeline_class).from_pretrained(model_path,torch_dtype=torch.float16,use_safetensors=True,low_cpu_mem_usage=True,local_files_only=True).to("cuda")
                    calibration_prompts = load_calib_prompts(calib_batch_size,calibration_prompts_path )
                    input_image = get_image(input_image_url,save_path,download=False)
                    def do_calibrate(pipeline, calibration_prompts, **kwargs):
                        for i_th, prompts in enumerate(calibration_prompts):
                            if i_th >= kwargs["calib_size"]:
                                return pipeline(prompt=prompts,num_inference_steps=kwargs["n_steps"],image=input_image)['images']
                    def forward_loop(model):
                        pipeline.vae = model
                        do_calibrate(pipeline=pipeline,calibration_prompts=calibration_prompts,calib_size=32 // calib_batch_size,n_steps=denoising_steps,)
                    quant_config = get_int8_config(
                                model,
                                quant_level=3.0,
                                alpha=0.8,
                                percentile=0.25,
                                num_inference_steps=denoising_steps
                            )
                    print("进行VAE量化")
                    mtq.quantize(model, quant_config, forward_loop)
                    mto.save(model, "vae_state_dict.pt")
                else:
                    mto.restore(model, "vae_state_dict.pt")
                quantize_lvl(model, quant_level=3.0)
                mtq.disable_quantizer(model, filter_func)
            '''
            with torch.inference_mode(), torch.autocast("cuda"):
                input = torch.randn(1,model.config['latent_channels'],image_height//8,image_width//8,dtype=torch.float16,device="cuda")
                torch.onnx.export(model,input,os.path.join(onnx_path,"vae_decoder.onnx"),export_params=True,opset_version=onnx_opset,do_constant_folding=True,input_names=['latent'],output_names=['images'],#dynamic_axes={
            #'latent': {0: 'B', 2: 'H', 3: 'W'},
            #'images': {0: 'B', 2: '8H', 3: '8W'}}
            )
        opt = Optimizer(onnx.load(os.path.join(onnx_path,"vae_decoder.onnx")), verbose=False)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        #if int8:
           #opt.fuse_mha_qkv_int8_sq()
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        if onnx_opt_graph.ByteSize() > 2147483648:
            onnx.save_model(onnx_opt_graph,os.path.join(onnx_opt_path,"vae_decoder.onnx"),save_as_external_data=True,all_tensors_to_one_file=True,convert_attribute=False)
        else:
            onnx.save(onnx_opt_graph,os.path.join(onnx_opt_path,"vae_decoder.onnx") )
