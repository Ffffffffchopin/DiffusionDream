from engine import Engine
import os


def get_clip_engine(engine_path,onnx_opt_path,int8,static_batch):
    engine = Engine(os.path.join(engine_path, "clip.plan"))
    if not os.path.exists(os.path.join(engine_path, "clip.plan")):
        update_output_names = None
        fp16amp = True
        bf16amp = False
        strongly_typed = False
        extra_build_args = {'verbose': True}
        extra_build_args['builder_optimization_level'] = 4 if int8 else 3
        max_batch = 1 if static_batch else 16
        '''
        if int8:
            extra_build_args['int8'] = True
            extra_build_args['precision_constraints'] = 'prefer'
        '''
        engine.build(os.path.join(onnx_opt_path,"clip.onnx"),strongly_typed=strongly_typed,fp16=fp16amp,bf16=bf16amp,input_profile={
            'input_ids': [(1, 77), (1, 77), (max_batch, 77)]
        },enable_refit=True,enable_all_tactics=True,timing_cache=None,update_output_names=update_output_names,**extra_build_args)
    return engine

def get_unet_engine(engine_path,onnx_opt_path,int8,static_batch,image_height,image_width,static_shape,do_classifier_free_guidance):
    engine = Engine(os.path.join(engine_path, "unet.plan")) 
    if not os.path.exists(os.path.join(engine_path, "unet.plan")):
        xB = 2 if do_classifier_free_guidance else 1 
        update_output_names = None
        fp16amp = True
        bf16amp = False
        strongly_typed = False
        extra_build_args = {'verbose': True}
        extra_build_args['builder_optimization_level'] = 4 if int8 else 3
        max_batch = 1 if static_batch else 16
        if not static_shape:
            image_height = image_height - 8 if image_height % 16 == 0 else image_height
            image_width = image_width - 8 if image_width % 16 == 0 else image_width
        input_profile = {"sample":[(xB*1,8,image_height//8,image_width//8),(xB*1,8,image_height//8,image_width//8),[xB*max_batch,8,image_height//8,image_width//8]],'encoder_hidden_states':[(xB*1,77,768),(xB*1,77,768),(xB*max_batch,77,768)]}
        if int8:
            extra_build_args['int8'] = True
            extra_build_args['precision_constraints'] = 'prefer'
        engine.build(os.path.join(onnx_opt_path,"unet.onnx"),strongly_typed=strongly_typed,fp16=fp16amp,bf16=bf16amp,input_profile=input_profile,enable_refit=False,enable_all_tactics=True,timing_cache=None,update_output_names=update_output_names,**extra_build_args)
    return engine

def get_vae_encoder_engine(engine_path,onnx_opt_path,int8,image_height,image_width,static_batch):
    engine = Engine(os.path.join(engine_path, "vae_encoder.plan"))
    if not os.path.exists(os.path.join(engine_path, "vae_encoder.plan")):
        update_output_names = None
        fp16amp = True
        bf16amp = False
        strongly_typed = False
        extra_build_args = {'verbose': True}
        extra_build_args['builder_optimization_level'] = 4 if int8 else 3
        max_batch = 1 if static_batch else 16
        input_profile = {"images":[(1,3,image_height,image_width),(1,3,image_height,image_width),(max_batch,3,image_height,image_width)]}
        '''
        if int8:
            extra_build_args['int8'] = True
            extra_build_args['precision_constraints'] = 'prefer'
        '''
        engine.build(os.path.join(onnx_opt_path,"vae_encoder.onnx"),strongly_typed=strongly_typed,fp16=fp16amp,bf16=bf16amp,input_profile=input_profile,enable_refit=False,enable_all_tactics=True,timing_cache=None,update_output_names=update_output_names,**extra_build_args)
    return engine

def get_vae_decoder_engine(engine_path,onnx_opt_path,int8,image_height,image_width,static_batch,vae):
    engine = Engine(os.path.join(engine_path, "vae_decoder.plan"))
    if not os.path.exists(os.path.join(engine_path, "vae_decoder.plan")):
        update_output_names = None
        fp16amp = True
        bf16amp = False
        strongly_typed = False
        extra_build_args = {'verbose': True}
        extra_build_args['builder_optimization_level'] = 4 if int8 else 3
        max_batch = 1 if static_batch else 16
        input_profile = {"latent":[(1,vae.config['latent_channels'],image_height//8,image_width//8),(1,vae.config['latent_channels'],image_height//8,image_width//8),(max_batch,vae.config['latent_channels'],image_height//8,image_width//8)]}
        '''
        if int8:
            extra_build_args['int8'] = True
            extra_build_args['precision_constraints'] = 'prefer'
        '''
        engine.build(os.path.join(onnx_opt_path,"vae_decoder.onnx"),strongly_typed=strongly_typed,fp16=fp16amp,bf16=bf16amp,input_profile=input_profile,enable_refit=False,enable_all_tactics=True,timing_cache=None,update_output_names=update_output_names,**extra_build_args)
    return engine
