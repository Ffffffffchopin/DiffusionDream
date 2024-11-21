from diffusers import StableDiffusionInstructPix2PixPipeline
import PIL
import torch
import requests
import io
import time
#from diffusers import AutoencoderTiny
#from diffusers import DDIMScheduler

#from diffusers.utils import load_image

torch.backends.cuda.matmul.allow_tf32 = True

start_time = time.perf_counter()

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



generator = torch.Generator(device="cuda").manual_seed(0)

image = get_image("https://www.helloimg.com/i/2024/10/20/6714d43670bd9.png","csgo.jpg",download=False)
#image = load_image("https://www.helloimg.com/i/2024/10/20/6714d43670bd9.png")
#image = PIL.ImageOps.exif_transpose(image)
#image = image.convert("RGB")





#pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix").to("cuda")
#pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("E:\models\instruct_pix2pix",torch_dtype=torch.float16,local_files_only=True,variant="fp16")
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("E:\\models\\fffchopin_instruct_pix2pix",torch_dtype=torch.float16,local_files_only=True,)
#pipeline.set_progress_bar_config(disable=True)

#pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config,torch_dtype=torch.float16)

#pipeline.vae = AutoencoderTiny.from_pretrained("E:\\models\\TinyVAE",torch_dtype=torch.float16)

pipeline = pipeline.to("cuda")

prompt="The siblings are all robots"

action="1,-1,-0.0294,0.0629"

print("Time to preprocess: ", time.perf_counter() - start_time)

start_time = time.perf_counter()

action_list = [action,action]

#ret=pipeline(action,image,num_inference_steps=30,generator=generator)['images'][0]
ret=pipeline(action,image,num_inference_steps=30,generator=generator).images

'''
for i in range(10):
    ret=pipeline(action,image,num_inference_steps=10,image_guidance_scale=1.5,guidance_scale=7.5,generator=generator).images[0]
    ret.save(f"output_{i}.jpg")
    image = ret
'''

end_time = time.perf_counter()

print("Time to process: ", end_time - start_time)

#print(len(ret))

ret[0].save("pipeline.jpg")
#ret[1].save("pipeline2.jpg")