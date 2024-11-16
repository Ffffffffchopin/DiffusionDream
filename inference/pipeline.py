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

def download_image(url):
    #image = requests.get(url, stream=True).raw
    #image = io.BytesIO(image)
    response = requests.get(url)
    byte = io.BytesIO(response.content)
    image = PIL.Image.open(byte)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.save("input.jpg")
    return image

generator = torch.Generator(device="cuda").manual_seed(0)

image = download_image("https://www.helloimg.com/i/2024/10/20/6714d43670bd9.png")
#image = load_image("https://www.helloimg.com/i/2024/10/20/6714d43670bd9.png")
#image = PIL.ImageOps.exif_transpose(image)
#image = image.convert("RGB")





#pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix").to("cuda")
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("E:\\models\\fffchopin_instruct_pix2pix",torch_dtype=torch.float16,local_files_only=True)
#pipeline.set_progress_bar_config(disable=True)

#pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config,torch_dtype=torch.float16)

#pipeline.vae = AutoencoderTiny.from_pretrained("E:\\models\\TinyVAE",torch_dtype=torch.float16)

pipeline = pipeline.to("cuda")

prompt="Zoom into the image"

action="1,0,0.0,0.0"

print("Time to preprocess: ", time.perf_counter() - start_time)

start_time = time.perf_counter()

ret=pipeline(action,image,num_inference_steps=40,mage_guidance_scale=0.0,guidance_scale=0.0,generator=generator)['images'][0]

'''
for i in range(10):
    ret=pipeline(action,image,num_inference_steps=10,image_guidance_scale=1.5,guidance_scale=7.5,generator=generator).images[0]
    ret.save(f"output_{i}.jpg")
    image = ret
'''

end_time = time.perf_counter()

print("Time to process: ", end_time - start_time)

#print(len(ret))

ret.save("pipeline2.jpg")