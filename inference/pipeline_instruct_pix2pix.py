from diffusers import StableDiffusionInstructPix2PixPipeline
import PIL
import torch

generator = torch.Generator(device="cuda").manual_seed(0)

image = PIL.Image.open("屏幕截图 2024-10-04 162903.png")
image = PIL.ImageOps.exif_transpose(image)
image = image.convert("RGB")

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("fffffchopin/Instruct_pix2pix").to("cuda")
#pipeline.set_progress_bar_config(disable=True)


ret=pipeline("tx:0.0215,ty:0.0065,tz:-0.0296,dx:0.0020,dy:-0.0023",image,num_inference_steps=20,image_guidance_scale=1.5,guidance_scale=7,generator=generator).images[0]

ret.save("output.png")