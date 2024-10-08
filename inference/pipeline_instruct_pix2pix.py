from diffusers import StableDiffusionInstructPix2PixPipeline
import PIL
import torch

generator = torch.Generator(device="cuda").manual_seed(0)

image = PIL.Image.open("test.png")
#image = PIL.ImageOps.exif_transpose(image)
image = image.convert("RGB")

#pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix").to("cuda")
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("/root/autodl-tmp/model/instruct_pix2pix").to("cuda")
#pipeline.set_progress_bar_config(disable=True)

prompt="Zoom into the image"

action="1,0,0.0020,-0.0023"


ret=pipeline(action,image,num_inference_steps=1000,image_guidance_scale=1.5,guidance_scale=7,generator=generator).images[0]

ret.save("output.png")