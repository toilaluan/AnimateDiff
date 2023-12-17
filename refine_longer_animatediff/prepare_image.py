import torch
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np
from PIL import Image
def prepare_image(images, height, width, use_simplified_condition_embedding=False, vae=None):
    image_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            (height, width), (1.0, 1.0), 
            ratio=(width/height, width/height)
        ),
        transforms.ToTensor(),
    ])

    controlnet_images = [image_transforms(image) for image in images]
    Image.fromarray((255. * (controlnet_images[0].cpu().numpy().transpose(1,2,0))).astype(np.uint8)).save("debug_conditional.png")

    controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
    controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")


    if use_simplified_condition_embedding:
        assert vae is not None
        num_controlnet_images = controlnet_images.shape[2]
        controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
        controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
        controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

    return controlnet_images