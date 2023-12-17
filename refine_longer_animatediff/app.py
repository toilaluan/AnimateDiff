import gradio as gr
import torch
import imageio
import numpy as np
from PIL import Image
import random
# from diffusers import StableDiffusionImg2ImgPipeline
from hed_detector import HEDdetector
from omegaconf import OmegaConf
from animatediff.utils.util import save_videos_grid
from refine_longer_animatediff.load_model import load_model
from refine_longer_animatediff.prepare_image import prepare_image
from upscalers import upscale
scribbler = HEDdetector()
model_config = OmegaConf.load("configs/demo/model_config.yaml")
inference_config = OmegaConf.load("configs/demo/inference_config.yaml")
pretrained_model_path = "models/StableDiffusion"
# refine_pipeline = StableDiffusionImg2ImgPipeline.from_single_file(model_config['dreambooth_path']).to("cuda")
pipeline = load_model(pretrained_model_path, inference_config, model_config)
savedir = "samples"


css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""
@torch.no_grad()
def infer(
    conditional_image,
    prompt_textbox, 
    negative_prompt_textbox, 
    sample_step_slider, 
    width_slider, 
    length_slider, 
    height_slider, 
    cfg_scale_slider, 
    seed_textbox,
):
    n_loop = (length_slider + 16) // 16 - 1
    print("**PREPARE IMAGE**")
    # conditional_image = scribbler(np.array(conditional_image))
    # conditional_image = Image.fromarray(conditional_image).convert("RGB")
    # conditional_image.save("scribble.png")
    conditional_images = prepare_image([conditional_image], height_slider, width_slider, pipeline.controlnet.use_simplified_condition_embedding, pipeline.vae)
    init_conditional_image = conditional_images[0]
    print("**START INFER")
    # manually set random seed for reproduction
    if seed_textbox != -1: 
        torch.manual_seed(seed_textbox)
    else: 
        torch.seed()
    
    print(f"current seed: {torch.initial_seed()}")
    print(f"sampling {prompt_textbox} ...")

    sample = pipeline(
        prompt_textbox,
        negative_prompt     = negative_prompt_textbox,
        num_inference_steps = sample_step_slider,
        guidance_scale      = cfg_scale_slider,
        width               = width_slider,
        height              = height_slider,
        video_length        = 16,
        controlnet_images = conditional_images,
        controlnet_image_index = [0],
    ).videos
    save_filename = f"{savedir}/sample/{prompt_textbox[:20]}.mp4"
    frames = save_videos_grid(sample, save_filename, do_save=False)

    for loop in range(n_loop):
        # conditional_images = [scribbler(np.array(image)) for image in frames[-3:]]
        conditional_images = [Image.fromarray(image).convert("RGB") for image in frames[-3:]]
        conditional_images = [upscale("R-ESRGAN 4x+ Anime6B", image, 2) for image in conditional_images]
        conditional_images.append(conditional_image)
        conditional_images = prepare_image(conditional_images, height_slider, width_slider, pipeline.controlnet.use_simplified_condition_embedding, pipeline.vae)
        sample = pipeline(
            prompt_textbox,
            negative_prompt     = negative_prompt_textbox,
            num_inference_steps = sample_step_slider,
            guidance_scale      = cfg_scale_slider,
            width               = width_slider,
            height              = height_slider,
            video_length        = 16,
            controlnet_images = conditional_images,
            controlnet_image_index = [4,2,0,8],
        ).videos
        frames.extend(save_videos_grid(sample, save_filename, do_save=False))
    frames = [upscale("R-ESRGAN 4x+ Anime6B", Image.fromarray(image), 4) for image in frames]
    frames = [np.array(frame, dtype=np.uint8) for frame in frames]
    imageio.mimsave(save_filename, frames, fps=8)
    
    return gr.Video.update(value=save_filename)

    
def ui():
    with gr.Blocks(css=css) as demo:
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for AnimateDiff.
                """
            )
            conditional_image = gr.Image(type="pil")
            
            prompt_textbox = gr.Textbox(label="Prompt", lines=2)
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2)
                
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        # sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=25, minimum=10, maximum=100, step=1)
                        
                    width_slider     = gr.Slider(label="Width",            value=512, minimum=256, maximum=1024, step=64)
                    height_slider    = gr.Slider(label="Height",           value=512, minimum=256, maximum=1024, step=64)
                    length_slider    = gr.Slider(label="Animation length", value=16,  minimum=8,   maximum=1024,   step=1)
                    cfg_scale_slider = gr.Slider(label="CFG Scale",        value=7.5, minimum=0,   maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])
            
                    generate_button = gr.Button(value="Generate", variant='primary')
                    
                result_video = gr.Video()

            generate_button.click(
                fn=infer,
                inputs=[
                    conditional_image,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sample_step_slider, 
                    width_slider, 
                    length_slider, 
                    height_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                ],
                outputs=result_video
            )
    return demo

if __name__ == "__main__":
    demo = ui()
    demo.queue().launch(share=False, show_error=True)