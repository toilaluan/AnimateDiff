import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionImg2ImgPipeline

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

# def reload_controlnet(pipeline):

def load_model(pretrained_model_path: str, inference_config: dict, model_config: dict):
    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").cuda()
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()
    # load controlnet model
    controlnet = controlnet_images = None
    unet.config.num_attention_heads = 8
    unet.config.projection_class_embeddings_input_dim = None

    controlnet_config = OmegaConf.load(model_config.controlnet_config)
    controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

    print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
    controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
    controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
    controlnet_state_dict.pop("animatediff_config", "")
    controlnet.load_state_dict(controlnet_state_dict)
    controlnet.cuda()

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()


    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    ).to("cuda")

    pipeline = load_weights(
        pipeline,
        # motion module
        motion_module_path         = model_config.get("motion_module", ""),
        motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
        # domain adapter
        adapter_lora_path          = model_config.get("adapter_lora_path", ""),
        adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
        # image layers
        dreambooth_model_path      = model_config.get("dreambooth_path", ""),
        lora_model_path            = model_config.get("lora_model_path", ""),
        lora_alpha                 = model_config.get("lora_alpha", 0.8),
    ).to("cuda")

    return pipeline


if __name__ == "__main__":
    model_config = OmegaConf.load("configs/demo/model_config.yaml")
    inference_config = OmegaConf.load("configs/demo/inference_config.yaml")
    pretrained_model_path = "models/StableDiffusion"

    pipeline = load_model(pretrained_model_path, inference_config, model_config)