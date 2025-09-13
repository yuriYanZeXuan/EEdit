import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path
import traceback

# Add project root to sys.path to allow imports from other directories
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_functions import *
from MyCodes.MyFluxInpaintPipeline import FluxInpaintPipeline
from transformers import T5EncoderModel
from diffusers.utils import load_image
import importlib
from MyCodes import MyFluxForward
import types

# --- Global Settings & Model Pre-loading ---
WEIGHTS_DIR = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux-fill"
pipe = None

def load_models(weights_dir, dtype=torch.bfloat16):
    global pipe
    if pipe is not None:
        print("Models are already loaded.")
        return pipe

    print("Loading models...")
    try:
        from MyCodes.FluxTransformer2DModel_PREDEFINE import FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_pretrained(
            weights_dir,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=True)
        
        text_encoder_2 = T5EncoderModel.from_pretrained(
            weights_dir, 
            subfolder="text_encoder_2", 
            torch_dtype=dtype,
            local_files_only=True)

        pipe_instance = FluxInpaintPipeline.from_pretrained(
            weights_dir, 
            transformer=None, 
            text_encoder_2=None, 
            torch_dtype=dtype,
            local_files_only=True)
        pipe_instance.transformer = transformer
        pipe_instance.text_encoder_2 = text_encoder_2
        
        pipe_instance.transformer.forward = types.MethodType(MyFluxForward.forward, pipe_instance.transformer)
        pipe_instance.to('cuda')
        pipe = pipe_instance
        print("Models loaded successfully.")
        return pipe
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# --- Image Generation (adapted from inpaint_gen.py) ---
def generate_image(input_dict, prompt, strength, mask_timestep, num_inference_steps):
    global pipe
    try:
        print("--- [START] Image Generation ---")
        
        print("Step 1: Checking if models are loaded...")
        if pipe is None:
            raise gr.Error("Models are not loaded. Please check the console for errors during startup.")
        print("Step 1: Models are ready.")

        print("Step 2: Processing inputs...")
        if input_dict["background"] is None:
            raise gr.Error("Please upload an image.")
        if input_dict["layers"] is None or len(input_dict["layers"]) == 0:
            raise gr.Error("Please draw a mask on the image.")

        main_image = Image.fromarray(input_dict["background"]).convert("RGB")
        
        mask_layer = input_dict["layers"][0]
        # Check if the mask is empty (all alpha values are 0)
        if np.all(mask_layer[:, :, 3] == 0):
             raise gr.Error("The drawn mask is empty. Please draw on the area you want to inpaint.")

        # The mask is the drawn layer. It's RGBA, we take the alpha channel.
        mask_array = mask_layer[:, :, 3]  # Alpha channel
        mask_image = Image.fromarray(mask_array).convert("RGB")
        print("Step 2: Inputs processed successfully.")


        height, width = main_image.height, main_image.width

        # Using default parameters from cache_configs.json and allowing some to be controlled by UI
        param = {
            "use_cache": True,
            "num_inference_steps": int(num_inference_steps),
            "cascade_num": 3,
            "fresh_ratio": 0.1,
            "fresh_threshold": 3,
            "soft_fresh_weight": 0.25,
            "tailing_step": 1,
            "strength": strength,
            "inv_skip": 3,
            "eta": 0.7,
            "gamma": 0.7,
            "stop_timestep": 6,
            "mask_timestep": int(mask_timestep),
            "cache_type": "ours_predefine"
        }

        cache_type = param['cache_type']
        ratio_scheduler = 'constant'
        use_attn_map = False

        model_kwargs = {
            'fresh_ratio': param['fresh_ratio'],
            'cache_type': cache_type,
            'ratio_scheduler': ratio_scheduler,
            'force_fresh': 'global',
            'fresh_threshold': param['fresh_threshold'],
            'soft_fresh_weight': param['soft_fresh_weight'],
            'tailing_step': param['tailing_step'],
            'hw': (height // 16, width // 16)
        }
        
        print("Step 3: Initializing cache...")
        edit_idx = edit_mask_parser(mask_image, cascade_num=param['cascade_num'], height=height, width=width)
        cache_dic, current = cache_init(model_kwargs, param['num_inference_steps'], edit_idx)
        current['edit_idx_merged'] = convert_to_cache_index(edit_idx, edit_base=param.get('edit_base', 2), bonus_ratio=param.get('bonus_ratio', 0.8), height=height, width=width)
        current['edit_idx_merged'] = current['edit_idx_merged'].to("cuda")

        if cache_type == 'ours_predefine':
            predefine_cache_fresh_indices(cache_dic, current)
        print("Step 3: Cache initialized successfully.")

        joint_attention_kwargs = {
            'use_attn_map': use_attn_map,
            'cache_dic': cache_dic,
            'use_cache': param['use_cache'],
            'current': current,
        }

        print("Step 4: Calling the generation pipeline...")
        torch.manual_seed(42)
        res = pipe.gen(
            prompt=prompt,
            image=main_image,
            mask_image=mask_image,
            num_inference_steps=param['num_inference_steps'],
            strength=param['strength'],
            height=height,
            width=width,
            joint_attention_kwargs=joint_attention_kwargs,
            generator=torch.Generator(device='cuda').manual_seed(42),
            eta=param['eta'],
            gamma=param['gamma'],
            skip_T=param['inv_skip'],
            stop_timestep=param['stop_timestep'],
            mask_timestep=param['mask_timestep']
        )
        print("Step 4: Generation pipeline finished.")
        
        print("--- [SUCCESS] Image Generation Complete ---")
        return res.images[0]
    except Exception as e:
        print("--- [ERROR] An error occurred during generation ---")
        traceback.print_exc()
        raise gr.Error(f"An error occurred during the generation process. Please check the console for details. Error: {e}")

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# FLUX Inpainting with Cache Control")
    gr.Markdown("Upload an image, paint a mask, provide a prompt, and adjust the parameters to generate an inpainted image.")

    with gr.Row():
        with gr.Column():
            image_input = gr.ImageEditor(label="Image with Mask", type="numpy")
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            generate_button = gr.Button("Generate")
        with gr.Column():
            image_output = gr.Image(label="Output Image")
    
    with gr.Accordion("Advanced Settings", open=False):
        strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Strength")
        mask_timestep_slider = gr.Slider(minimum=0, maximum=50, value=18, step=1, label="Mask Timestep (Cache Interval)")
        steps_slider = gr.Slider(minimum=10, maximum=100, value=28, step=1, label="Number of Inference Steps")

    generate_button.click(
        fn=generate_image,
        inputs=[
            image_input,
            prompt_input,
            strength_slider,
            mask_timestep_slider,
            steps_slider,
        ],
        outputs=image_output
    )

if __name__ == "__main__":
    # Pre-load the models on startup
    load_models(WEIGHTS_DIR)
    demo.launch(share=True)
