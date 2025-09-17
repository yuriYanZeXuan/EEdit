import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path
import traceback
import time
import glob
from datetime import datetime
from io import BytesIO
import tempfile

# Add project root to sys.path to allow imports from other directories
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_functions import *
from MyCodes.MyFluxInpaintPipeline import FluxInpaintPipeline
from transformers import T5EncoderModel
from diffusers.utils import load_image
import importlib
from MyCodes import MyFluxForward # Import the new forward pass
import types

# --- Global Settings & Model Pre-loading ---
WEIGHTS_DIR = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/flux"
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
        
        # Apply the new, compatible forward pass for the flux-fill model
        pipe_instance.transformer.forward = types.MethodType(MyFluxForward.forward, pipe_instance.transformer)
        pipe_instance.to('cuda')
        pipe = pipe_instance
        print("Models loaded successfully.")
        return pipe
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# --- Image Generation (adapted from inpaint_gen.py) ---
def generate_image(input_dict, prompt, strength, mask_timestep, num_inference_steps, use_rf_inversion, eta, gamma, start_timestep, stop_timestep, use_cache, cached_image_path_state):
    global pipe
    try:
        print("--- [START] Image Generation ---")
        start_time = time.time()
        
        print("Step 1: Checking if models are loaded...")
        if pipe is None:
            raise gr.Error("Models are not loaded. Please check the console for errors during startup.")
        print("Step 1: Models are ready.")

        print("Step 2: Processing inputs...")
        # --- [OPTIMIZATION] ---
        # Prioritize loading the image from the server-side path if it exists.
        # This avoids the slow re-upload from the client.
        if cached_image_path_state and os.path.exists(cached_image_path_state):
            print(f"Loading image directly from server path: {cached_image_path_state}")
            main_image = Image.open(cached_image_path_state).convert("RGB")
            
            # Launder the image to strip metadata, mimicking a user upload.
            buffer = BytesIO()
            main_image.save(buffer, format="PNG")
            buffer.seek(0)
            main_image = Image.open(buffer)
            
            # Since we loaded from cache, the mask must be from the user's drawing on the client.
            if input_dict["layers"] is None or len(input_dict["layers"]) == 0:
                raise gr.Error("Please draw a mask on the image.")
            mask_layer = input_dict["layers"][0]

        # Fallback to using the uploaded image data if no valid server path is available.
        else:
            if input_dict["background"] is None:
                raise gr.Error("Please upload an image.")
            if input_dict["layers"] is None or len(input_dict["layers"]) == 0:
                raise gr.Error("Please draw a mask on the image.")
            main_image = Image.fromarray(input_dict["background"]).convert("RGB")
            mask_layer = input_dict["layers"][0]

        # Ensure image dimensions are multiples of 16 for model compatibility.
        w, h = main_image.size
        new_w, new_h = w, h
        if new_w % 16 != 0:
            new_w = new_w - (new_w % 16)
        if new_h % 16 != 0:
            new_h = new_h - (new_h % 16)
        if (w, h) != (new_w, new_h):
            print(f"Adjusting image size from {w}x{h} to {new_w}x{new_h} for model compatibility.")
            main_image = main_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
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
            "use_cache": use_cache,
            "num_inference_steps": int(num_inference_steps),
            "cascade_num": 3,
            "fresh_ratio": 0.1,
            "fresh_threshold": 3,
            "soft_fresh_weight": 0.25,
            "tailing_step": 1,
            "inv_skip": 3,
            "cache_type": "ours_predefine"
        }

        # If cache is disabled, also disable step skipping for a true baseline measurement.
        if not use_cache:
            param['inv_skip'] = 1
            param['fresh_threshold'] = 1
            print("Cache is disabled. Forcing inv_skip to 1 for baseline performance measurement.")

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

        if cache_type == 'ours_predefine' and use_cache:
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
            strength=strength,
            height=height,
            width=width,
            joint_attention_kwargs=joint_attention_kwargs,
            generator=torch.Generator(device='cuda').manual_seed(42),
            use_rf_inversion=use_rf_inversion,
            eta=eta,
            gamma=gamma,
            skip_T=param['inv_skip'],
            start_timestep=int(start_timestep),
            stop_timestep=int(stop_timestep),
            mask_timestep=int(mask_timestep)
        )
        print("Step 4: Generation pipeline finished.")
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"--- [SUCCESS] Image Generation Complete in {inference_time:.2f}s ---")
        # Return the result and None to clear the cached image path state after use
        return res.images[0], f"Inference Time: {inference_time:.2f} seconds", None
    except Exception as e:
        print("--- [ERROR] An error occurred during generation ---")
        traceback.print_exc()
        raise gr.Error(f"An error occurred during the generation process. Please check the console for details. Error: {e}")

# --- New Helper Functions ---
def load_latest_image():
    """Loads the most recent image from the cache directory."""
    cache_dir = "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/demo/history/images"
    try:
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")
        
        list_of_files = glob.glob(os.path.join(cache_dir, '*'))
        if not list_of_files:
            raise FileNotFoundError(f"No files found in cache directory: {cache_dir}")
            
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading latest image: {latest_file}")
        
        img = Image.open(latest_file).convert("RGB")

        # "Launder" the image by saving and reloading it from an in-memory buffer.
        # This strips potentially problematic metadata (e.g., EXIF orientation tags)
        # that can cause model crashes, making it behave like a fresh user upload.
        buffer = BytesIO()
        img.save(buffer, format="PNG") # Use a lossless format to avoid quality loss
        buffer.seek(0)
        img = Image.open(buffer)

        # Return both the image data for display and the file path for state tracking
        return np.array(img), latest_file
    except Exception as e:
        print(f"Error loading latest image: {e}")
        traceback.print_exc()
        # Return None for both outputs on error
        return None, None

def save_image_for_download(image_to_save):
    """Saves the image to a temporary file and returns the path."""
    if image_to_save is None:
        gr.Warning("No image available to save.")
        return None

    try:
        # Create a temporary file with a .png extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img = Image.fromarray(image_to_save)
            img.save(tmp.name)
            print(f"Image saved to temporary file: {tmp.name}")
            return tmp.name
    except Exception as e:
        print(f"Error creating temporary file for download: {e}")
        traceback.print_exc()
        raise gr.Error(f"Failed to prepare image for download. Error: {e}")

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# FLUX Inpainting with Cache Control")
    gr.Markdown("Upload an image, paint a mask, provide a prompt, and adjust the parameters to generate an inpainted image.")

    # Add a State component to store the server-side path of the cached image
    cached_image_path_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            image_input = gr.ImageEditor(label="Image with Mask", type="numpy")
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            with gr.Row():
                generate_button = gr.Button("Generate", variant="primary")
                load_cache_button = gr.Button("Load Latest from Cache")
        with gr.Column():
            image_output = gr.Image(label="Output Image")
            time_output = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                save_button = gr.Button("Save Image")
            download_output = gr.File(label="Download Image", visible=False)

    with gr.Accordion("Advanced Settings", open=True):
        use_cache_checkbox = gr.Checkbox(label="Use Cache", value=True)
        use_rf_inversion_checkbox = gr.Checkbox(label="Use RF Inversion (Recommended)", value=True)
        strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Strength")
        eta_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Eta (Inversion Guidance)")
        gamma_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Gamma (Inversion Noise)")
        start_timestep_slider = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Start Timestep (Inversion)")
        stop_timestep_slider = gr.Slider(minimum=0, maximum=50, value=3, step=1, label="Stop Timestep (Inversion)")
        mask_timestep_slider = gr.Slider(minimum=0, maximum=50, value=18, step=1, label="Mask Timestep (Constraint)")
        steps_slider = gr.Slider(minimum=10, maximum=100, value=28, step=1, label="Number of Inference Steps")

    generate_button.click(
        fn=generate_image,
        inputs=[
            image_input,
            prompt_input,
            strength_slider,
            mask_timestep_slider,
            steps_slider,
            use_rf_inversion_checkbox,
            eta_slider,
            gamma_slider,
            start_timestep_slider,
            stop_timestep_slider,
            use_cache_checkbox,
            cached_image_path_state, # Pass the state as input
        ],
        outputs=[image_output, time_output, cached_image_path_state] # Clear the state after use
    )

    load_cache_button.click(
        fn=load_latest_image,
        inputs=[],
        outputs=[image_input, cached_image_path_state] # Update both image and state
    )
    
    # When a user uploads a new image, clear the cached path state
    # to ensure the uploaded image is used, not the old cached one.
    image_input.upload(lambda: None, outputs=cached_image_path_state)
    image_input.clear(lambda: None, outputs=cached_image_path_state)

    def show_download(filepath):
        if filepath:
            return gr.update(visible=True, value=filepath)
        return gr.update(visible=False)

    save_button.click(
        fn=save_image_for_download,
        inputs=[image_output],
        outputs=[download_output]
    ).then(
        fn=show_download,
        inputs=[download_output],
        outputs=[download_output]
    )

if __name__ == "__main__":
    # Pre-load the models on startup
    load_models(WEIGHTS_DIR)
    demo.launch(share=True)
