import argparse
from pathlib import Path
import torch
import json
from cache_functions import *
from MyCodes.MyFluxInpaintPipeline import FluxInpaintPipeline
from transformers import T5EncoderModel
from diffusers.utils import load_image
import importlib
from MyCodes import MyFluxForward
import os
import types

def get_next_number(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    files = [f for f in os.listdir(dirname)]
    if not files:
        return 1
    nums = [int(f.split('.')[0].split('-')[-1]) for f in files if f.split('.')[0].split('-')[-1].isdigit()]
    return max(nums) + 1 if nums else 1


def parse_args():
    parser = argparse.ArgumentParser(description='code for inpaint')
    parser.add_argument('--weights_dir', type=str, default='/root/your-path/weights',
                       help='model weights directory')
    parser.add_argument('--config_path', type=str, 
                       default='configs/inpaint/example_config.json',
                       help='path of config file')
    parser.add_argument('--img_config', type=str,
                       default='configs/inpaint/example_imgs.json', 
                       help='path of image config file')
    parser.add_argument('--output_dir', type=str,
                       default='test_outputs/inpaint',
                       help='output directory')
    parser.add_argument('--use_predefine', type=bool,
                       default=False,
                       help='whether to use predefine')
    return parser.parse_args()

def load_models(args, dtype=torch.bfloat16):
    if args.use_predefine:
        from MyCodes.FluxTransformer2DModel_PREDEFINE import FluxTransformer2DModel
    else:
        from MyCodes.FluxTransformer2DModel import FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_single_file(
        pretrained_model_link_or_path_or_dict=f"{args.weights_dir}/flux1-dev.safetensors",
        config=f"{args.weights_dir}/transformer_config.json",
        torch_dtype=dtype,
        local_files_only=True)
    
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.weights_dir, 
        subfolder="text_encoder_2", 
        torch_dtype=dtype)

    pipe = FluxInpaintPipeline.from_pretrained(
        args.weights_dir, 
        transformer=None, 
        text_encoder_2=None, 
        torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2
    
        
    pipe.transformer.forward = types.MethodType(MyFluxForward.forward, pipe.transformer)
    pipe.to('cuda')
    
    return pipe

def generate_image(pipe, img_config, param_config, output_dir):
    main_image = load_image(img_config["image"])
    mask_image = load_image(img_config["mask_image"])
    prompt = img_config["prompt"]
    height=main_image.height
    width=main_image.width
    for param_idx,param in enumerate(param_config['params']):
        if 'cache_type' in param:
            ratio_scheduler = 'constant'
            use_attn_map=False
            if param['cache_type'] == 'ours_cache':
                cache_type = 'ours_cache'
            elif param['cache_type'] == 'ours_predefine':
                cache_type = 'ours_predefine'
            
        model_kwargs = {
            'fresh_ratio': param['fresh_ratio'],
            'cache_type': cache_type,
            'ratio_scheduler': ratio_scheduler,
            'force_fresh': 'global',
            'fresh_threshold': param['fresh_threshold'],
            'soft_fresh_weight': param['soft_fresh_weight'],
            'tailing_step': param['tailing_step'],
            'hw': (height//16,width//16)
        }
        edit_idx=edit_mask_parser(mask_image,cascade_num=param['cascade_num'])
        cache_dic, current = cache_init(model_kwargs, param['num_inference_steps'],edit_idx)
        current['edit_idx_merged']=convert_to_cache_index(edit_idx,edit_base=param.get('edit_base',2),bonus_ratio=param.get('bonus_ratio',0.8))
        current['edit_idx_merged']=current['edit_idx_merged'].to("cuda")
        if cache_type=='ours_predefine':
            predefine_cache_fresh_indices(cache_dic, current)
        joint_attention_kwargs = {
            'use_pulid':False,
            'use_attn_map': use_attn_map,
            'cache_dic': cache_dic,
            'use_cache': param['use_cache'],
            'current': current,
        }
        torch.manual_seed(42)
        res = pipe(
            prompt=prompt,
            image=main_image, 
            mask_image=mask_image,
            num_inference_steps=param['num_inference_steps'],
            strength=param['strength'],
            height=height,
            width=width,
            joint_attention_kwargs=joint_attention_kwargs,
            generator=torch.Generator(device='cuda').manual_seed(42),
            skip_T=3 if 'inv_skip' not in param else param['inv_skip'],
            skip_T_fwd=1 if 'fwd_skip' not in param else param['fwd_skip'],
            stop_timestep=4 if 'stop_timestep' not in param else param['stop_timestep']
        )
        image=res.images[0]
        num = get_next_number(output_dir)
        image.save(f"{output_dir}/{num:03d}.png")
        
    

def main():
    args = parse_args()
    
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    
    pipe = load_models(args)
    
    with open(args.img_config, 'r') as f:
        if args.img_config.endswith('.jsonl'):
            img_configs = [json.loads(line) for line in f]
        else:
            img_configs = json.load(f)
    with open(args.config_path, 'r') as f:
        param_config = json.load(f)
    
    if args.img_config.endswith('.jsonl'):
        for img_config in img_configs:
            generate_image(pipe,img_config['imgs'][0], param_config, args.output_dir)
    else:
        for img_config in img_configs['imgs']:
            generate_image(pipe,img_config, param_config, args.output_dir)
            
if __name__ == "__main__":
    main()