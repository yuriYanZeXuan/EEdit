import numpy as np
import os
import json
import random
import torch
import json
from PIL import Image 
import cv2

def mask_decode(encoded_mask, image_shape=[512,512]):
    """decode mask to np array"""
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len = min(encoded_mask[i+1], length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j] = 1
            
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:] = 1
    mask_array[-1,:] = 1
    mask_array[:,0] = 1
    mask_array[:,-1] = 1
            
    return mask_array

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    mask_dir = "./input/inpaint/masks"
    os.makedirs(mask_dir, exist_ok=True)

    inpaint_mapping={
        '0':'random',
        '1':'change_object',
        '2':'add_object',
        '3':'delete_object',
        '4':'change_attribute_content',
        '5':'change_attribute_pose',
        '6':'change_attribute_color',
        '7':'change_attribute_material',
        '8':'change_background',
        '9':'change_style'
    }
    with open("./input/inpaint/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
        edit_category_list = ['0','1','2','3','4','5','6','7','8','9']
        
        for idx, (key, item) in enumerate(editing_instruction.items()):
            output_dict = {"imgs": []}
            if item["editing_type_id"] not in edit_category_list:
                continue
                
            # get prompt
            editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
            editing_type_id = item["editing_type_id"]
            # decode and save mask
            mask = mask_decode(item["mask"])
            mask = (mask * 255).astype(np.uint8)  # convert to 0-255 range
            mask_path = os.path.join(mask_dir, f"mask-{idx:03d}.png")
            cv2.imwrite(mask_path, mask)
            