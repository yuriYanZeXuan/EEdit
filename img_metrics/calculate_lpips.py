import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import lpips

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # value range [0, 1] -> [-1, 1]
    x = x * 2 - 1

    return x

def calculate_lpips(img1, img2, device):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    # print("calculate_lpips...")

    assert img1.shape == img2.shape

    # images [batch_size, channel, h, w]

    # value range [0, 1] -> [-1, 1]
    img1 = trans(img1)
    img2 = trans(img2)

    lpips_results = []

    # get an image
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    loss_fn.to(device)

    # calculate lpips of an image
    lpips_results.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
    
    lpips_results = np.array(lpips_results)
    
    lpips = {}
    lpips_std = {}

    lpips[0] = np.mean(lpips_results)
    lpips_std[0] = np.std(lpips_results)

    result = {
        "value": lpips,
        "value_std": lpips_std,
        "image_setting": img1.shape,
        "image_setting_name": "channel, height, width",
    }

    return result

# test code / using example

def main():
    CHANNEL = 3
    SIZE = 64
    img1 = torch.zeros(CHANNEL, SIZE, SIZE, requires_grad=False)
    img2 = torch.ones(CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json
    result = calculate_lpips(img1, img2, device)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()