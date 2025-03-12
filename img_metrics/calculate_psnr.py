import numpy as np
import torch
from tqdm import tqdm
import math

def img_psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

def trans(x):
    return x

def calculate_psnr(img1, img2):
    psnr_value = img_psnr(img1.numpy(), img2.numpy())
    
    result = {
        "value": psnr_value,
        "img_setting": img1.shape,
        "img_setting_name": "channel, height, width",
    }

    return result


def main():
    CHANNEL = 3
    SIZE = 64
    img1 = torch.zeros(CHANNEL, SIZE, SIZE, requires_grad=False)
    img2 = torch.zeros(CHANNEL, SIZE, SIZE, requires_grad=False)

    import json
    result = calculate_psnr(img1, img2)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()