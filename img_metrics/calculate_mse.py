import numpy as np
from PIL import Image
from tqdm import tqdm

def img_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_mse(img1, img2):
    if isinstance(img1, Image.Image):
        img1 = np.array(img1).astype(np.float32) / 255.0
    if isinstance(img2, Image.Image):
        img2 = np.array(img2).astype(np.float32) / 255.0
    if hasattr(img1, 'numpy'):
        img1 = img1.numpy()
    if hasattr(img2, 'numpy'):
        img2 = img2.numpy()
    
    mse_value = img_mse(img1, img2)
    
    result = {
        "value": float(mse_value),
        "img_setting": list(img1.shape),  # [height, width, channels]
        "img_setting_name": "height, width, channels"
    }
    
    return result

def main():
    test_img1 = np.zeros((64, 64, 3))
    test_img2 = np.ones((64, 64, 3)) * 0.1  # 略微不同的图像
    
    import json
    result = calculate_mse(test_img1, test_img2)
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
