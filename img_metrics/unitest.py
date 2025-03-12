import os
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from calculate_fid import calculate_fid_for_test
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
from calculate_mse import calculate_mse
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img)

def calculate_metrics(folder1, folder2, output_file, device='cuda'):
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    
    results = []
    
    for img1_name, img2_name in tqdm(zip(files1, files2), total=len(files1)):
        img1_path = os.path.join(folder1, img1_name)
        img2_path = os.path.join(folder2, img2_name)
        
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        if img1.shape!=img2.shape:
            transform_resize = transforms.Resize((img1.shape[1], img1.shape[2]))
            img2 = transform_resize(img2)
        metrics = {
            'image_pair': (img1_name, img2_name),
            'fid': calculate_fid_for_test(img1, img2, device),
            'psnr': calculate_psnr(img1, img2)['value'],
            'ssim': calculate_ssim(img1, img2)['value'],
            'lpips': calculate_lpips(img1, img2, device)['value'][0],
            'mse': calculate_mse(img1, img2)['value'],
        }
        
        results.append(metrics)
        
    avg_metrics = {
        'avg_fid': sum(r['fid'] for r in results) / len(results),
        'avg_psnr': sum(r['psnr'] for r in results) / len(results),
        'avg_ssim': sum(r['ssim'] for r in results) / len(results),
        'avg_lpips': sum(r['lpips'] for r in results) / len(results),
        'avg_mse': sum(r['mse'] for r in results) / len(results)
    }
    
    final_results = {
        'individual_results': results,
        'average_metrics': avg_metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    return final_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='calculate image metrics')
    parser.add_argument('--folder1', type=str, required=True,
                        help='folder1')
    parser.add_argument('--folder2', type=str, required=True, 
                        help='folder2')
    parser.add_argument('--output', type=str, required=True,
                        help='output path')
    
    args = parser.parse_args()
    
    results = calculate_metrics(args.folder1, args.folder2, args.output)
    print("average metrics:")
    print(json.dumps(results['average_metrics'], indent=4))
