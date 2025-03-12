import os
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
from calculate_clip import calculate_clip_metrics,load_clip_model
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img)

def calculate_metrics(folder1,prompt_file, output_file, device='cuda'):
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
    prompt_list=[]
    if prompt_file.endswith('.jsonl'):
        with open(prompt_file, 'r') as f:
            for line in f:
                prompt_f = json.loads(line)
                if "imgs" in prompt_f:
                    for item in prompt_f["imgs"]:
                        prompt_list.append(item["prompt"])
                elif "prompt" in prompt_f: 
                    prompt_list.append(prompt_f["prompt"])
    elif prompt_file.endswith('.json'):
        with open(prompt_file, 'r') as f:
            prompt_f = json.load(f)
            if "imgs" in prompt_f:
                for item in prompt_f["imgs"]:
                    prompt_list.append(item["prompt"])
            elif "prompt" in prompt_f: 
                 prompt_list.append(prompt_f["prompt"])
    else:
        raise ValueError("prompt_file err")
    results = []
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    for img_name1,  prompt in tqdm(zip(files1,  prompt_list), total=len(files1)):
        img_path1 = os.path.join(folder1, img_name1)
        img1 = load_image(img_path1)
        res = calculate_clip_metrics(img1, prompt,metric)
        metrics = {
            'image_pair': (img_name1,),
            'clip_t': res['clip_t'],
            'text_prompt': res['text_prompt']
        }
        results.append(metrics)
    avg_metrics = {
        'avg_clip_t': sum(r['clip_t'] for r in results) / len(results)
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
    parser = argparse.ArgumentParser(description='clip metrics')
    parser.add_argument('--folder1', type=str, required=True,
                        help='folder1')
    parser.add_argument('--prompt_file', type=str, required=True, 
                        help='prompt list path')
    parser.add_argument('--output', type=str, required=True,
                        help='output path')
    
    args = parser.parse_args()
    
    results = calculate_metrics(args.folder1, args.prompt_file, args.output)
    print("average metrics:")
    print(json.dumps(results['average_metrics'], indent=4))