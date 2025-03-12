import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
from torch import randint
# "openai/clip-vit-large-patch14"
def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_name).to('cuda')
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def get_image_features(image, model, processor):
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

def calculate_clip_metrics(image1, text, metric=None):
    if metric is None:
        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score = metric(image1, text)
    result = {
        "clip_t": score.detach().item(),
        "text_prompt": text
    }
    return result

def main():
    test_image1 = torch.randint(0,255,(3,224,224))
    test_image2 = torch.randint(0,255,(3,224,224))
    test_text = "a massy image"
    
    import json
    result = calculate_clip_metrics(test_image1, test_image2, test_text)
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
