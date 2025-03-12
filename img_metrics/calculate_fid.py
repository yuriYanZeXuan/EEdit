import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img)
def load_single_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img

def get_inception_model(device):
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    def forward_hook(module, input, output):
        module.output = output
    
    inception.avgpool.register_forward_hook(forward_hook)
    return inception

def get_activations_for_single_image(image, model, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(img_tensor)
        act = model.avgpool.output
        act = act.squeeze(-1).squeeze(-1)  # shape: [2048]
        activations = act.cpu().numpy()
    return activations

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0])*eps).dot(sigma2 + np.eye(sigma2.shape[0])*eps))
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean)
    return fid

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_fid_for_two_images(img_path1, img_path2, device='cuda'):
    img1 = load_single_image(img_path1)
    img2 = load_single_image(img_path2)

    model = get_inception_model(device)

    act1 = get_activations_for_single_image(img1, model, device=device)
    act2 = get_activations_for_single_image(img2, model, device=device)

    act1 = act1[None, :]
    act2 = act2[None, :]

    mu1, sigma1 = calculate_activation_statistics(act1)
    mu2, sigma2 = calculate_activation_statistics(act2)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def calculate_fid_for_test(img1, img2, device='cuda'):
    import numpy as np
    model = get_inception_model(device)

    img1 = transforms.ToPILImage()(img1.cpu())
    img2 = transforms.ToPILImage()(img2.cpu())

    act1 = get_activations_for_single_image(img1, model, device=device)
    act2 = get_activations_for_single_image(img2, model, device=device)

    act1 = act1.reshape(1, -1)
    act2 = act2.reshape(1, -1)

    mu1, sigma1 = calculate_activation_statistics(act1)
    mu2, sigma2 = calculate_activation_statistics(act2)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

if __name__ == "__main__":
    image1_path = 'xxx'
    image2_path = 'xxx'
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)
    fid_score=calculate_fid_for_test(img1, img2, device='cuda')
    print("FID:", fid_score)
