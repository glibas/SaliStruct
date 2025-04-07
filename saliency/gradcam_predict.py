# File: saliency/gradcam_predict.py

import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ExifTags
import numpy as np
import cv2
from tqdm import tqdm



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval().cpu()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.cpu()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam = torch.nn.functional.relu(grad_cam)
        grad_cam = torch.nn.functional.interpolate(grad_cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        return grad_cam


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])



def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((224, 224))
    tensor = transform(image_resized).unsqueeze(0)
    return tensor, image.size, image


def predict_saliency(model, grad_cam, image_tensor, original_size):
    saliency = grad_cam(image_tensor)
    saliency = (saliency * 255).astype(np.uint8)
    saliency = cv2.resize(saliency, (original_size[0], original_size[1]), interpolation=cv2.INTER_LINEAR)  # Resize to (W, H)
    return saliency


def generate_saliency_maps(input_dir, output_dir, model, grad_cam):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(os.listdir(input_dir)):
        if file.lower().endswith(('.png', '.jpg')):
            img_path = os.path.join(input_dir, file)
            image_tensor, original_size, _ = preprocess_image(img_path)
            saliency_map = predict_saliency(model, grad_cam, image_tensor, original_size)
            out_path = os.path.join(output_dir, file)
            cv2.imwrite(out_path, saliency_map)