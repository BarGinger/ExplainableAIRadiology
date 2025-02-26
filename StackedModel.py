import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class StackedModel(nn.Module):
    def __init__(self, n_labels, freeze_layers):
        super(StackedModel, self).__init__()
        self.base_model1 = models.mobilenet_v2(pretrained=True)
        self.base_model2 = models.densenet169(pretrained=True)

        # Modify the first convolutional layer to accept 1-channel input for MobileNetV2
        self.base_model1.features[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Modify the first convolutional layer to accept 1-channel input for DenseNet169
        self.base_model2.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if freeze_layers:
            for param in self.base_model1.parameters():
                param.requires_grad = False
            for param in self.base_model2.parameters():
                param.requires_grad = False

        
        # # Ensure the modified first layers are not frozen
        # self.base_model1.features[0][0].weight.requires_grad = True
        # self.base_model2.features.conv0.weight.requires_grad = True

        self.base_model1.classifier = nn.Identity()
        self.base_model2.classifier = nn.Identity()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1664 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.base_model1.features(x)
        x1 = self.global_avg_pool(x1)
        x1 = torch.flatten(x1, 1)

        x2 = self.base_model2.features(x)
        x2 = self.global_avg_pool(x2)
        x2 = torch.flatten(x2, 1)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
    

# Grad-CAM implementation
def apply_gradcam_stacked_densenet_mobilenet(model, image_path, transform):
    """
    Apply Grad-CAM to the specified image using the provided model.
    
    Args:
        model: The trained model.
        image_path: Path to the image for visualization.
        transform: Image transformation pipeline.
        target_layers: List of target layers for Grad-CAM.
    
    Returns:
        Tuple of (resized original image, Grad-CAM heatmap, overlayed image).
    """

    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    target_layers = [model.base_model1.features.denseblock4.denselayer32, model.base_model2.features[-1][0]]


    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img, dtype=np.float32) / 255.0

    input_tensor = transform(img).unsqueeze(0)

    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    if original_img.shape[:2] != grayscale_cam.shape:
        original_img_resized = cv2.resize(original_img, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
    else:
        original_img_resized = original_img

    overlay = show_cam_on_image(original_img_resized, grayscale_cam, use_rgb=True)

    return original_img_resized, overlay

def upload_stacked_models(n_labels=1, freeze_layers=True):
    """
    Create a stacked model using DenseNet169 and MobileNetV2 as base models.

    Parameters:
    - n_labels: Number of output labels (classes).
    - freeze_layers: Boolean indicating whether to freeze the original layers.

    Returns:
    - The stacked model.
    """
    model = StackedModel(n_labels, freeze_layers)
    return model