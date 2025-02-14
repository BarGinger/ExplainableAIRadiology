import torch.nn as nn
import torchvision.models as models

def upload_pretrained(pretrained_model, add_layers=True, n_labels=5, freeze_layers=True):
    """
    Modify a pre-trained model by adding custom layers and optionally freezing the original layers.

    Parameters:
    - pretrained_model: The pre-trained model to be modified.
    - add_layers: Boolean indicating whether to add custom layers.
    - n_labels: Number of output labels (classes).
    - freeze_layers: Boolean indicating whether to freeze the original layers.

    Returns:
    - The modified model.
    """
    if freeze_layers:
        for param in pretrained_model.parameters():
            param.requires_grad = False

    if add_layers:
        pretrained_model.fc = nn.Sequential(
            nn.Linear(pretrained_model.fc.in_features, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, n_labels)
        )

    return pretrained_model

def upload_pretrained_vit(vit_model, add_layers=True, n_labels=5, freeze_layers=True):
    """
    Modify a pre-trained Vision Transformer (ViT) model by adding custom layers and optionally freezing the original layers.

    Parameters:
    - vit_model: The pre-trained ViT model to be modified.
    - add_layers: Boolean indicating whether to add custom layers.
    - n_labels: Number of output labels (classes).
    - freeze_layers: Boolean indicating whether to freeze the original layers.

    Returns:
    - The modified ViT model.
    """
    if freeze_layers:
        for param in vit_model.parameters():
            param.requires_grad = False
            
    if add_layers:
        # Access the first module inside the Sequential container to get in_features
        if isinstance(vit_model.heads, nn.Sequential):
            in_features = vit_model.heads[0].in_features
        else:
            in_features = vit_model.heads.in_features

        vit_model.heads = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, n_labels)
        )
    
    return vit_model