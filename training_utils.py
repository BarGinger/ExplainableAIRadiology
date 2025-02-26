## Helper functions for training PyTorch models.

import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from preprocessing_utils import get_class_names


def train_model(model, train_loader: DataLoader, test_loader: DataLoader, criterion, optimizer, scheduler, num_epochs=5, model_name='model', device='cuda', patience=5):
    """
    Train a PyTorch model with the given parameters.

    Parameters:
    - model: The PyTorch model to be trained.
    - train_loader: DataLoader for the training dataset.
    - test_loader: DataLoader for the validation dataset.
    - criterion: The loss function.
    - optimizer: The optimizer.
    - scheduler: The learning rate scheduler.
    - num_epochs: Number of epochs to train the model.
    - model_name: model name to save the trained model.
    - device: Device to train the model on ('cpu' or 'cuda').
    - patience: Number of epochs to wait for improvement before stopping early.

    Returns:
    - train_losses: List of training losses for each epoch.
    - train_accuracies: List of training accuracies for each epoch.
    - test_losses: List of validation losses for each epoch.
    - test_accuracies: List of validation accuracies for each epoch.
    - test_aucs: List of validation AUCs for each epoch.
    """
    model.to(device)

    current_dir = os.getcwd()
    save_filename = f"{model_name}.pkl"
    save_path = f"finetuned_models/{save_filename}"

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    test_aucs = []

    best_loss = float('inf')
    epochs_no_improve = 0

    print(f"Training {model_name} model for {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        total_train = 0
        correct_train = 0
        train_loss = 0.0

        # Add a progress bar for the training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Accuracy and loss calculation
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                # Calculate accuracy per sample (averaged over labels)
                correct_train += (predicted == labels).sum().item() / labels.size(1)
                total_train += labels.size(0)
                train_loss += loss.item()

                pbar.update(1)

        # Store the average metrics for this epoch
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Evaluation phase (Test set)
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        all_labels = []
        all_outputs = []

        with tqdm(total=len(test_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Evaluation", unit="batch") as pbar:
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item() / labels.size(1)
                    test_loss += criterion(outputs, labels).item()

                    all_labels.append(labels.cpu().numpy())
                    all_outputs.append(outputs.cpu().numpy())

                    pbar.update(1)

        test_accuracy = correct_test / total_test
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_accuracy)

        # Calculate AUC
        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)
        test_auc = roc_auc_score(all_labels, all_outputs)
        test_aucs.append(test_auc)

        # Print the results for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}, Test AUC: {test_aucs[-1]:.4f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_outputs)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - Epoch {epoch + 1}')
        plt.legend(loc="lower right")
        plt.show()

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model for epoch {epoch + 1} for model {model_name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} for model {model_name}")
                break

        # Adjust the learning rate based on the validation loss
        scheduler.step(test_loss)

    save_model(model, model_name=model_name, device=device)

    return train_losses, train_accuracies, test_losses, test_accuracies, test_aucs

def upload_pretrained_densenet169(pretrained_model, add_layers=True, n_labels=1, freeze_layers=True):
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
        in_features = pretrained_model.classifier.in_features
        pretrained_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_labels)
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

def upload_pretrained_densenet121(pretrained_model, add_layers=True, n_labels=1, freeze_layers=True):
    if freeze_layers:
        for param in pretrained_model.parameters():
            param.requires_grad = False

    if add_layers:
        in_features = pretrained_model.classifier.in_features
        pretrained_model.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, n_labels)
        )

    return pretrained_model


def upload_stacked_models(n_labels=1, freeze_layers=True):
    """
    Create a stacked model using DenseNet169 and MobileNetV2 as base models.

    Parameters:
    - n_labels: Number of output labels (classes).
    - freeze_layers: Boolean indicating whether to freeze the original layers.

    Returns:
    - The stacked model.
    """
    class StackedModel(nn.Module):
        def __init__(self, n_labels, freeze_layers):
            super(StackedModel, self).__init__()
            self.base_model1 = models.densenet169(pretrained=True)
            self.base_model2 = models.mobilenet_v2(pretrained=True)

            if freeze_layers:
                for param in self.base_model1.parameters():
                    param.requires_grad = False
                for param in self.base_model2.parameters():
                    param.requires_grad = False

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

    model = StackedModel(n_labels, freeze_layers)
    return model

def save_model(model, model_name, device='cuda'):
    """
    Save a PyTorch model to a file using pickle and onnx.

    Parameters:
    - model: The PyTorch model to be saved.
    - model_name: The name of the model to save.
    """
    # # Save the trained model using pickle
    # with open(filename, 'wb') as f:
    #     pickle.dump(model, f)

    pickle_path = f"finetuned_models/{model_name}.pkl"
    print(f"Model pickled saved in {pickle_path}")
    # Save the model using pickle
    torch.save(model, pickle_path)

    pth_path = f"finetuned_models/{model_name}.pth"
    print(f"Model pth saved in {pth_path}")
    torch.save(model.state_dict(), pth_path) 

    # Define dummy input for ONNX export (batch size 1, 3 channels, 224x224 image size)
    dummy_input = torch.randn(1, 1, 224, 224).to(device)  # Move dummy input to GPU

    # Export the model to ONNX format
    onnx_path = f"finetuned_models/{model_name}.onnx"
    torch.onnx.export(model, dummy_input, onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=11)

    print(f"Model saved as onnx in {onnx_path}")


def load_model(filename, device='cuda'):
    """
    Load a PyTorch model from a file using the specified model class.

    Parameters:
    - model_class: The class of the model to be loaded.
    - filename: The name of the file to load the model from.
    - device: Device to load the model on ('cpu' or 'cuda').

    Returns:
    - The loaded PyTorch model.
    """
    path = f"finetuned_models/{filename}"
    loaded_model = torch.load(path, map_location=device)
    return loaded_model


def predict_model(model, loader, device='cuda'):
    """
    Predict using a trained PyTorch model.

    Parameters:
    - model: The trained PyTorch model.
    - loader: DataLoader for the dataset to predict on.
    - device: Device to run the predictions on ('cpu' or 'cuda').

    Returns:
    - predictions: Numpy array of predictions.
    - labels: Numpy array of true labels.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

# Function to plot ROC curve
def plot_roc_curve(labels, preds, model_name):
    """
    Plot the ROC curve for the given model.

    Parameters:
    - labels: Numpy array of true labels.
    - preds: Numpy array of predicted probabilities.
    - model_name: Name of the model.

    Returns:
    - None
    """
    class_names = get_class_names()
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for model {model_name}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
