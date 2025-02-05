import os
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models

def train_model(model, train_loader: DataLoader, test_loader: DataLoader, criterion, optimizer, num_epochs=5, save_filename='model.pth', device='cuda'):

    model.to(device)

    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, save_filename)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_train = 0
        correct_train = 0
        train_loss = 0.0


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

        # Store the average metrics for this epoch
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Evaluation phase (Test set)
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item() / labels.size(1)
                test_loss += criterion(outputs, labels).item()

        test_accuracy = correct_test / total_test
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_accuracy)

        # Print the results for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)

    return train_losses, train_accuracies, test_losses, test_accuracies



def upload_pretrained(pretrained_model, add_layers=True, n_labels=5, freeze_layers=True):
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