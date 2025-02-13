import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the output size after convolutions and pooling
        # The input image size is (1, 128, 251)
        # self.fc1_input_height = 128 // 2 // 2  # Two max-pooling layers reduce height by factor of 2 each
        # self.fc1_input_width = 251 // 2 // 2   # Two max-pooling layers reduce width by factor of 2 each


        self.fc1_input_size = 100352
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
