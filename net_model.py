import torch.nn as nn
import torch

"""
A CNN model for object detection.
Methods:
    forward(x):
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output predictions.
"""

class CnnModel(nn.Module):
    def __init__(self, dropout):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout)

        self.flatten_size = 128 * 52 * 52
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        out = self.fc2(x)

        return out

