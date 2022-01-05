import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, in_channels, layer_1_dim, layer_2_dim, n_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=layer_1_dim, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=layer_1_dim, out_channels=layer_2_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.fc_block = nn.LazyLinear(n_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_block(x)
        return x