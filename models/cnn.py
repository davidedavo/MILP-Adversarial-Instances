import torch
import torch.nn as nn


def get_conv_block(in_channels, layers_dim, kernel_sizes):
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * len(layers_dim)
    elif isinstance(kernel_sizes, list):
        if len(layers_dim) != len(kernel_sizes):
            raise AttributeError("layers_dim and kernel_sizes are incompatible")
    else:
        raise AttributeError("kernel_sizes must be int or list")

    in_ch = in_channels
    modules = []
    for dim, k_size in zip(layers_dim, kernel_sizes):
        modules += [
            nn.Conv2d(in_channels=in_ch, out_channels=dim, kernel_size=k_size),
            nn.ReLU(),
        ]
        in_ch = dim
    return nn.Sequential(*modules)


def get_fc_block(layers_dim, n_classes):
    modules = [nn.Flatten()]
    for i in range(len(layers_dim)):
        if i == 0:
            modules += [nn.LazyLinear(layers_dim[i])]
        else:
            modules += [nn.Linear(layers_dim[i - 1], layers_dim[i])]
        modules += [nn.ReLU()]

    modules += [nn.LazyLinear(n_classes)]
    return nn.Sequential(*modules)


class CNN(nn.Module):

    def __init__(self, in_channels, params, n_classes):
        super().__init__()
        self.conv_block = get_conv_block(in_channels, params["conv_layers"], params["kernel_sizes"])
        if params["max_pool2d"]:
            self.max_pool2d = nn.MaxPool2d(params["max_pool2d"], params["max_pool2d"])
        self.fc_block = get_fc_block(params["linear_layers"], n_classes)

    def forward(self, x):
        x = self.conv_block(x)
        if self.max_pool2d:
            x = self.max_pool2d(x)
        x = self.fc_block(x)
        return x


class DNN(nn.Module):
    def __init__(self, in_channels, params, n_classes):
        super().__init__()
        self.fc_block = get_fc_block(params["linear_layers"], n_classes)

    def forward(self, x):
        x = self.fc_block(x)
        return x
