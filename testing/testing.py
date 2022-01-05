import torch
from milp.milp import convolution, MILPModel
import numpy as np
import torch.nn.functional as F


def get_tensor(np_array):
    return torch.from_numpy(np_array)


def test_conv(input, kernel, bias=None):
    input_tensor = get_tensor(input).unsqueeze(0)
    kernel_tensor = get_tensor(kernel)
    bias_tensor = get_tensor(bias) if bias else None
    A = F.conv2d(input_tensor, kernel_tensor, bias_tensor).squeeze(0).numpy()
    B = convolution(input, kernel, bias)
    X = A == B
    return np.allclose(A, B)


def test_milp(input, kernel):
    m = MILPModel( ,input)
    m.add_conv_layer(32,253,253,kernel)


if __name__ == "__main__":
    X = np.random.rand(3, 255, 255)
    kernel = np.random.rand(32, 3, 3, 3)
    if test_conv(X, kernel):
        print("Test passed")
    else:
        print("Test failed")


    test_milp(X, kernel)
