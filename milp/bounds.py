import torch.nn as nn


def bound_propagation(layers, initial_bound):
    """
    Function that performs upper and lower bounds over the network.
    Args:
        layers: List of PyTorch layers
        initial_bound: Tuple containing initial lower and upper bounds (l,b)

    Returns:
    Performed bounds.
    """

    l, u = initial_bound
    bounds = []

    for layer in layers:
        if isinstance(layer, nn.Flatten):
            l_ = nn.Flatten()(l)
            u_ = nn.Flatten()(u)
        elif isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()
                  + layer.bias[:, None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()
                  + layer.bias[:, None]).t()
        elif isinstance(layer, nn.Conv2d):
            l_ = (nn.functional.conv2d(l, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(u, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

            u_ = (nn.functional.conv2d(u, layer.weight.clamp(min=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  nn.functional.conv2d(l, layer.weight.clamp(max=0), bias=None,
                                       stride=layer.stride, padding=layer.padding,
                                       dilation=layer.dilation, groups=layer.groups) +
                  layer.bias[None, :, None, None])

        elif isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)

        bounds.append((l_, u_))
        l, u = l_, u_
    return bounds


import cvxpy as cp


def form_milp(model, c, initial_bounds, bounds):
    linear_layers = [(layer, bound) for layer, bound in zip(model, bounds) if isinstance(layer, nn.Linear)]
    d = len(linear_layers) - 1

    # create cvxpy variables
    z = ([cp.Variable(layer.in_features) for layer, _ in linear_layers] +
         [cp.Variable(linear_layers[-1][0].out_features)])
    v = [cp.Variable(layer.out_features, boolean=True) for layer, _ in linear_layers[:-1]]

    # extract relevant matrices
    W = [layer.weight.detach().cpu().numpy() for layer, _ in linear_layers]
    b = [layer.bias.detach().cpu().numpy() for layer, _ in linear_layers]
    l = [l[0].detach().cpu().numpy() for _, (l, _) in linear_layers]
    u = [u[0].detach().cpu().numpy() for _, (_, u) in linear_layers]
    l0 = initial_bound[0][0].view(-1).detach().cpu().numpy()
    u0 = initial_bound[1][0].view(-1).detach().cpu().numpy()

    # add ReLU constraints
    constraints = []
    for i in range(len(linear_layers) - 1):
        constraints += [z[i + 1] >= W[i] @ z[i] + b[i],
                        z[i + 1] >= 0,
                        cp.multiply(v[i], u[i]) >= z[i + 1],
                        W[i] @ z[i] + b[i] >= z[i + 1] + cp.multiply((1 - v[i]), l[i])]

    # final linear constraint
    constraints += [z[d + 1] == W[d] @ z[d] + b[d]]

    # initial bound constraints
    constraints += [z[0] >= l0, z[0] <= u0]

    return cp.Problem(cp.Minimize(c @ z[d + 1]), constraints), (z, v)