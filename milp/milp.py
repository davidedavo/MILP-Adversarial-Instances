import numpy as np
from tqdm import tqdm
from gurobipy import *
from gurobipy.gurobipy import Model, GRB, MLinExpr, MVar, max_
from torch import nn


class MILPModel:

    def __init__(self, torch_model, X):
        self.torch_model = torch_model
        self.current_layer = 0
        self.input = X.detach().numpy()
        self.a = []
        self.s = []
        self.z = []
        self.model = Model("AdversarialInstance")
        self.layers = torch_model.get_layers()
        self.build_milp()

    def build_milp(self):
        input = self.input
        for i, layer in enumerate(self.layers):
            self.current_layer += 1
            if isinstance(layer, nn.Conv2d):
                if len(self.layers) > i + 1 and isinstance(self.layers[i + 1], nn.ReLU):
                    self.add_conv2D_layer(layer, input, relu=True)
                else:
                    self.add_conv2D_layer(layer, input, relu=False)
            elif isinstance(layer, nn.Linear):
                if len(self.layers) > i + 1 and isinstance(self.layers[i + 1], nn.ReLU):
                    self.add_fc_layer(layer, input, relu=True)
                else:
                    self.add_fc_layer(layer, input, relu=False)
            elif isinstance(layer, nn.ReLU):
                print("relu")
            elif isinstance(layer, nn.MaxPool2d):
                self.add_maxpool2D_layer(layer, input)
            else:
                raise TypeError("Layer not supported")
            input = self.a[-1]

        self.model.setObjective(sum([a.sum() for a in self.a]), GRB.MINIMIZE)
        # self.model.write("model.lp")

    def optimize(self):
        self.model.optimize()
        res = self.a[-1].getAttr("x")
        return res

    def add_conv2D_layer(self, conv2d: nn.Conv2d, x, relu):
        assert conv2d.padding == (0, 0)
        assert conv2d.stride == (1, 1)

        iC = conv2d.in_channels
        oC = conv2d.out_channels
        kH, kW = conv2d.kernel_size

        assert iC == x.shape[0]
        _, iH, iW = x.shape

        oH = iH - kH + 1
        oW = iW - kW + 1

        kernel = conv2d.weight.data.detach().numpy()
        bias = conv2d.bias.data.detach().numpy()
        print(kernel.shape)

        self.a.append(
            self.model.addMVar(shape=(oC, oH, oW), vtype=GRB.CONTINUOUS, name=f'a_{self.current_layer}',
                               lb=0))  # activation
        y = self.a[-1]

        if relu:
            self.s.append(
                self.model.addMVar(shape=(oC, oH, oW), vtype=GRB.CONTINUOUS,
                                   name=f's_{self.current_layer}', lb=0))  # slack
            self.z.append(
                self.model.addMVar(shape=(oC, oH, oW), vtype=GRB.BINARY,
                                   name=f'z_{self.current_layer}'))  # indicator

            z = self.z[-1]
            s = self.s[-1]

            for h in range(oH):
                for w in range(oW):
                    self.model.addConstr((
                            sum([kernel[:, i, j, :] @ x[i, h + j, w:kW + w] for i in range(iC)
                                 for j in range(kH)]) == y[:, h, w] - s[:, h, w])
                        , name=f"conv_{self.current_layer}_{h}_{w}")  # TODO MANCA BIAS

                    self.model.addConstrs(
                        ((z[c, h, w] == 1) >> (y[c, h, w] <= 0) for c in range(oC)),
                        name=f"ind_a_{self.current_layer}_{h}_{w}")

                    self.model.addConstrs(
                        ((z[c, h, w] == 0) >> (s[c, h, w] <= 0) for c in range(oC)),
                        name=f"ind_s_{self.current_layer}_{h}_{w}")
        else:
            for h in range(oH):
                for w in range(oW):
                    self.model.addConstr((
                            sum([kernel[:, i, j, :] @ x[i, h + j, w:kW + w] for i in range(iC)
                                 for j in range(kH)]) == y[:, h, w]), name=f"conv_{self.current_layer}_{h}_{w}")

    def add_fc_layer(self, layer, x, relu):
        bias = layer.bias.data.detach().numpy()
        weights = layer.weight.data.detach().numpy()
        in_features = weights.shape[1]
        out_features = layer.out_features

        self.a.append(self.model.addMVar(shape=(out_features,1), vtype=GRB.CONTINUOUS, name='x'))  # activation

        y = self.a[-1]
        if relu:
            self.s.append(self.model.addMVar(shape=(out_features,1), vtype=GRB.CONTINUOUS, name='s'))  # slack
            self.z.append(self.model.addMVar(shape=(out_features,1), vtype=GRB.BINARY, name='z'))  # indicator

            z = self.z[-1]
            s = self.s[-1]

            self.model.addConstr((weights @ x + bias == y - s), name=f"fc_{self.current_layer}")

            self.model.addConstrs(
                ((z[i] == 1) >> (y[i] <= 0) for i in range(out_features)), name=f"indicator_a_{self.current_layer}")
            self.model.addConstrs(
                ((z[i] == 0) >> (s[i] <= 0) for i in range(out_features)), name=f"indicator_s{self.current_layer}")
        else:
            self.model.addConstr((weights @ x + bias == y), name=f"fc_{self.current_layer}")


    def add_maxpool2D_layer(self, layer: nn.MaxPool2d, x):
        assert layer.padding == 0
        k = layer.kernel_size
        stride = layer.stride

        C, iH, iW = x.shape

        oH = (iH - k) // stride + 1
        oW = (iW - k) // stride + 1

        self.current_layer += 1
        self.a.append(
            self.model.addMVar(shape=(C, oH, oW), vtype=GRB.CONTINUOUS, name=f'a_{self.current_layer}', lb=0))
        y = self.a[-1]

        self.model.addConstrs(
            (y[c, h, w] == max_([x[c, h * stride + i, w * stride + j] for i in range(k) for j in range(k)])
             for c in range(C) for h in range(oH) for w in range(oW)))
