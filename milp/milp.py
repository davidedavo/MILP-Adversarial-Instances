from random import randint

import numpy as np
from tqdm import tqdm
from gurobipy import *
from gurobipy.gurobipy import Model, GRB, MLinExpr, MVar, max_, abs_
from torch import nn


def element_at(self, index):
    if not isinstance(index, int):
        raise TypeError("index must be an integer")
    if len(self.shape) == 1:
        return self.__getitem__(index)
    shape = self.shape
    idxs = []
    for i, dim in enumerate(shape):
        stride = int(np.prod(self.shape[i + 1:]))
        idxs.append(index // stride)
        index -= idxs[-1] * stride
    return self.__getitem__(tuple(idxs))


class MILPModel:
    def __init__(self, torch_model, X, y_true, n_classes, include_last_layer=False):
        MVar.element_at = element_at
        self.torch_model = torch_model
        self.current_layer = 0
        self.input = X.detach().numpy()
        self.y_true = y_true.item()
        self.n_classes = n_classes
        self.x_adv = None
        self.d = None
        self.include_last_layer = include_last_layer
        self.a = []
        self.s = []
        self.z = []
        self.model = Model("AdversarialInstance")
        # self.model.setParam(GRB.Param.FeasibilityTol, 1e-2)
        # self.model.setParam(GRB.Param.IntFeasTol, 1e-1)
        # self.model.setParam(GRB.Param.MIPGap, 1)
        self.layers = torch_model.get_layers()
        self.build_milp()

    def get_adversarial_label(self):
        lab = self.y_true
        while self.y_true == lab:
            lab = randint(0, self.n_classes - 1)
        return lab

    def calc_upper_bounds(self):
        pass

    def build_milp(self):
        iC, iH, iW = self.input.shape
        self.x_adv = self.addContinuous(self.input.shape, name_prefix="x_adv", lb=float("-inf"))
        self.d = self.addContinuous(self.input.shape, name_prefix="d", lb=0)
        self.model.addConstrs((-self.d[c,h,w] <= self.x_adv[c,h,w] - self.input[c,h,w] for c in range(iC) for h in range(iH) for w in range(iW)), name=f"absconstr_lb")
        self.model.addConstrs((self.x_adv[c,h,w] - self.input[c,h,w] <= self.d[c,h,w] for c in range(iC) for h in range(iH) for w in range(iW)), name=f"absconstr_ub")
        input = self.x_adv
        input = self.input
        layers = self.layers if self.include_last_layer else self.layers[:-1]
        for i, layer in enumerate(layers):
            self.current_layer += 1
            if isinstance(layer, nn.Conv2d):
                if len(layers) > i + 1 and isinstance(layers[i + 1], nn.ReLU):
                    self.add_conv2D_layer(layer, input, relu=True)
                else:
                    self.add_conv2D_layer(layer, input, relu=False)
            elif isinstance(layer, nn.Linear):
                if len(layers) > i + 1 and isinstance(layers[i + 1], nn.ReLU):
                    self.add_fc_layer(layer, input, relu=True)
                else:
                    self.add_fc_layer(layer, input, relu=False)
            elif isinstance(layer, nn.ReLU):
                pass
            elif isinstance(layer, nn.MaxPool2d):
                self.add_maxpool2D_layer(layer, input)
            elif isinstance(layer, nn.Flatten):
                pass
            else:
                raise TypeError("Layer not supported")
            input = self.a[-1]

        preds = self.a[-1]
        adv_label = self.get_adversarial_label()
        self.model.addConstrs((preds[adv_label] >= preds[i] for i in range(self.n_classes) if i != adv_label),
                              name=f"pred_constr")
        self.model.setObjective(self.d.sum(), GRB.MINIMIZE)
        #x = self.a[:-1] if self.include_last_layer else self.a
        #self.model.setObjective(sum([x.sum() for x in self.a]), GRB.MINIMIZE)
        #self.model.write("model.lp")

    def optimize(self):
        self.model.optimize()
        res = self.a[-1].getAttr("x")
        if not self.include_last_layer:
            res = self.compute_last_layer(res)
        return res

    def setAdversarialProblem(self):
        x = self.input.reshape(-1, )
        dim = x.shape[0]

        self.x_adv = self.addContinuous((dim,), name_prefix="adv_instance", ub=self.input.max(), lb=self.input.min())
        self.d = self.addContinuous((dim,), name_prefix="d", lb=0)
        self.model.addConstrs((- self.d[i] <= self.x_adv[i] - x[i] for i in range(dim)), name=f"absconstr_lb")
        self.model.addConstrs((self.d[i] >= self.x_adv[i] - x[i] for i in range(dim)), name=f"absconstr_ub")
        preds = self.a[-1]
        adv_label = self.get_adversarial_label()
        self.model.addConstrs((preds[adv_label] >= 1.2 * preds[i] for i in range(self.n_classes) if i != adv_label),
                              name=f"pred_constr")
        self.model.setObjective(self.d.sum() + sum([x.sum() for x in self.a]), GRB.MINIMIZE)

    def addContinuous(self, shape, name_prefix, lb=None, ub=None):
        return self.model.addMVar(shape=shape, vtype=GRB.CONTINUOUS, name=f'{name_prefix}_{self.current_layer}', lb=lb,
                                  ub=ub)

    def addBinary(self, shape, name_prefix, *args, **kwargs):
        return self.model.addMVar(shape=shape, vtype=GRB.BINARY, name=f'{name_prefix}_{self.current_layer}')

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

        lb = 0 if relu else float('-inf')
        self.a.append(self.addContinuous(shape=(oC, oH, oW), name_prefix=f'a', lb=lb))

        y = self.a[-1]

        s = None
        if relu:
            self.s.append(self.addContinuous(shape=(oC, oH, oW), name_prefix=f's', lb=0))  # slack
            self.z.append(self.addBinary(shape=(oC, oH, oW), name_prefix=f'z', lb=0))  # indicator

            z = self.z[-1]
            s = self.s[-1]

            self.model.addConstrs(
                ((z[c, h, w] == 1) >> (y[c, h, w] <= 0) for c in range(oC) for h in range(oH) for w in range(oW)),
                name=f"ind_a_{self.current_layer}")

            self.model.addConstrs(
                ((z[c, h, w] == 0) >> (s[c, h, w] <= 0) for c in range(oC) for h in range(oH) for w in range(oW)),
                name=f"ind_s_{self.current_layer}")

        for c in range(oC):
            for h in range(oH):
                for w in range(oW):
                    rhs = y[c, h, w] - s[c, h, w] if s else y[c, h, w]
                    self.model.addConstr((
                            sum([kernel[c, i, k_h, k_w] * x[i, h + k_h, w + k_w] for i in range(iC) for k_h in range(kH)
                                 for k_w in range(kW)])
                            + bias[c] == rhs), name=f"conv_{self.current_layer}_{c}_{h}_{w}")

    def add_fc_layer(self, layer, x, relu):
        bias = layer.bias.data.detach().numpy()
        weights = layer.weight.data.detach().numpy()
        in_features = weights.shape[1]
        out_features = layer.out_features

        lb = 0 if relu else float('-inf')
        self.a.append(self.addContinuous(shape=(out_features,), name_prefix='a', lb=lb))  # activation
        y = self.a[-1]
        s = None
        if relu:
            self.s.append(self.addContinuous(shape=(out_features,), name_prefix='s', lb=0))  # slack
            self.z.append(self.addBinary(shape=(out_features,), name_prefix='z', lb=0))  # indicator

            z = self.z[-1]
            s = self.s[-1]

            self.model.addConstrs(
                ((z[i] == 1) >> (y[i] <= 0) for i in range(out_features)), name=f"indicator_a_{self.current_layer}")
            self.model.addConstrs(
                ((z[i] == 0) >> (s[i] <= 0) for i in range(out_features)), name=f"indicator_s{self.current_layer}")

        for i in range(out_features):
            rhs = y[i] - s[i] if s else y[i]
            self.model.addConstr((sum(weights[i, j] * x.element_at(j) for j in range(in_features)) + bias[i] == rhs),
                                 name=f"fc_{self.current_layer}_{i}")

    def add_maxpool2D_layer(self, layer: nn.MaxPool2d, x):
        assert layer.padding == 0
        k = layer.kernel_size
        stride = layer.stride

        C, iH, iW = x.shape

        oH = (iH - k) // stride + 1
        oW = (iW - k) // stride + 1

        self.a.append(
            self.addContinuous(shape=(C, oH, oW), name_prefix=f'a', lb=0))
        y = self.a[-1]

        self.model.addConstrs(
            (y[c, h, w] == max_([x[c, h * stride + i, w * stride + j] for i in range(k) for j in range(k)])
             for c in range(C) for h in range(oH) for w in range(oW)))

    # def add_flatten_layer(self, layer, x):
    #     iC, iH, iW = x.shape
    #     dim = iC * iH * iW
    #     self.a.append(self.addContinuous((dim,), name_prefix="a", lb=0))
    #     y = self.a[-1]
    #     ind = 0
    #     for c in range(iC):
    #         for h in range(iH):
    #             for w in range(iW):
    #                 self.model.addConstr((x[c, h, w] == y[ind]), name=f"flatten_{self.current_layer}_{ind}")
    #                 ind += 1

    def compute_last_layer(self, x):
        layer = self.layers[-1]
        bias = layer.bias.data.detach().numpy()
        weights = layer.weight.data.detach().numpy()
        x = x.reshape(-1)
        return weights @ x + bias
