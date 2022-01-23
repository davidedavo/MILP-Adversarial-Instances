from random import randint

import numpy as np
from tqdm import tqdm
from gurobipy import *
from gurobipy.gurobipy import Model, GRB, MLinExpr, MVar, max_, abs_
from torch import nn

from milp.milp_layers import LayerMILP, Conv2dMILP, LinearMILP, ReluMILP, MaxPool2dMILP


class MILPModel:
    def __init__(self, torch_model, X, y_true, n_classes):

        self.current_layer = 0
        self.input = X.detach().numpy()
        self.y_true = y_true.item()
        self.n_classes = n_classes
        self.x_adv = None
        self.d = None
        self.x = []
        self.model = Model("AdversarialInstance")
        self.mapper = TorchToMILPMapper(torch_model)
        # self.model.setParam(GRB.Param.FeasibilityTol, 1e-2)
        # self.model.setParam(GRB.Param.IntFeasTol, 1e-1)
        # self.model.setParam(GRB.Param.MIPGap, 1)
        self.layers = []

    def add_layer(self, layer: LayerMILP):
        x = layer.compute_layer(self.x[-1], self.model)
        self.layers.append(layer)
        self.model.update()
        self.x.append(x)

    def get_adversarial_label(self):
        lab = self.y_true
        while self.y_true == lab:
            lab = randint(0, self.n_classes - 1)
        return lab

    def calc_upper_bounds(self):
        pass

    def optimize(self):
        self.model.write("model.lp")
        self.model.optimize()
        res = self.x[-1].getAttr("x")
        return res

    def buildAdversarialProblem(self):
        x = self.input
        iC, iH, iW = x.shape

        self.x_adv = self.model.addMVar(shape=x.shape, vtype=GRB.CONTINUOUS, ub=self.input.max(), lb=self.input.min(),
                                        name="x_adv")
        self.x = [self.x_adv]
        self.d = self.model.addMVar(shape=x.shape, vtype=GRB.CONTINUOUS, lb=0, name="d")
        self.model.addConstrs(
            (- self.d[c, h, w] <= self.x_adv[c, h, w] - x[c, h, w] for c in range(iC) for h in range(iH) for w in
             range(iW)), name=f"absconstr_lb")
        self.model.addConstrs(
            (self.d[c, h, w] >= self.x_adv[c, h, w] - x[c, h, w] for c in range(iC) for h in range(iH) for w in
             range(iW)), name=f"absconstr_ub")
        self.mapper.map(self)
        preds = self.x[-1]
        adv_label = self.get_adversarial_label()
        self.model.addConstrs((preds[adv_label] >= preds[i] for i in range(self.n_classes) if i != adv_label),
                              name=f"pred_constr")
        self.model.setObjective(self.d.sum(), GRB.MINIMIZE)

    def buildInferenceProblem(self):
        self.x = [self.input]
        self.mapper.map(self)
        self.model.setObjective(sum([x.sum() for x in self.x[1:]]), GRB.MINIMIZE)


class TorchToMILPMapper:
    def __init__(self, torch_model, ):
        self.layers = torch_model.get_layers()

    def map(self, model: MILPModel):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                assert layer.stride == (1, 1)
                assert layer.padding == (0, 0)
                model.add_layer(Conv2dMILP(i, layer.weight.detach().numpy(), layer.bias.detach().numpy()))
            elif isinstance(layer, nn.Linear):
                model.add_layer(LinearMILP(i, layer.weight.detach().numpy(), layer.bias.detach().numpy()))
            elif isinstance(layer, nn.ReLU):
                model.add_layer(ReluMILP(i))
            elif isinstance(layer, nn.MaxPool2d):
                model.add_layer(MaxPool2dMILP(i, layer.kernel_size, layer.stride))
            elif isinstance(layer, nn.Flatten):
                pass
            else:
                raise TypeError("Layer not supported")
