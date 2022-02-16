import numpy as np
from gurobipy.gurobipy import Model, GRB
from torch import nn

from milp.milp_layers import LayerMILP, Conv2dMILP, LinearMILP, ReluMILP, MaxPool2dMILP, FlattenMILP
from milp.milp_variable import Variable


class MILPModel:
    def __init__(self, torch_model, x: np.ndarray, y_true: int, n_classes, bounds):
        self.current_layer = 0
        self.input = x
        self.y_true = y_true
        self.n_classes = n_classes
        self.x_adv = None
        self.d = None
        self.model = Model("AdversarialInstance")
        self.mapper = TorchToMILPMapper(torch_model)

        self.model.setParam(GRB.Param.Presolve, 2)
        self.model.setParam(GRB.Param.LogToConsole, 0)
        self.model.setParam(GRB.Param.LogFile, "output.log")

        self.layers = []
        self.vars = []
        self.input_bounds = None
        self.set_input_bounds(bounds, eps=0.)

        self.adv_constrs = {}

    def set_input_bounds(self, bounds, eps=0.):
        lb, ub = bounds
        if eps == 0.:
            self.input_bounds = (bounds[0] * np.ones(self.input.shape), bounds[1] * np.ones(self.input.shape))
        else:
            print("--- EPSILON ---")
            delta = ub - lb
            var = delta * eps
            self.input_bounds = (self.input - var, self.input + var)

    def add_layer(self, layer: LayerMILP):
        var = layer.compute_layer(self.model, self.vars[-1])
        self.layers.append(layer)
        self.vars.append(var)
        self.model.update()

    def optimize(self):
        self.model.write("model.lp")
        self.model.optimize()
        status = self.model.Status
        if status == GRB.OPTIMAL:
            out_var = self.vars[-1]
            res = out_var.x.getAttr("x")
            adv = self.x_adv.getAttr("x")
            diff = self.d.getAttr("x").sum()
            return res, adv, diff
        return None

    def setup(self):
        x = self.input

        self.x_adv = self.model.addMVar(shape=x.shape, vtype=GRB.CONTINUOUS, lb=self.input_bounds[0],
                                        ub=self.input_bounds[1], name="x_adv")
        self.vars = [Variable(self.x_adv, self.input_bounds[0], self.input_bounds[1])]

        self.d = self.model.addMVar(shape=x.shape, vtype=GRB.CONTINUOUS, lb=0, name="d")

        self.mapper.map(self)

    def buildAdversarialProblem(self, y_target):
        self.model.setObjective(self.d.sum(), GRB.MINIMIZE)
        for key in self.adv_constrs:
            self.model.remove(self.adv_constrs[key])

        self._set_adversarial_constrs(y_target)
        for key in self.adv_constrs:
            constrs = self.adv_constrs[key]
            self.adv_constrs[key] = self.model.addConstrs((constrs[i] for i in range(len(constrs))), name=key)

    def _set_adversarial_constrs(self, y_target):
        x = self.input
        iC, iH, iW = x.shape

        self.adv_constrs["absconstr_lb"] = [- self.d[c, h, w] <= self.x_adv[c, h, w] - x[c, h, w] for c in range(iC) for
                                            h in range(iH) for w in range(iW)]
        self.adv_constrs["absconstr_ub"] = [self.d[c, h, w] >= self.x_adv[c, h, w] - x[c, h, w]
                                            for c in range(iC) for h in range(iH) for w in range(iW)]

        preds = self.vars[-1].x
        self.adv_constrs["pred_constr"] = [preds[y_target] >= 1.1 * preds[i]
                                           for i in range(self.n_classes) if i != y_target]

    def buildInferenceProblem(self):
        self.vars = [Variable(self.input, self.input_bounds[0], self.input_bounds[1])]
        self.mapper.map(self)
        self.model.setObjective(sum([x.x.sum() for x in self.vars[1:]]), GRB.MINIMIZE)


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
                model.add_layer(FlattenMILP(i))
            else:
                raise TypeError("Layer not supported")
            model.model.write("model.lp")
