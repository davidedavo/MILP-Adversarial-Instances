import sys
from abc import ABC, abstractmethod
import numpy as np
from gurobipy.gurobipy import Model, GRB, max_


class LayerMILP(ABC):
    def __init__(self, index):
        self.output_variables = None
        self.index = index
        self.constrs = []

    def add_output_variable(self, model: Model, shape, lb=float("-inf"), ub=float("inf")):
        if self.output_variables:
            raise Exception("Layer already has an output variable")
        self.output_variables = model.addMVar(shape=shape, vtype=GRB.CONTINUOUS, name=f'x_{self.index}', lb=lb, ub=ub)
        return self.output_variables

    @abstractmethod
    def compute_layer(self, x, model: Model, upper_bounds=None):
        pass


class NonLinearLayerMILP(LayerMILP, ABC):
    def __init__(self, index):
        super().__init__(index)
        self.indicator_variables = []

    def add_binary_variable(self, model: Model):
        var_idx = len(self.indicator_variables)
        a = model.addVar(vtype=GRB.BINARY, name=f"a_{self.index}[{var_idx}]")
        self.indicator_variables.append(a)
        return a


class LinearMILP(LayerMILP):
    def __init__(self, index, weights, bias):
        super().__init__(index)
        self.weights = weights
        self.bias = bias

    def compute_layer(self, x, model: Model, upper_bounds=None):
        out_features, in_features = self.weights.shape

        y = self.add_output_variable(model, shape=(out_features,))

        for i in range(out_features):
            constr = model.addConstr(
                (sum(self.weights[i, j] * x.element_at(j) for j in range(in_features)) + self.bias[i] == y[i]),
                name=f"fc_{self.index}_{i}")
            self.constrs.append(constr)

        return y


class Conv2dMILP(LayerMILP):
    def __init__(self, index, kernel, bias):
        super().__init__(index)
        self.kernel = kernel
        self.bias = bias

    def compute_layer(self, x, model: Model, upper_bounds=None):
        oC, iC, kH, kW = self.kernel.shape

        assert iC == x.shape[0]
        _, iH, iW = x.shape

        oH = iH - kH + 1
        oW = iW - kW + 1

        target_shape = (oC, oH, oW)
        y = self.add_output_variable(model, shape=target_shape)

        self.constrs = [model.addConstr((sum([self.kernel[c, i, k_h, k_w] * x[i, h + k_h, w + k_w]
                                              for i in range(iC) for k_h in range(kH) for k_w in range(kW)]) +
                                         self.bias[c] == y[c, h, w]),
                                        name=f"conv_{self.index}_{c}_{h}_{w}") for c in range(oC) for h in range(oH) for
                        w in range(oW)]
        return y


class MaxPool2dMILP(LayerMILP):
    def __init__(self, index, kernel_size: int, stride: int):
        super().__init__(index)
        self.kernel_size = kernel_size
        self.stride = stride

    def compute_layer(self, x, model: Model, upper_bounds=None):
        C, iH, iW = x.shape

        oH = (iH - self.kernel_size) // self.stride + 1
        oW = (iW - self.kernel_size) // self.stride + 1

        y = self.add_output_variable(model, shape=(C, oH, oW))

        self.constrs = model.addConstrs(
            (y[c, h, w] == max_(
                [x[c, h * self.stride + i, w * self.stride + j] for i in range(self.kernel_size) for j in
                 range(self.kernel_size)])
             for c in range(C) for h in range(oH) for w in range(oW)), name=f"maxpool_{self.index}")
        return y


class ReluMILP(NonLinearLayerMILP):
    def __init__(self, index):
        super().__init__(index)

    def compute_layer(self, x, model: Model, upper_bounds=None):
        y = self.add_output_variable(model, shape=x.shape, lb=0)
        dim = np.prod(x.shape)
        for i in range(dim):
            constr_name = f"relu_{self.index}[{i}]"
            x_el = x.element_at(i)
            y_el = y.element_at(i)
            l = x_el.lb
            l = l if l != float("-inf") else -1000
            u = x_el.ub
            u = u if u != float("+inf") else 1000
            if u <= 0:
                model.addConstr(y_el == 0, name=constr_name)
            elif l >= 0:
                model.addConstr(y_el == x_el, name=constr_name)
            else:
                a = self.add_binary_variable(model)
                model.addConstr(((a == 1) >> (x_el >= 0)), name=f"ind_pos_{self.index}")
                model.addConstr(((a == 0) >> (x_el <= 0)), name=f"ind_pos_{self.index}")
                model.addConstr((y_el <= x_el - l * (1 - a)), name=f"{constr_name}_c1")
                model.addConstr((y_el >= x_el), name=f"{constr_name}_c2")
                model.addConstr((y_el <= u * a), name=f"{constr_name}_c3")
                # model.addConstr((y_el <= a * x_el), name=f"{constr_name}_c1")
                # model.addConstr((y_el >= a * x_el), name=f"{constr_name}_c2")
                # model.addConstr((y_el == a * x_el), name=f"{constr_name}_c2")
        return y
