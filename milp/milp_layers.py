import sys
from abc import ABC, abstractmethod
import numpy as np
import torch
from gurobipy.gurobipy import Model, GRB, max_, MVar
import torch.nn.functional as F
from torch import nn, from_numpy

from milp import element_at, dict_to_list
from milp.milp_variable import Variable

BIGM = 1000


class LayerMILP(ABC):
    def __init__(self, index, layer_name, milp_presolve):
        self.index = index
        self.var = None
        self.binary_vars = None
        self.layer_name = layer_name
        self._constrs = []
        self.relu = False
        self.milp_presolve = milp_presolve

    def set_activation_var(self, model: Model, input_var: Variable) -> MVar:
        if self.var is not None:
            raise Exception("Activation variable already defined for this layer.")
        shape = self._compute_output_shape(input_var.shape)
        if not self.milp_presolve:
            lb, ub = self._compute_ia_bounds(input_var.lb, input_var.ub)
        else:
            lb = 0 if self.relu else float("-inf")
            ub = None
        activation_name = f"x_{self.index}"
        x = model.addMVar(shape=shape, vtype=GRB.CONTINUOUS, name=activation_name, lb=lb, ub=ub)
        self.var = Variable(x, lb, ub)
        if self.milp_presolve:
            self._compute_milp_bounds(model, input_var, self.var)
        return self.var.x

    def add_binary_var(self, model: Model, shape):
        self.binary_vars = model.addMVar(shape=shape, name=f"a_{self.index}", vtype=GRB.BINARY)
        return self.binary_vars

    def compute_layer(self, model: Model, input: Variable) -> Variable:
        y = self.set_activation_var(model, input)
        x = input.x
        if self.relu:
            a = self.add_binary_var(model, y.shape)
        else:
            a = None

        self.constrs = self._get_contraints(x, y)

        model.addConstrs((self.constrs[i] for i in range(len(self.constrs))), name=f"{self.layer_name}_{self.index}")
        return self.var

    @abstractmethod
    def _compute_output_shape(self, input_shape: tuple):
        pass

    @abstractmethod
    def _compute_ia_bounds(self, input_lbs, input_ubs) -> tuple:
        pass

    @abstractmethod
    def _get_contraints(self, x, y) -> dict:
        pass

    def _compute_milp_bounds(self, model: Model, input_var: Variable, output_var: Variable):
        model.update()
        constrs = self._get_contraints(input_var.x, output_var.x)
        for ind in np.ndindex(output_var.shape):
            y = output_var.x[ind]
            constr = constrs[y]
            if hasattr(constr, '__iter__'):
                c = model.addConstrs((constr[i] for i in range(len(constr))), name="final")
            else:
                c = model.addConstr(constrs[y], name="final")
            model.setObjective(y, GRB.MAXIMIZE)
            model.optimize()
            output_var.ub[ind] = min([model.objVal, output_var.ub[ind]])
            y.ub = output_var.ub[ind]
            model.remove(c)
        model.update()

    @property
    def constrs(self):
        return self._constrs

    @constrs.setter
    def constrs(self, value:dict):
        self._constrs = dict_to_list(value)


class LinearMILP(LayerMILP):
    def __init__(self, index, weights, bias, relu):
        super().__init__(index, "linear")
        self.weights = weights
        self.bias = bias
        self.relu = relu

    def _compute_ia_bounds(self, lb, ub):
        _lb = (self.weights.clip(min=0) @ lb + self.weights.clip(max=0) @ ub
               + self.bias[:, None])
        _ub = (self.weights.clip(min=0) @ ub + self.weights.clip(max=0) @ lb
               + self.bias[:, None])
        return _lb, _ub

    def _compute_output_shape(self, input_shape: tuple):
        out_features = self.weights.shape[0]
        return out_features,

    def _get_contraints(self, x, y) -> dict:
        out_features, in_features = self.weights.shape
        constrs = {}
        for i in range(out_features):
            def f(j): return x.element_at(j) if hasattr(x, "element_at") else x[j]

            constr = sum(self.weights[i, j] * f(j) for j in range(in_features)) + self.bias[i] == y[i]
            constrs[y[i]] = constr
        return constrs


class Conv2dMILP(LayerMILP):
    def __init__(self, index, kernel, bias, relu):
        super().__init__(index, "conv2d")
        self.weights = kernel
        self.bias = bias
        self.relu = relu

    def _compute_ia_bounds(self, lb, ub):
        lbt, ubt = from_numpy(lb).float(), from_numpy(ub).float()
        w, b = from_numpy(self.weights), from_numpy(self.bias)
        lbt, ubt = lbt.unsqueeze(dim=0), ubt.unsqueeze(dim=0)

        _lb = (F.conv2d(lbt, w.clamp(min=0), bias=None) + F.conv2d(ubt, w.clamp(max=0), bias=b)).squeeze(dim=0)

        _ub = (F.conv2d(ubt, w.clamp(min=0), bias=None) + F.conv2d(lbt, w.clamp(max=0), bias=b)).squeeze(dim=0)
        return _lb.numpy(), _ub.numpy()

    def _compute_output_shape(self, input_shape: tuple):
        _, iH, iW = input_shape
        oC, iC, kH, kW = self.weights.shape
        assert iC == input_shape[0]

        oH = iH - kH + 1
        oW = iW - kW + 1
        return oC, oH, oW

    def _get_contraints(self, x, y) -> dict:
        _, iC, kH, kW = self.weights.shape
        oC, oH, oW = y.shape
        return {y[c, h, w]: sum([self.weights[c, i, k_h, k_w] * x[i, h + k_h, w + k_w]
                                 for i in range(iC) for k_h in range(kH) for k_w in range(kW)]) +
                            self.bias[c] == y[c, h, w] for c in range(oC) for h in range(oH) for w in range(oW)}


class MaxPool2dMILP(LayerMILP):
    def __init__(self, index, kernel_size: int, stride: int):
        super().__init__(index, "maxpool2d")
        self.kernel_size = kernel_size
        self.stride = stride

    def _compute_ia_bounds(self, lb, ub):
        lbt, ubt = from_numpy(lb).float(), from_numpy(ub).float()
        _lb = F.max_pool2d(lbt, kernel_size=self.kernel_size, stride=self.stride)
        _ub = F.max_pool2d(ubt, kernel_size=self.kernel_size, stride=self.stride)
        return _lb.numpy(), _ub.numpy()

    def _compute_output_shape(self, input_shape: tuple):
        C, iH, iW = input_shape
        oH = (iH - self.kernel_size) // self.stride + 1
        oW = (iW - self.kernel_size) // self.stride + 1
        return C, oH, oW

    def _get_contraints(self, x, y) -> dict:
        C, oH, oW = y.shape
        return {y[c, h, w]: y[c, h, w] == max_([x[c, h * self.stride + i, w * self.stride + j]
                                                for i in range(self.kernel_size) for j in range(self.kernel_size)])
                for c in range(C) for h in range(oH) for w in range(oW)}


class ReluMILP(LayerMILP):

    def __init__(self, index):
        super().__init__(index, "relu")

    def _compute_ia_bounds(self, lb: np.ndarray, ub: np.ndarray):
        # TODO MIGLIORABILE
        _lb = lb.clip(min=0)
        _ub = ub.clip(min=0)
        return _lb, _ub

    def _compute_output_shape(self, input_shape: tuple):
        return input_shape

    def _get_contraints(self, x, y) -> dict:
        dim = np.prod(x.shape)
        constrs = {}
        for i in range(dim):  # TODO IMPROVE HERE
            x_el, y_el, a_el = x.element_at(i), y.element_at(i), a.element_at(i)
            l = x_el.lb if x_el.lb != float("-inf") else -BIGM
            u = x_el.ub if x_el.ub != float("+inf") else BIGM
            constrs[y_el] = []
            if u <= 0:
                constrs[y_el] += [y_el == 0]
            elif l >= 0:
                constrs[y_el] += [y_el == x_el]
            else:
                constrs[y_el] += [(a_el == 1) >> (x_el >= 0)]
                constrs[y_el] += [(a_el == 0) >> (x_el <= 0)]
                constrs[y_el] += [y_el <= x_el - l * (1 - a_el)]
                constrs[y_el] += [y_el >= x_el]
                constrs[y_el] += [y_el <= u * a_el]
        return constrs

    def _compute_milp_bounds(self, model: Model, input_var: Variable, output_var: Variable):
        pass


class FlattenMILP(LayerMILP):
    def __init__(self, index):
        super().__init__(index, "flatten")

    def compute_layer(self, model: Model, input: Variable) -> Variable:
        x = input.x
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            y = x.reshape(-1, 1)
        elif isinstance(x, MVar):
            y = x
        else:
            raise AttributeError("Not supported type for input.x")
        lb, ub = self._compute_ia_bounds(input.lb, input.ub)
        self.var = Variable(y, lb, ub)
        return self.var

    def _compute_ia_bounds(self, lb, ub):
        return lb.reshape(-1, 1), ub.reshape(-1, 1)

    def _compute_output_shape(self, input_shape: tuple):
        pass

    def _get_contraints(self, x, y, a=None) -> dict:
        pass
