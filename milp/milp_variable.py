from typing import Union

import numpy as np
from gurobipy.gurobipy import MVar, Model
import torch


class Variable:
    def __init__(self, x, lb, ub):
        self._lb = self._process_bound(lb)
        self._ub = self._process_bound(ub)
        self._x = x

    def _process_bound(self, bound):
        if isinstance(bound, int) or isinstance(bound, float):
            return bound * np.ones(self.x.shape)
        elif hasattr(bound, "shape"):
            return bound
        else:
            raise AttributeError("Bound type not supported")

    @property
    def x(self) -> Union[MVar, np.ndarray, torch.Tensor]:
        return self._x

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def shape(self):
        return self.x.shape