from gurobipy.gurobipy import MVar
import numpy as np


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

MVar.element_at = element_at
