from abc import ABC, abstractmethod

from gurobipy.gurobipy import Model


class MILPLayer(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
    def compute_layer(self):
        pass


class MILPModel:
    def __init__(self, name):
        self.model = Model(name)
        self.layers = []

    def add_layer(self, layer: MILPLayer):
        """
        Add a layer to the model
        Args:
            layer:

        Returns:
        """
        self.layers.append(layer)

    def optimize(self):
        pass

    def precompute(self):
        pass

    def check_ub_exists(self):
        pass



