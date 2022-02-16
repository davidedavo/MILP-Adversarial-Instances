from gurobipy.gurobipy import Model
import numpy as np

if __name__ == "__main__":
    m = Model("asdsad")
    lb = np.random.randn(2, 3)
    print(f"lb matrix = {lb}")
    ub = np.random.randn(2, 3)
    print(f"ub matrix = {ub}")
    x = m.addMVar(shape=(2, 3), lb=lb, ub=ub)
    m.update()
    for i in range(2):
        for j in range(3):
            print(f"Var[{i},{j}]: lb = {x[i, j].lb}; ub = {x[i, j].ub}")
    k = 4
