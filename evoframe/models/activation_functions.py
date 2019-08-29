import numpy as np

class ActivationFunctions:
    def get_arctan():
        return np.arctan

    def get_id():
        return lambda n: n

    def get_sigmoid(x):
        return 1/(1+np.exp(-x))

    def get_relu(x):
        return max(x, 0)
