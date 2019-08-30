import numpy as np

class ActivationFunctions:
    def get_arctan():
        return np.arctan

    def get_id():
        return lambda n: n

    def get_sigmoid():
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        return sigmoid

    def get_relu():
        def relu(x):
            return np.maximum(x, 0)
        return relu
