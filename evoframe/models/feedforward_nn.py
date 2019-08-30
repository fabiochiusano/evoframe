from evoframe.models.model import Model
from evoframe.models.utils import mask_matrix
import numpy as np
import pickle

class FeedForwardNetwork(Model):
    """FeedForward network without biases"""
    def __init__(self, layer_sizes, activations, last_activation):
        # random initialization
        self.layer_sizes = layer_sizes
        self.weights = np.array(
            [np.random.randn(layer_sizes[index], layer_sizes[index+1])
            for index
            in range(len(layer_sizes)-1)]
        )
        #self.biases = np.array([np.random.randn(ls) for ls in layer_sizes[1:]])
        self.activations = activations
        self.last_activation = last_activation

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        for i, layer in enumerate(self.weights):
            out = np.dot(out, layer)
            if i < len(self.weights) - 1:
                out = self.activations(out)
            else:
                out = self.last_activation(out)
        return out

    def es_1_copy(self):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation)
        new_model.weights = self.weights
        return new_model

    def es_1_mutation(self, sigma=0.3, keep_perc=0.3):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation)
        x = []
        for w in self.weights:
            noise = np.random.randn(*w.shape) * sigma
            mask = mask_matrix(w, keep_perc=keep_perc)
            x.append(w + noise * mask)
        new_model.weights = np.array(x)
        return new_model

    def es_2_crossover(self, other, keep_perc=0.8):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation)
        x = []
        for i,w in enumerate(self.weights):
            mask = mask_matrix(w, keep_perc=0.8)
            x.append(w * mask + other.weights[i] * (1 - mask))
        new_model.weights = np.array(x)
        return new_model

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
