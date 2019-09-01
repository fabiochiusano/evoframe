from evoframe.models.model import Model
from evoframe.models.utils import mask_tensor
import numpy as np

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
        self.biases = np.array([np.random.randn(ls) for ls in layer_sizes[1:]])
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
            out += self.biases[i]
        return out

    def es_1_copy(self):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation)
        new_model.weights = self.weights
        new_model.biases = self.biases
        return new_model

    def es_1_mutation(self, sigma=0.3, keep_perc=0.3):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation)

        new_weights = []
        for w in self.weights:
            noise = np.random.randn(*w.shape) * sigma
            mask = mask_tensor(w, keep_perc=keep_perc)
            new_weights.append(w + noise * mask)
        new_model.weights = np.array(new_weights)

        new_biases = []
        for b in self.biases:
            noise = np.random.randn(*b.shape) * sigma
            mask = mask_tensor(b, keep_perc=keep_perc)
            new_biases.append(b + noise * mask)
        new_model.biases = np.array(new_biases)

        return new_model

    def es_2_crossover(self, other, keep_perc=0.8):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation)

        new_weights = []
        for i,w in enumerate(self.weights):
            mask = mask_tensor(w, keep_perc=0.8)
            new_weights.append(w * mask + other.weights[i] * (1 - mask))
        new_model.weights = np.array(new_weights)

        new_biases = []
        for i,b in enumerate(self.biases):
            mask = mask_tensor(b, keep_perc=0.8)
            new_biases.append(b * mask + other.biases[i] * (1 - mask))
        new_model.biases = np.array(new_biases)

        return new_model

    def es_n_rewards_gradient(self, pop, rewards, learning_rate=0.03, sigma=0.1):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation)

        pop_size = len(pop)
        rewards = np.array(rewards)
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std # Z-score the rewards
        update_factor = learning_rate / (pop_size * sigma)

        # Update weights
        new_weights = []
        for index, w in enumerate(self.weights):
            layer_population = np.array([p.weights[index] for p in pop])
            new_weights.append(w + update_factor * np.dot(layer_population.T, rewards).T)
        new_model.weights = np.array(new_weights)

        # Update biases
        new_biases = []
        for index, b in enumerate(self.biases):
            layer_population = np.array([p.biases[index] for p in pop])
            new_biases.append(b + update_factor * np.dot(layer_population.T, rewards).T)
        new_model.biases = np.array(new_biases)

        return new_model
