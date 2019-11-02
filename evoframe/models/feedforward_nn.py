from evoframe.models.model import Model
from evoframe.models.utils import mask_tensor
import numpy as np

class FeedForwardNetwork(Model):
    """FeedForward network without biases"""
    def __init__(self, layer_sizes, activations, last_activation, with_bias=True):
        # random initialization
        self.layer_sizes = layer_sizes
        self.weights = np.array(
            [np.random.randn(layer_sizes[index], layer_sizes[index+1])
            for index
            in range(len(layer_sizes)-1)]
        )
        self.with_bias = with_bias
        if self.with_bias:
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
            if self.with_bias:
                out += self.biases[i]
        return out

    def es_1_copy(self):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation, with_bias=self.with_bias)
        new_model.weights = self.weights
        if self.with_bias:
            new_model.biases = self.biases
        return new_model

    def es_1_mutation(self, sigma=0.3, keep_perc=0.3):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation, with_bias=self.with_bias)

        new_weights = []
        for w in self.weights:
            noise = np.random.randn(*w.shape) * sigma
            mask = mask_tensor(w, keep_perc=keep_perc)
            new_weights.append(w + noise * mask)
        new_model.weights = np.array(new_weights)

        if self.with_bias:
            new_biases = []
            for b in self.biases:
                noise = np.random.randn(*b.shape) * sigma
                mask = mask_tensor(b, keep_perc=keep_perc)
                new_biases.append(b + noise * mask)
            new_model.biases = np.array(new_biases)

        return new_model

    def es_2_crossover(self, other, keep_perc=0.8):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation, with_bias=self.with_bias)

        new_weights = []
        for i,w in enumerate(self.weights):
            mask = mask_tensor(w, keep_perc=0.8)
            new_weights.append(w * mask + other.weights[i] * (1 - mask))
        new_model.weights = np.array(new_weights)

        if self.with_bias:
            new_biases = []
            for i,b in enumerate(self.biases):
                mask = mask_tensor(b, keep_perc=0.8)
                new_biases.append(b * mask + other.biases[i] * (1 - mask))
            new_model.biases = np.array(new_biases)

        return new_model

    def es_n_rewards_gradient(self, pop, rewards, learning_rate=0.3, keep_perc=1.):
        new_model = FeedForwardNetwork(self.layer_sizes, self.activations, self.last_activation, with_bias=self.with_bias)

        selected_indexes = np.random.choice(list(range(len(pop))), size=int(len(pop)*keep_perc), replace=True)
        selected_individuals = [pop[i] for i in selected_indexes]
        selected_rewards = np.array([rewards[i] for i in selected_indexes])
        std = selected_rewards.std()
        if std == 0:
            return new_model
        selected_rewards = (selected_rewards - selected_rewards.mean()) / std # Z-score the rewards

        # Update weights
        new_weights = []
        for index, w in enumerate(self.weights): # for each layer
            layer_population = np.array([p.weights[index] for p in selected_individuals]) # get the index-th layer of the whole population
            new_weights.append(w + learning_rate * np.dot(layer_population.T, selected_rewards).T / len(selected_indexes)) # new layer is old layer + weighted sum of z-scored rewards
            #print(learning_rate * np.dot(layer_population.T, selected_rewards).T / len(selected_indexes))
        new_model.weights = np.array(new_weights)

        # Update biases
        if self.with_bias:
            new_biases = []
            for index, b in enumerate(self.biases):
                layer_population = np.array([p.biases[index] for p in selected_individuals])
                new_biases.append(b + learning_rate * np.dot(layer_population.T, selected_rewards).T / len(selected_indexes))
            new_model.biases = np.array(new_biases)

        return new_model

    def es_n_rewards_gradient_and_mutation(self, pop, rewards, learning_rate=0.3, sigma=0.3, keep_perc=0.3):
        new_model = self.es_n_rewards_gradient(pop, rewards, learning_rate)
        new_model = new_model.es_1_mutation(sigma, keep_perc)
        return new_model
