import numpy as np
from evoframe.games import Game
from itertools import product

class GuessFunction(Game):
    """This game consists in learning a function from n dimensions to m dimensions, evaluating points
        on a grid.

        1 player game.

        Agent interface:
        - input: n-dimensional np.array filled with values
        - output: m-dimensional np.array filled with values
        """
    def __init__(self, func, input_dim, input_domains, sample_every):
        self.func = func
        self.input_dim = input_dim # input dimension
        self.input_domains = input_domains # domain for each scalar in the input vector
        self.sample_every = sample_every # sampling bin for each scalar in the input vector

    def play(self, agent):
        error = 0
        ranges = []
        for i in range(self.input_dim):
            input_domain = self.input_domains[i]
            sample_every = self.sample_every[i]
            ranges.append(np.arange(input_domain[0], input_domain[1], sample_every))
        for input_tuple in product(*ranges):
            input_array = np.array(input_tuple)
            predictions = agent.predict(input_array)
            error += np.sum(np.square(self.func(input_array) - predictions))
        reward = -error
        return reward
