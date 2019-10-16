import numpy as np
from evoframe.games import Game

class GuessPoint(Game):
    """This game consists in learning to output specific numbers (m-dimensional) given
        some numbers in input (n-dimensional).

        1 player game.

        Agent interface:
        - input: n-dimensional np.array filled with values
        - output: m-dimensional np.array filled with values
        """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def play(self, agent):
        predictions = agent.predict(self)
        error = np.sum(np.square(self.outputs - predictions))
        reward = -error
        return reward
