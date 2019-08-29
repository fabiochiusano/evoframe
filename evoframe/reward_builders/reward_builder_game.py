import evoframe.env as ENV
from evoframe.reward_builders import RewardBuilder
import copy

def dict_functional_update(d, key, value):
    new_d = copy.deepcopy(d)
    new_d[key] = value
    return new_d

class RewardBuilderGame(RewardBuilder):
    def __init__(self):
        self.game_creation_function = None
        self.agent_wrapper_func = None
        self.competitive_tournament = False
        self.keep_only = -1

    def with_game_creation_function(self, game_creation_function):
        self.game_creation_function = game_creation_function
        return self

    def with_agent_wrapper_func(self, agent_wrapper_func):
        self.agent_wrapper_func = agent_wrapper_func
        return self

    def with_competitive_tournament(self):
        self.competitive_tournament = True
        return self

    def with_keep_only(self, n):
        self.keep_only = n
        return self

    def is_ok(self):
        game_creation_function_defined = self.game_creation_function != None
        agent_wrapper_func_defined = self.agent_wrapper_func != None
        return game_creation_function_defined

    def get_reward_function(self):
        game_creation_function = self.game_creation_function
        agent_wrapper_func = self.agent_wrapper_func
        competitive_tournament = self.competitive_tournament
        keep_only = self.keep_only

        def reward_function(model, env):
            reward = 0
            if competitive_tournament:
                prev_bests = env[ENV.ENV_KEY_CURRENT_POPULATION]
                if keep_only >= 1:
                    prev_bests = prev_bests[:keep_only]
                for prev_best in prev_bests:
                    game = game_creation_function()
                    reward += game.play(agent_wrapper_func(model), agent_wrapper_func(prev_best))
            else:
                game = game_creation_function()
                reward += game.play(agent_wrapper_func(model))
            return reward

        return reward_function

    def get_update_env_f(self):
        game_creation_function = self.game_creation_function
        agent_wrapper_func = self.agent_wrapper_func
        competitive_tournament = self.competitive_tournament
        keep_only = self.keep_only

        def update_env_f(env, new_pop):
            if competitive_tournament:
                env = dict_functional_update(env, ENV.ENV_KEY_CURRENT_POPULATION, new_pop)
            return env

        return update_env_f

    def get(self):
        if self.is_ok():
            return self.get_reward_function(), self.get_update_env_f()
        else:
            print("RewardBuilder is not correctly fed")
            return None
