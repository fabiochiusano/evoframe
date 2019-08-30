from evoframe.env import Env
from evoframe.reward_builders import RewardBuilder
import copy
from enum import Enum

class TournamentMode(Enum):
    VS_CURRENT_POP = 1
    VS_BEST_OF_EACH_GEN = 2

class RewardBuilderGame(RewardBuilder):
    def __init__(self):
        self.game_creation_function = None
        self.agent_wrapper_func = None
        self.competitive_tournament = False
        self.keep_only = -1
        self.tournament_mode = None

    def with_game_creation_function(self, game_creation_function):
        self.game_creation_function = game_creation_function
        return self

    def with_agent_wrapper_func(self, agent_wrapper_func):
        self.agent_wrapper_func = agent_wrapper_func
        return self

    def with_competitive_tournament(self, tournament_mode):
        self.competitive_tournament = True
        self.tournament_mode = tournament_mode
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
                prev_bests = env[Env.ENV_KEY_TOURNAMENT]
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
        tournament_mode = self.tournament_mode

        def update_env_f(env, new_pop):
            if competitive_tournament:
                if tournament_mode == TournamentMode.VS_CURRENT_POP:
                    env[Env.ENV_KEY_TOURNAMENT] = new_pop
                elif tournament_mode == TournamentMode.VS_BEST_OF_EACH_GEN:
                    if Env.ENV_KEY_TOURNAMENT not in env:
                        env[Env.ENV_KEY_TOURNAMENT] = []
                    env[Env.ENV_KEY_TOURNAMENT] += [new_pop[0]]
                while len(env[Env.ENV_KEY_TOURNAMENT]) > keep_only:
                    env[Env.ENV_KEY_TOURNAMENT].pop(0)
            return env

        return update_env_f

    def get(self):
        if self.is_ok():
            return self.get_reward_function(), self.get_update_env_f()
        else:
            print("RewardBuilder is not correctly fed")
            return None
