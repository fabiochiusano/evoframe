from evoframe.reward_builders import RewardBuilder
import copy
from enum import Enum
import evoframe.func_with_context as fwc
import numpy as np
from evoframe.os import pickle_load

class TournamentMode(Enum):
    VS_CURRENT_POP = 1
    VS_BEST_OF_EACH_GEN = 2 # from most recent to oldest
    VS_BEST_OF_GEN_EVERY = 3

class RewardBuilderGame(RewardBuilder):
    def __init__(self):
        self.game_creation_function = None
        self.agent_wrapper_func = None
        self.competitive_tournament = False
        self.keep_only = 100000000
        self.tournament_mode = None
        self.context = {}

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

    def with_context(self, context):
        self.context = context
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
        tournament_mode = self.tournament_mode

        def reward_function(context, model):
            reward = 0
            if competitive_tournament:
                if tournament_mode == TournamentMode.VS_CURRENT_POP:
                    opponents = context["epochs"][self.context["cur_epoch"]]["models"]
                elif tournament_mode == TournamentMode.VS_BEST_OF_EACH_GEN:
                    epochs = sorted(list(context["epochs"].keys()), reverse=True) # from last epoch
                    cur_epoch = context["cur_epoch"]
                    first_epoch_to_consider = max(cur_epoch - keep_only, 1)
                    opponents = []
                    if cur_epoch == 1: # Special case of first gen: add at least one model
                        opponents.append(context["epochs"][cur_epoch]["models"][0])
                    for epoch in range(first_epoch_to_consider, cur_epoch):
                        highest_reward_index = np.array(context["epochs"][epoch]["rewards"]).argmax()
                        opponents.append(context["epochs"][epoch]["models"][highest_reward_index])
                #elif tournament_mode == TournamentMode.VS_BEST_OF_GEN_EVERY:
                #    pickle_load("experiments/{}/models/epoch_{}/model_{}.pkl".format(experiment_name, epoch, i_model))
                opponents = opponents[:keep_only]
                for opponent in opponents:
                    reward += game_creation_function().play(agent_wrapper_func(model), agent_wrapper_func(opponent))[0]
                    reward += game_creation_function().play(agent_wrapper_func(opponent), agent_wrapper_func(model))[1]
            else:
                game = game_creation_function()
                reward += game.play(agent_wrapper_func(model))

            if len(context["epochs"][self.context["cur_epoch"]]["rewards"]) == 0:
                context["epochs"][self.context["cur_epoch"]]["rewards"] = []
            context["epochs"][self.context["cur_epoch"]]["rewards"] += [reward]

            return reward

        return fwc.func_with_context(reward_function, context=self.context)

    def get(self):
        if self.is_ok():
            return self.get_reward_function()#, self.get_update_env_f()
        else:
            print("RewardBuilder is not correctly fed")
            return None
