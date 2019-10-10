from evoframe.reward_builders import RewardBuilder
import copy
from enum import Enum
import numpy as np
from evoframe.utility import *
from evoframe.context import recursively_default_dict

class TournamentMode(Enum):
    VS_CURRENT_POP = 1
    VS_LAST_POP = 2
    VS_BEST_OF_EACH_GEN = 3 # from most recent to oldest
    VS_PEAKS = 4

class RewardBuilderGame(RewardBuilder):
    def __init__(self):
        self.game_creation_func = None
        self.agent_wrapper_func = None
        self.competitive_tournament = False
        self.keep_only = 100000000
        self.select_every = 1
        self.tournament_mode = None
        self.use_weight_normalization = False
        self.max_weight = 0

    def with_game_creation_func(self, game_creation_func):
        self.game_creation_func = game_creation_func
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

    def with_select_every(self, n):
        self.select_every = n
        return self

    def with_weight_normalization(self, max_weight):
        self.use_weight_normalization = True
        self.max_weight = max_weight
        return self

    def is_ok(self):
        game_creation_func_defined = self.game_creation_func != None
        agent_wrapper_func_defined = self.agent_wrapper_func != None
        return game_creation_func_defined

    def get_reward_funcs(self):
        game_creation_func = self.game_creation_func
        agent_wrapper_func = self.agent_wrapper_func
        competitive_tournament = self.competitive_tournament
        keep_only = self.keep_only
        tournament_mode = self.tournament_mode
        use_weight_normalization = self.use_weight_normalization
        max_weight = self.max_weight

        def get_context_func(pop, cur_epoch, pop_size):
            context = recursively_default_dict()
            if competitive_tournament:
                if tournament_mode == TournamentMode.VS_CURRENT_POP:
                    context["current_pop"] = pop
                elif tournament_mode == TournamentMode.VS_LAST_POP:
                    if cur_epoch == 1:
                        context["last_pop"] = pop
                    else:
                        context["last_pop"] = pickle_load_models_of_epoch(experiment_name, cur_epoch-1, pop_size)
                elif tournament_mode == TournamentMode.VS_BEST_OF_EACH_GEN:
                    if cur_epoch == 1:
                        context["last_bests"] = [pop[0]]
                    else:
                        best_models = []
                        for epoch in range(cur_epoch - keep_only, cur_epoch):
                            best_models.append(pickle_load_best_model_of_epoch(experiment_name, epoch, pop_size))
                        context["last_bests"] = best_models
                elif tournament_mode == TournamentMode.VS_PEAKS:
                    if cur_epoch <= 3:
                        context["last_peaks"] = pop[:cur_epoch]
                    else:
                        best_models = []
                        best_rewards = []
                        for epoch in range(cur_epoch - keep_only, cur_epoch):
                            best_models.append(pickle_load_best_model_of_epoch(experiment_name, epoch, pop_size))
                            best_rewards.append(pickle_load_best_reward_of_epoch(experiment_name, epoch, pop_size))
                        peak_models = []
                        for i in range(2, len(best_models)):
                            r1 = int(best_rewards[i-2])
                            r2 = int(best_rewards[i-1])
                            r3 = int(best_rewards[i])
                            if r1 < r2 and r2 >= r3:
                                peak_models.append(best_models[i])
                        context["last_peaks"] = (peak_models + best_models)[:keep_only]
            return context

        def reward_func(model, context, cur_epoch, pop_size):
            reward = 0
            if competitive_tournament:
                if tournament_mode == TournamentMode.VS_CURRENT_POP:
                    opponents = context["current_pop"]
                elif tournament_mode == TournamentMode.VS_LAST_POP:
                    opponents = context["last_pop"]
                elif tournament_mode == TournamentMode.VS_BEST_OF_EACH_GEN:
                    opponents = context["last_bests"]
                elif tournament_mode == TournamentMode.VS_PEAKS:
                    opponents = context["last_peaks"]
                opponents = opponents[:keep_only]
                for opponent in opponents:
                    reward += game_creation_func(context).play(agent_wrapper_func(model), agent_wrapper_func(opponent))[0]
                    reward += game_creation_func(context).play(agent_wrapper_func(opponent), agent_wrapper_func(model))[1]
            else:
                game = game_creation_func(context)
                reward += game.play(agent_wrapper_func(model))

            #if use_weight_normalization:
            #    reward_weights = np.sum([np.sum(np.power(w, 2)) for w in model.weights])
            #    weights_size = np.sum([w.size for w in model.weights])
            #    reward_biases = np.sum([np.sum(np.power(b, 2)) for b in model.biases])
            #    biases_size = np.sum([b.size for b in model.biases])
            #    reward += (max_weight - (reward_weights + reward_biases) / (weights_size + biases_size)) / max_weight

            return reward

        return reward_func, get_context_func

    def get(self):
        if self.is_ok():
            return self.get_reward_funcs()
        else:
            print("RewardBuilder is not correctly fed")
            return None
