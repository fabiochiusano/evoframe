from evoframe.reward_builders import RewardBuilder
import copy
from enum import Enum
import numpy as np
from evoframe.os import pickle_load
from evoframe.experiment_results import get_best_model_of_epoch

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

    def get_reward_func(self):
        game_creation_func = self.game_creation_func
        agent_wrapper_func = self.agent_wrapper_func
        competitive_tournament = self.competitive_tournament
        keep_only = self.keep_only
        tournament_mode = self.tournament_mode
        use_weight_normalization = self.use_weight_normalization
        max_weight = self.max_weight

        def reward_func(context, model):
            reward = 0
            if competitive_tournament:
                cur_epoch = context["cur_epoch"]
                pop_size = context["pop_size"]

                if tournament_mode == TournamentMode.VS_CURRENT_POP:
                    #cur_pop = context["epochs"][cur_epoch]["models"]
                    cur_pop = context["population"]["models"][-pop_size:]
                    opponents = cur_pop

                elif tournament_mode == TournamentMode.VS_LAST_POP:
                    if cur_epoch == 1:
                        #cur_pop = context["epochs"][cur_epoch]["models"]
                        cur_pop = context["population"]["models"][-pop_size:]
                        opponents = cur_pop
                    else:
                        #last_pop = context["epochs"][cur_epoch - 1]["models"]
                        last_pop = context["population"]["models"][-pop_size*2:-pop_size]
                        opponents = last_pop

                elif tournament_mode == TournamentMode.VS_BEST_OF_EACH_GEN:
                    if cur_epoch == 1:
                        #cur_pop = context["epochs"][cur_epoch]["models"]
                        cur_pop = context["population"]["models"][-pop_size:]
                        context["tournament"]["models"] = []
                        context["tournament"]["last_updated_epoch"] = 1
                        opponents = [cur_pop[0]]
                    else:
                        if context["tournament"]["last_updated_epoch"] < cur_epoch:
                            #last_pop = context["epochs"][cur_epoch - 1]["models"]
                            last_pop = context["population"]["models"][-pop_size*2:-pop_size]
                            #last_rewards = context["epochs"][cur_epoch - 1]["rewards"]
                            last_rewards = context["population"]["rewards"][-pop_size:]
                            last_best_model = last_pop[np.array(last_rewards).argmax()]
                            context["tournament"]["models"] += [last_best_model]
                            context["tournament"]["models"] = context["tournament"]["models"][-keep_only:]
                            context["tournament"]["last_updated_epoch"] = cur_epoch
                        opponents = context["tournament"]["models"]

                elif tournament_mode == TournamentMode.VS_PEAKS:
                    if cur_epoch == 1:
                        #cur_pop = context["epochs"][cur_epoch]["models"]
                        cur_pop = context["population"]["models"][-pop_size:]
                        opponents = [cur_pop[0]]
                        context["tournament"]["best_models"] = []
                        context["tournament"]["best_models_rewards"] = []
                        context["tournament"]["peak_models"] = []
                        context["tournament"]["last_updated_epoch"] = 1
                    elif cur_epoch <= 3:
                        #cur_pop = context["epochs"][cur_epoch]["models"]
                        cur_pop = context["population"]["models"][-pop_size:]
                        if context["tournament"]["last_updated_epoch"] < cur_epoch:
                            #last_pop = context["epochs"][cur_epoch - 1]["models"]
                            last_pop = context["population"]["models"][-pop_size*2:-pop_size]
                            #last_rewards = context["epochs"][cur_epoch - 1]["rewards"]
                            last_rewards = context["population"]["rewards"][-pop_size:]
                            last_best_model = last_pop[np.array(last_rewards).argmax()]
                            context["tournament"]["best_models"] += [last_best_model]
                            context["tournament"]["best_models_rewards"] += [np.array(last_rewards).max()]
                            context["tournament"]["last_updated_epoch"] = cur_epoch
                        opponents = context["tournament"]["best_models"] + [cur_pop[0]]
                    else:
                        if context["tournament"]["last_updated_epoch"] < cur_epoch:
                            #last_pop = context["epochs"][cur_epoch - 1]["models"]
                            last_pop = context["population"]["models"][-pop_size*2:-pop_size]
                            #last_rewards = context["epochs"][cur_epoch - 1]["rewards"]
                            last_rewards = context["population"]["rewards"][-pop_size:]
                            last_best_model = last_pop[np.array(last_rewards).argmax()]
                            context["tournament"]["best_models"] += [last_best_model]
                            context["tournament"]["best_models_rewards"] += [np.array(last_rewards).max()]
                            # look for peaks
                            r1 = int(context["tournament"]["best_models_rewards"][-3])
                            r2 = int(context["tournament"]["best_models_rewards"][-2])
                            r3 = int(context["tournament"]["best_models_rewards"][-1])
                            if r1 < r2 and r2 >= r3:
                                context["tournament"]["peak_models"] += [context["tournament"]["best_models"][-2]]
                            # apply keep_only
                            context["tournament"]["best_models"] = context["tournament"]["best_models"][-keep_only:]
                            context["tournament"]["best_models_rewards"] = context["tournament"]["best_models_rewards"][-keep_only:]
                            context["tournament"]["peak_models"] = context["tournament"]["peak_models"][-keep_only:]
                            context["tournament"]["last_updated_epoch"] = cur_epoch
                        opponents = (context["tournament"]["peak_models"] + context["tournament"]["best_models"])

                opponents = opponents[:keep_only]
                for opponent in opponents:
                    reward += game_creation_func(context).play(agent_wrapper_func(model), agent_wrapper_func(opponent))[0]
                    reward += game_creation_func(context).play(agent_wrapper_func(opponent), agent_wrapper_func(model))[1]
            else:
                game = game_creation_func(context)
                reward += game.play(agent_wrapper_func(model))

            if use_weight_normalization:
                reward_weights = np.sum([np.sum(np.power(w, 2)) for w in model.weights])
                weights_size = np.sum([w.size for w in model.weights])
                reward_biases = np.sum([np.sum(np.power(b, 2)) for b in model.biases])
                biases_size = np.sum([b.size for b in model.biases])
                reward += (max_weight - (reward_weights + reward_biases) / (weights_size + biases_size)) / max_weight

            #if len(context["epochs"][self.context["cur_epoch"]]["rewards"]) == 0:
                #context["epochs"][self.context["cur_epoch"]]["rewards"] = []
            #context["epochs"][self.context["cur_epoch"]]["rewards"] += [reward]
            if len(context["population"]["rewards"]) == 0:
                context["population"]["rewards"] = []
            context["population"]["rewards"] += [reward]

            return reward

        return reward_func

    def get(self):
        if self.is_ok():
            return self.get_reward_func()
        else:
            print("RewardBuilder is not correctly fed")
            return None
