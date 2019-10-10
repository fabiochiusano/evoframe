from evoframe.context import recursively_default_dict
from evoframe.utility import *
import numpy as np

class EvolutionBuilder:
    def __init__(self):
        self.reward_func = None
        self.get_new_pop_func = None
        self.get_model_func = None
        self.get_context_func = None

    def with_reward_func(self, reward_func):
        self.reward_func = reward_func
        return self

    def with_get_new_pop_func(self, get_new_pop_func):
        self.get_new_pop_func = get_new_pop_func
        return self

    def with_get_model_func(self, get_model_func):
        self.get_model_func = get_model_func
        return self

    def with_get_context_func(self, get_context_func):
        self.get_context_func = get_context_func
        return self

    def is_ok(self):
        return self.reward_func and self.get_new_pop_func and self.get_model_func and self.get_context_func

    def get_evolution_func(self):
        reward_func = self.reward_func
        get_new_pop_func = self.get_new_pop_func
        get_model_func = self.get_model_func
        get_context_func = self.get_context_func

        def generate_pop(pop_size):
            pop = [get_model_func() for i in range(pop_size)]
            operators = ["first_gen"] * pop_size
            return pop, operators

        def compute_rewards(pop, cur_epoch, pop_size):
            context = get_context_func(pop, cur_epoch, pop_size)
            rewards = [reward_func(p, context, cur_epoch, pop_size) for p in pop]
            return rewards

        def pickle_results(models, rewards, operators, experiment_name, epoch, pop_size, num_epochs):
            pickle_save_pop_size(experiment_name, pop_size)
            pickle_save_num_epochs(experiment_name, num_epochs)
            pickle_save_models_of_epoch(models, epoch, pop_size, experiment_name)
            pickle_save_rewards(pickle_load_rewards(experiment_name) + rewards, experiment_name)
            pickle_save_operators(pickle_load_operators(experiment_name) + operators, experiment_name)

        def evolution_func(experiment_name, pop_size, num_epochs, cur_epoch=1):
            pop, operators = generate_pop(pop_size)
            while cur_epoch <= num_epochs:
                rewards = compute_rewards(pop, cur_epoch, pop_size)
                pickle_results(pop, rewards, operators, experiment_name, cur_epoch, pop_size, num_epochs)
                if cur_epoch < num_epochs:
                    pop, operators = get_new_pop_func(pop, rewards, pop_size)
                print("Epoch {}, best reward is {}".format(cur_epoch, max(rewards)))
                cur_epoch += 1

        return evolution_func

    def get(self):
        if self.is_ok():
            return self.get_evolution_func()
        else:
            print("EvolutionBuilder is not correctly fed.")
            return None
