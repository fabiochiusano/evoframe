from evoframe.context import recursively_default_dict, indexes_of_epoch
from evoframe.os import pickle_save, pickle_load, exist_file
import numpy as np

class EvolutionBuilder:
    def __init__(self):
        self.reward_func = None
        self.get_new_pop_func = None
        self.get_model_func = None

    def with_reward_func(self, reward_func):
        self.reward_func = reward_func
        return self

    def with_get_new_pop_func(self, get_new_pop_func):
        self.get_new_pop_func = get_new_pop_func
        return self

    def with_get_model_func(self, get_model_func):
        self.get_model_func = get_model_func
        return self

    def is_ok(self):
        return self.reward_func and self.get_new_pop_func and self.get_model_func

    def get_evolution_func(self):
        reward_func = self.reward_func
        get_new_pop_func = self.get_new_pop_func
        get_model_func = self.get_model_func

        def generate_pop(context):
            pop = [get_model_func(context) for i in range(context["pop_size"])]
            context["population"]["models"] = pop
            context["population"]["operators"] = ["first_gen"] * context["pop_size"]
            return pop

        def compute_rewards(context, population):
            rewards = np.array([reward_func(context, p) for p in population])
            return rewards

        def rank_pop(pop, rewards):
            pop_rewards = list(zip(pop, rewards))
            pop_rewards_sorted = sorted(pop_rewards, key=lambda p: p[1], reverse=True)
            pop_rewards_unzipped = list(zip(*pop_rewards_sorted)) # inverse of 'zip'
            return pop_rewards_unzipped[0], pop_rewards_unzipped[1]

        def pickle_models(context, epoch):
            experiment_name = context["experiment_name"]
            first_index, last_index = indexes_of_epoch(epoch, context)
            for i,model in enumerate(context["population"]["models"][:context["pop_size"]]):
                pickle_save(model, filename="experiments/{}/models/model_{}.pkl".format(experiment_name, first_index + i))

            filename = "experiments/{}/rewards.pkl".format(experiment_name)
            if exist_file(filename):
                rewards = pickle_load(filename=filename)
            else:
                rewards = []
            rewards += context["population"]["rewards"][:context["pop_size"]]
            pickle_save(rewards, filename=filename)

            filename = "experiments/{}/operators.pkl".format(experiment_name)
            if exist_file(filename):
                operators = pickle_load(filename=filename)
            else:
                operators = []
            operators += context["population"]["operators"][:context["pop_size"]]
            pickle_save(operators, filename=filename)

            context["population"]["models"] = context["population"]["models"][context["pop_size"]:]
            context["population"]["rewards"] = context["population"]["rewards"][context["pop_size"]:]
            context["population"]["operators"] = context["population"]["operators"][context["pop_size"]:]
            #print(len(context["population"]["models"]), len(context["population"]["rewards"]), len(context["population"]["operators"]))

        def evolution_func(context):
            pop = generate_pop(context)
            while context["cur_epoch"] <= context["num_epochs"]:
                rewards = compute_rewards(context, pop)
                pop, rewards = rank_pop(pop, rewards)
                if context["cur_epoch"] > 1:
                    pickle_models(context, context["cur_epoch"] - 1) # pickle second last gen (can't pickle last gen because the rewards are needed in tournament mode)
                if context["cur_epoch"] < context["num_epochs"]:
                    pop = get_new_pop_func(context, pop, rewards)
                else:
                    pickle_models(context, context["cur_epoch"])
                print("Epoch {}, best reward is {}".format(context["cur_epoch"], rewards[0]))
                context["cur_epoch"] += 1

        return evolution_func

    def get_context_init_func(self):
        def context_init_func(experiment_name, pop_size, num_epochs, cur_epoch=1):
            context = recursively_default_dict()
            context["pop_size"] = pop_size
            context["num_epochs"] = num_epochs
            context["cur_epoch"] = cur_epoch
            context["experiment_name"] = experiment_name
            pickle_save(context["pop_size"], filename="experiments/{}/pop_size.pkl".format(experiment_name))
            pickle_save(context["num_epochs"], filename="experiments/{}/num_epochs.pkl".format(experiment_name))
            return context
        return context_init_func

    def get(self):
        if self.is_ok():
            return self.get_evolution_func()
        else:
            print("EvolutionBuilder is not correctly fed.")
            return None
