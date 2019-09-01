from pathos.multiprocessing import ProcessPool
import numpy as np
from evoframe.models.utils import pickle_save
import shutil
import os

def worker_process(arg):
    # arg is a tuple (reward_func, env), where env is another tuple
    reward_func, p = arg
    return reward_func(p)

class PopulationManager:
    def __init__(self, pop_size, get_model_f, reward_f, get_new_pop_func, pickle_models_after_gens=1, context={}, num_threads=1):
        self.pop_size = pop_size
        self.get_model_f = get_model_f
        self.reward_f = reward_f
        self.get_new_pop_func = get_new_pop_func
        self.num_threads = num_threads
        self.context = context
        self.pickle_models_after_gens = pickle_models_after_gens

    def generate_pop(self):
        pop = [self.get_model_f() for i in range(self.pop_size)]
        self.context["epochs"][self.context["cur_epoch"]]["models"] = pop
        self.context["epochs"][self.context["cur_epoch"]]["operators"] = ["first_gen"] * self.pop_size
        return pop

    def rank_pop(self, pop, rewards):
        pop_rewards = list(zip(pop, rewards))
        pop_rewards_sorted = sorted(pop_rewards, key=lambda p: p[1], reverse=True)
        pop_rewards_unzipped = list(zip(*pop_rewards_sorted)) # inverse of 'zip'
        return pop_rewards_unzipped[0], pop_rewards_unzipped[1]

    def compute_rewards(self, pool, population):
        if pool is not None:
            worker_args = ((self.reward_f, p) for p in population)
            rewards = pool.map(worker_process, worker_args)
        else:
            rewards = [self.reward_f(p) for p in population]
        rewards = np.array(rewards)
        return rewards

    def save_context_epoch(self, epoch, experiment_name):
        context = self.context
        for i,model in enumerate(self.context["epochs"][epoch]["models"]):
            pickle_save(model, filename="{}/models/epoch_{}/model_{}.pkl".format(experiment_name, epoch, i))
        self.context["epochs"][epoch].pop("models", None)
        rewards = self.context["epochs"][epoch]["rewards"]
        pickle_save(rewards, filename="{}/rewards/epoch_{}.pkl".format(experiment_name, epoch, i))
        operators = self.context["epochs"][epoch]["operators"]
        pickle_save(operators, filename="{}/operators/epoch_{}.pkl".format(experiment_name, epoch, i))
        self.context["epochs"].pop(epoch, None)

    def pickle_models(self, experiment_name):
        cur_epoch = self.context["cur_epoch"]
        if cur_epoch == self.context["num_epochs"]: # Last epoch, save all remaining models
            for epoch in range(1, self.context["num_epochs"] + 1):
                if "models" in self.context["epochs"][epoch]:
                    self.save_context_epoch(epoch, experiment_name)
            pickle_save(self.context["pop_size"], filename="{}/pop_size.pkl".format(experiment_name))
            pickle_save(self.context["num_epochs"], filename="{}/num_epochs.pkl".format(experiment_name))
        else: # Pickle models that are not needed anymore in memory
            epoch_to_check = cur_epoch - self.pickle_models_after_gens
            if "models" in self.context["epochs"][epoch_to_check]:
                self.save_context_epoch(epoch_to_check, experiment_name)

    def clean_experiment_directory(self, experiment_name):
        if os.path.exists(experiment_name) and os.path.isdir(experiment_name):
            shutil.rmtree(experiment_name)
            print("Cleaned {} directory.".format(experiment_name))

    def run(self, num_epochs, experiment_name):
        self.clean_experiment_directory(experiment_name)
        pool = ProcessPool(self.num_threads) if self.num_threads > 1 else None
        self.context["pop_size"] = self.pop_size
        self.context["num_epochs"] = num_epochs
        self.context["cur_epoch"] = 1
        pop = self.generate_pop()
        for cur_epoch in range(1, num_epochs + 1):
            rewards = self.compute_rewards(pool, pop)
            pop, rewards = self.rank_pop(pop, rewards)
            self.pickle_models(experiment_name)
            if cur_epoch < num_epochs:
                self.context["cur_epoch"] = cur_epoch + 1
                pop = self.get_new_pop_func(pop, rewards)
            print("Epoch {}, best reward is {}".format(cur_epoch,rewards[0]))
