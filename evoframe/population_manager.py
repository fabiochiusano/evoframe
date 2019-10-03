from pathos.multiprocessing import ProcessPool
import numpy as np
from evoframe.os import pickle_save, pickle_load, maybe_delete_dir, exist_file
from evoframe.context import recursively_default_dict, indexes_of_epoch

def worker_process(arg):
    # arg is a tuple (reward_func, env), where env is another tuple
    reward_func, p = arg
    return reward_func(p)

class PopulationManager:
    def __init__(self, pop_size, get_model_f, reward_f, get_new_pop_func,
                context=recursively_default_dict(), num_threads=1):
        self.pop_size = pop_size
        self.get_model_f = get_model_f
        self.reward_f = reward_f
        self.get_new_pop_func = get_new_pop_func
        self.num_threads = num_threads
        self.context = context

    def generate_pop(self):
        pop = [self.get_model_f(self.context) for i in range(self.pop_size)]
        self.context["population"]["models"] = pop
        self.context["population"]["operators"] = ["first_gen"] * self.pop_size
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
            rewards = [self.reward_f(self.context, p) for p in population]
        rewards = np.array(rewards)
        return rewards

    def save_context_epoch(self, epoch, experiment_name):
        #for i,model in enumerate(self.context["epochs"][epoch]["models"]):
            #pickle_save(model, filename="experiments/{}/models/epoch_{}/model_{}.pkl".format(experiment_name, epoch, i))
        first_index, last_index = indexes_of_epoch(epoch, self.context)
        for i,model in enumerate(self.context["population"]["models"]):
            pickle_save(model, filename="experiments/{}/models/model_{}.pkl".format(experiment_name, first_index + i))
        #self.context["epochs"][epoch].pop("models", None)
        #rewards = self.context["epochs"][epoch]["rewards"]
        #pickle_save(rewards, filename="experiments/{}/rewards/epoch_{}.pkl".format(experiment_name, epoch))
        filename = "experiments/{}/rewards.pkl".format(experiment_name)
        if exist_file(filename):
            rewards = pickle_load(filename=filename)
        else:
            rewards = []
        rewards += self.context["population"]["rewards"]
        pickle_save(rewards, filename="experiments/{}/rewards.pkl".format(experiment_name))
        #operators = self.context["epochs"][epoch]["operators"]
        #pickle_save(operators, filename="experiments/{}/operators/epoch_{}.pkl".format(experiment_name, epoch))
        filename = "experiments/{}/operators.pkl".format(experiment_name)
        if exist_file(filename):
            operators = pickle_load(filename=filename)
        else:
            operators = []
        operators += self.context["population"]["operators"]
        pickle_save(operators, filename="experiments/{}/operators.pkl".format(experiment_name))
        #self.context["epochs"].pop(epoch, None)
        self.context["population"]["models"] = self.context["population"]["models"][last_index-first_index:]
        self.context["population"]["rewards"] = self.context["population"]["rewards"][last_index-first_index:]
        self.context["population"]["operators"] = self.context["population"]["operators"][last_index-first_index:]

    def pickle_models(self, experiment_name):
        cur_epoch = self.context["cur_epoch"]
        if cur_epoch == 1:
            return
        else:
            self.save_context_epoch(cur_epoch - 1, experiment_name)

    def clean_experiment_directory(self, experiment_name):
        maybe_delete_dir("experiments/{}".format(experiment_name))

    def initialize_context(self, num_epochs, experiment_name):
        self.context["pop_size"] = self.pop_size
        self.context["num_epochs"] = num_epochs
        self.context["cur_epoch"] = 1
        self.context["experiment_name"] = experiment_name
        pickle_save(self.context["pop_size"], filename="experiments/{}/pop_size.pkl".format(experiment_name))
        pickle_save(self.context["num_epochs"], filename="experiments/{}/num_epochs.pkl".format(experiment_name))

    def run(self, num_epochs, experiment_name):
        self.clean_experiment_directory(experiment_name)
        pool = ProcessPool(self.num_threads) if self.num_threads > 1 else None
        self.initialize_context(num_epochs, experiment_name)
        pop = self.generate_pop()
        for cur_epoch in range(1, num_epochs + 1):
            rewards = self.compute_rewards(pool, pop)
            pop, rewards = self.rank_pop(pop, rewards)
            self.pickle_models(experiment_name) # pickle second last gen (can't pickle last gen because it's needed for new gen creation)
            self.context["cur_epoch"] = cur_epoch + 1
            if cur_epoch < num_epochs:
                pop = self.get_new_pop_func(self.context, pop, rewards)
            else:
                self.pickle_models(experiment_name)
            print("Epoch {}, best reward is {}".format(cur_epoch,rewards[0]))
            print("Number of peaks is {}".format(len(self.context["tournament"]["peak_models"])))
