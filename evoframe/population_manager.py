from pathos.multiprocessing import ProcessPool
import numpy as np

def worker_process(arg):
    # arg is a tuple (reward_func, env), where env is another tuple
    reward_func, p = arg
    return reward_func(p)

class PopulationManager:
    def __init__(self, pop_size, get_model_f, reward_f, get_new_pop_func, context={}, num_threads=1):
        self.pop_size = pop_size
        self.get_model_f = get_model_f
        self.reward_f = reward_f
        self.get_new_pop_func = get_new_pop_func
        self.num_threads = num_threads
        self.context = context

    def generate_pop(self):
        pop = [self.get_model_f() for i in range(self.pop_size)]
        self.context["epochs"][self.context["cur_epoch"]]["models"] = pop
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

    def run(self, num_epochs):
        pool = ProcessPool(self.num_threads) if self.num_threads > 1 else None
        self.context["pop_size"] = self.pop_size
        self.context["num_epochs"] = num_epochs
        self.context["cur_epoch"] = 1
        pop = self.generate_pop()
        for cur_epoch in range(1, num_epochs + 1):
            rewards = self.compute_rewards(pool, pop)
            pop, rewards = self.rank_pop(pop, rewards)
            if cur_epoch < num_epochs:
                self.context["cur_epoch"] = cur_epoch + 1
                pop = self.get_new_pop_func(pop, rewards)
            print("Epoch {}, best reward is {}".format(cur_epoch,rewards[0]))
