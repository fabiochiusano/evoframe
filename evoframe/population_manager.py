from pathos.multiprocessing import ProcessPool
import numpy as np

def worker_process(arg):
    # arg is a tuple (reward_func, env), where env is another tuple
    reward_func, env = arg
    return reward_func(*env)

class PopulationManager:
    def __init__(self, pop_size, get_model_f, reward_f, update_env_f, get_new_pop_func, num_threads=1):
        self.pop_size = pop_size
        self.get_model_f = get_model_f
        self.reward_f = reward_f
        self.update_env_f = update_env_f
        self.get_new_pop_func = get_new_pop_func
        self.num_threads = num_threads

    def generate_pop(self):
        return [self.get_model_f() for i in range(self.pop_size)]

    def rank_pop(self, pop, rewards):
        pop_rewards = list(zip(pop, rewards))
        pop_rewards_sorted = sorted(pop_rewards, key=lambda p: p[1], reverse=True)
        pop_rewards_unzipped = list(zip(*pop_rewards_sorted)) # inverse of 'zip'
        return pop_rewards_unzipped[0], pop_rewards_unzipped[1]

    def compute_rewards(self, pool, population, env):
        if pool is not None:
            worker_args = ((self.reward_f, [p, env]) for p in population)
            rewards = pool.map(worker_process, worker_args)
        else:
            rewards = [self.reward_f(*[p, env]) for p in population]
        rewards = np.array(rewards)
        return rewards

    def run(self, num_epochs):
        pool = ProcessPool(self.num_threads) if self.num_threads > 1 else None
        pop = self.generate_pop()
        env = {"rewards": {}}
        env = self.update_env_f(env, pop)
        for num_epoch in range(num_epochs):
            rewards = self.compute_rewards(pool, pop, env)
            pop, rewards = self.rank_pop(pop, rewards)
            env["rewards"][num_epoch] = rewards
            if num_epoch < num_epochs - 1:
                pop = self.get_new_pop_func(pop, rewards)
                env = self.update_env_f(env, pop)
            print("Epoch {}, best reward is {}".format(num_epoch,rewards[0]))
        return pop, env
