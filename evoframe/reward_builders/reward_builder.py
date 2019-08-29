from abc import ABC, abstractmethod

class RewardBuilder(ABC):
    @abstractmethod
    def get_reward_function(self):
        pass

    @abstractmethod
    def get_update_env_f(self):
        pass

    @abstractmethod
    def get(self):
        pass
