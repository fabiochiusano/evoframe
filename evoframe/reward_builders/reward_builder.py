from abc import ABC, abstractmethod

class RewardBuilder(ABC):
    @abstractmethod
    def get_reward_funcs(self):
        pass

    @abstractmethod
    def get(self):
        pass
