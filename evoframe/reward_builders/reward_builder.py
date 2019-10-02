from abc import ABC, abstractmethod

class RewardBuilder(ABC):
    @abstractmethod
    def get_reward_func(self):
        pass

    @abstractmethod
    def get(self):
        pass
