from abc import ABC, abstractmethod

class RewardBuilder(ABC):
    @abstractmethod
    def get_reward_function(self):
        pass

    @abstractmethod
    def get(self):
        pass
