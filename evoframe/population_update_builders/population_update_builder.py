from abc import ABC, abstractmethod

class PopulationUpdateBuilder:
    @abstractmethod
    def get_update_pop_func(self):
        pass

    @abstractmethod
    def get(self):
        pass
