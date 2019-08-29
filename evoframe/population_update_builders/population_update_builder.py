from abc import ABC, abstractmethod

class PopulationUpdateBuilder:
    @abstractmethod
    def get_update_pop_f(self):
        pass

    @abstractmethod
    def get(self):
        pass
