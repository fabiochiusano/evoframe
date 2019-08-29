import numpy as np

class SelectorFunctionFactory:
    def get_geometric_selector_function(p=0.9):
        p = p

        def geometric_selector_function(pop, rewards, how_many):
            selected = []
            while len(selected) < how_many:
                for i in range(len(pop)):
                    if np.random.rand(1)[0] < p:
                        selected.append(pop[i])
                        break
            return selected

        return geometric_selector_function
