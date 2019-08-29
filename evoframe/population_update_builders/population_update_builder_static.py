from evoframe.population_update_builders import PopulationUpdateBuilder

class PopulationUpdateBuilderStatic(PopulationUpdateBuilder):
    def __init__(self):
        self.operators = []
        self.percs = []
        self.args_list = []
        self.selector_f = None

    def add_selector_f(self, selector_f):
        self.selector_f = selector_f
        return self

    def add_operator(self, operator, perc, *args):
        self.operators.append(operator)
        self.percs.append(perc)
        self.args_list.append(args)
        return self

    def normalize_percs(self):
        self.percs = [perc / sum(self.percs) for perc in self.percs]

    def is_ok(self):
        at_least_one = len(self.operators) > 0
        has_selector_f = self.selector_f != None
        return at_least_one and has_selector_f

    def get_update_pop_f(self):
        operators = self.operators
        percs = self.percs
        args_list = self.args_list
        selector_f = self.selector_f

        def update_pop_f(pop, rewards):
            pop_size = len(pop)
            new_pop = []
            for op, perc, args, i in zip(self.operators, self.percs, self.args_list, range(len(self.operators))):
                if i == len(self.operators) - 1:
                    num_individuals = pop_size - len(new_pop)
                else:
                    num_individuals = int(pop_size * perc)
                if "_1_" in op:
                    parent = self.selector_f(pop, rewards, 1)
                    new_individuals = [getattr(parent[0], op)(*args) for i in range(num_individuals)]
                elif "_2_" in op:
                    parents = self.selector_f(pop, rewards, 2)
                    new_individuals = [getattr(parents[0], op)(parents[1], *args) for i in range(num_individuals)]
                new_pop += new_individuals
            return new_pop

        return update_pop_f

    def get(self):
        if self.is_ok():
            self.normalize_percs()
            return self.get_update_pop_f()
        else:
            print("PopulationUpdateBuilder is not correctly fed.")
            return None
