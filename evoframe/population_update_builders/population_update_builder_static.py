from evoframe.population_update_builders import PopulationUpdateBuilder

class PopulationUpdateBuilderStatic(PopulationUpdateBuilder):
    def __init__(self):
        self.operators = []
        self.percs = []
        self.args_list = []
        self.selector_func = None

    def add_selector_func(self, selector_func):
        self.selector_func = selector_func
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
        has_selector_func = self.selector_func != None
        return at_least_one and has_selector_func

    def get_update_pop_func(self):
        operators = self.operators
        percs = self.percs
        args_list = self.args_list
        selector_func = self.selector_func

        def rank_pop(pop, rewards):
            pop_rewards = list(zip(pop, rewards))
            pop_rewards_sorted = sorted(pop_rewards, key=lambda p: p[1], reverse=True)
            pop_rewards_unzipped = list(zip(*pop_rewards_sorted)) # inverse of 'zip'
            return pop_rewards_unzipped[0], pop_rewards_unzipped[1]

        def update_pop_func(pop, rewards, pop_size):
            # Sort pop by rewards
            pop, rewards = rank_pop(pop, rewards)

            new_pop = []
            new_operators = []
            for op, perc, args, i in zip(operators, percs, args_list, range(len(operators))):
                if i == len(operators) - 1:
                    num_individuals = pop_size - len(new_pop)
                else:
                    num_individuals = int(pop_size * perc)
                new_individuals = []
                if "_1_" in op:
                    for i in range(num_individuals):
                        parents = selector_func(pop, rewards, 1)
                        new_individuals.append(getattr(parents[0], op)(*args))
                elif "_2_" in op:
                    for i in range(num_individuals):
                        parents = selector_func(pop, rewards, 2)
                        new_individuals.append(getattr(parents[0], op)(parents[1], *args))
                elif "_n_rewards_" in op:
                    for i in range(num_individuals):
                        parents = selector_func(pop, rewards, 1)
                        new_individuals.append(getattr(parents[0], op)(pop, rewards, *args))

                new_pop += new_individuals
                op_name = op + "".join(["_" + str(arg) for arg in args])
                new_operators += [op_name for i in range(num_individuals)]

            return new_pop, new_operators

        return update_pop_func

    def get(self):
        if self.is_ok():
            self.normalize_percs()
            return self.get_update_pop_func()
        else:
            print("PopulationUpdateBuilder is not correctly fed.")
            return None
