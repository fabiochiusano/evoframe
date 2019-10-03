import collections

def recursively_default_dict():
    return collections.defaultdict(recursively_default_dict)

def indexes_of_epoch(epoch, context):
    return context["pop_size"] * (epoch - 1), context["pop_size"] * epoch
