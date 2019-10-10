import collections

def recursively_default_dict():
    return collections.defaultdict(recursively_default_dict)
