import numpy as np
import dill as pickle
import os

MODEL_OPERATOR_PREFIX = "es_"

def mask_tensor(tensor, keep_perc):
    shape = tensor.shape
    mask = np.full(np.prod(np.array(shape)), 0)
    mask[:int(len(mask) * keep_perc)] = 1
    np.random.shuffle(mask)
    mask = np.reshape(mask, shape)
    return mask

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def pickle_save(model, filename='model.pkl'):
    directories = "/".join(filename.split("/")[:-1])
    maybe_make_dir(directories)
    with open(filename, 'wb') as fp:
        pickle.dump(model, fp)

def pickle_load(filename='model.pkl'):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)
