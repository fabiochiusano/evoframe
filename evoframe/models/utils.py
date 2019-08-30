import numpy as np

def mask_tensor(tensor, keep_perc):
    shape = tensor.shape
    mask = np.full(np.prod(np.array(shape)), 0)
    mask[:int(len(mask) * keep_perc)] = 1
    np.random.shuffle(mask)
    mask = np.reshape(mask, shape)
    return mask

MODEL_OPERATOR_PREFIX = "es_"
