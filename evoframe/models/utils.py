import numpy as np

def mask_matrix(matrix, keep_perc):
    shape = matrix.shape
    mask = np.full(shape[0] * shape[1], 0)
    mask[:int(len(matrix) * keep_perc)] = 1
    np.random.shuffle(mask)
    mask = np.reshape(mask, shape)
    return mask

MODEL_OPERATOR_PREFIX = "es_"
