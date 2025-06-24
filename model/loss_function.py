import numpy as np

def binaryCrossEntropyLoss(y_true, y_pred):
    eps = 1e-8  # avoid log(0)
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))