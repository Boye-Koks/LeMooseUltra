import numpy as np

def to_categorical(lbl):
    """Convert the labels to categoricals"""
    new_lbl = np.zeros(shape=(lbl.shape[0], np.max(lbl) + 1))
    new_lbl[np.arange(lbl.shape[0]), np.squeeze(lbl)] = 1

    return new_lbl