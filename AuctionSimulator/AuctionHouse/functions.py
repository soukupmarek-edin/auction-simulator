import numpy as np


def sigmoid(a, s=1):
    return 1 / (1 + np.exp(-s * a))


def reverse_sigmoid(a, s=1):
    return 1 - 1 / (1 + np.exp(-s * a))


def min_max_transform(arr, scale=1):
    return scale*(arr - arr.min()) / (arr.max() - arr.min())


def standardize(arr):
    return (arr-arr.mean())/arr.std()
