import torch
import logging
import sys
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.distributions import Categorical

import numpy as np


def get_logger():
    logger = get_logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)5s() ] %(message)s"
    formatter = logging.Formatter(FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def sample_state_transition_matrix(states):
    states = np.random.rand(states, states)
    return states/states.sum(axis=1, keepdims=True)


def pack_list(X, enforce_sorted=False):
    """
    Takes a list of lists of features.

    Returns packed and padded sequences.
    """
    X = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in X]
    return pack_sequence(X, enforce_sorted=enforce_sorted)


def unpack_list(X):
    """
    Takes a packed sequence structure that corresponds to the last pack
    call.  Untransforms it and then unsorts it, so it corresponds to the
    original input.
    """
    X_unpacked, lengths = pad_packed_sequence(X, batch_first=True)
    # X_unpacked = X_unpacked[X.sorted_indices]
    # lengths = lengths[X.sorted_indices]
    seqs = [X_unpacked[i, :l] for i, l in enumerate(lengths)]
    return seqs


def kmeans_init(X, k):
    """
    Returns centroids for clusters using the kmeans++ algorithm. Accepts the
    data and the number of desired clusters (k).

    See: https://en.wikipedia.org/wiki/K-means%2B%2B
    """
    n = X.shape[0]

    idx = torch.randint(high=n, size=(1,))
    centroids = X[idx]
    X_s = torch.cat([X[0:idx], X[idx+1:]])

    while centroids.shape[0] < k:
        A = centroids.unsqueeze(dim=1)
        B = X_s.unsqueeze(dim=0)
        dis = (A-B).pow(2)
        dis = dis.sum(dim=-1).squeeze()
        if len(dis.shape) > 1:
            min_dis = dis.min(dim=1)[0]
        else:
            min_dis = dis
        idx = Categorical(probs=min_dis.softmax(0)).sample((1,))
        centroids = torch.cat((centroids, X_s[idx]))
        X_s = torch.cat([X_s[0:idx], X_s[idx+1:]])

    return centroids
