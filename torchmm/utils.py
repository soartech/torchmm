import torch
import logging
import sys
from functools import wraps
from functools import partial
import timeit

from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.distributions import Categorical

import numpy as np


def get_logger():
    """
    A method to generate a standard logger.
    """
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


def timefun(f):
    """
    A decorator function for calling Timer with autorange on the provided
    function.

    This decorator can be applied to any function to print certain timing
    outputs. Note, it will potentially run the method multiple times on every
    call.
    """
    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            result = timeit.Timer(partial(f, *args, **kwds)).autorange()
        except Exception:
            result = [10, timeit.Timer(partial(f, *args, **kwds)).timeit(10)]
        a = [a for a in args]
        a += ["%s=%s" % (k, kwds[k]) for k in kwds]
        print("Timing %s%s: %0.7f (num runs=%i)" % (f.__name__, tuple(a),
                                                    result[1], result[0]))
        return f(*args, **kwds)

    return wrapper


def pack_list(X, enforce_sorted=False):
    """
    Takes a list of lists of features or a list of numpy arrays of features.

    Returns packed and padded sequences that can be used in conjunction with
    the hmm_packed model.
    """
    X = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in X]
    return pack_sequence(X, enforce_sorted=enforce_sorted)


def unpack_list(X):
    """
    Takes a packed sequence structure that corresponds to the last pack
    call. Untransforms it and then unsorts it, so it corresponds to the
    original input.
    """
    X_unpacked, lengths = pad_packed_sequence(X, batch_first=True)
    # X_unpacked = X_unpacked[X.sorted_indices]
    # lengths = lengths[X.sorted_indices]
    seqs = [X_unpacked[i, :l] for i, l in enumerate(lengths)]
    return seqs


def pick_far_datapoint_by_states(states, X):
    """
    Given a list of states and datapoints, this returns the index of a
    datapoint in X that is on average the least probability of belonging to all
    the other centroids.
    """
    distances = []
    for s in states:
        dis = -1 * s.log_prob(X)
        distances.append(dis)

    dis = torch.stack(distances)
    dis = dis.sum(dim=-1).squeeze()
    if len(dis.shape) > 1:
        min_dis = dis.min(dim=1)[0]
    else:
        min_dis = dis
    return Categorical(probs=min_dis.softmax(0)).sample((1,))


def pick_far_datapoint(centroids, X):
    """
    Given a list of centroids and AVAILABLE datapoints (X), this returns the
    index of a datapoint in X that is on average furthest away from all the
    centroids.
    """
    dis = pairwise_distance(centroids, X)
    if len(dis.shape) > 1:
        min_dis = dis.min(dim=1)[0]
    else:
        min_dis = dis
    return Categorical(probs=min_dis.softmax(0)).sample((1,))


def kmeans_init(X, k):
    """
    Returns centroids for clusters using the kmeans++ initialization algorithm.
    Accepts the data and the number of desired clusters (k).

    See: https://en.wikipedia.org/wiki/K-means%2B%2B
    """
    n = X.shape[0]

    idx = torch.randint(high=n, size=(1,))
    centroids = X[idx]
    X_s = torch.cat([X[0:idx], X[idx+1:]])

    while centroids.shape[0] < k:
        idx = pick_far_datapoint(centroids, X_s)
        centroids = torch.cat((centroids, X_s[idx]))
        X_s = torch.cat([X_s[0:idx], X_s[idx+1:]])

    return centroids


def pairwise_distance(data1, data2=None):
    """
    Compute pairwise distances between data1 and data2, if data2 is None, then
    pairwise distance between data1 and itself is computed.
    """
    if data2 is None:
        data2 = data1
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    dis = (A-B)**2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


def kmeans(X, k, tol=1e-4):
    """
    A simple pytorch implmementation of kmeans.

    Given X, an N x F tensor where N is the number of data points and F is the
    number of features. It groups these data into k clusters.

    The output of this is a tensor of centroids with shape k x F.
    """
    initial_state = kmeans_init(X, k)

    while True:
        dis = pairwise_distance(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()

        for index in range(k):
            selected = torch.nonzero(choice_cluster == index).squeeze()
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(
            torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

        if center_shift ** 2 < tol:
            break

    return initial_state
