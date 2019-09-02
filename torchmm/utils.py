import torch
import logging
import sys
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

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
