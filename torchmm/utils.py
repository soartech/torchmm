import torch
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def pack_list(X):
    """
    Takes a list of lists of features.

    Returns packed and padded sequences.
    """
    X = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in X]
    return pack_sequence(X, enforce_sorted=False)


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
