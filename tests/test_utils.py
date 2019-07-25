import torch
from torchmm.utils import pack_list
from torchmm.utils import unpack_list


def test_pack_and_unpack_lists():
    seqs = [[0, 0, 0, 1, 1],
            [1, 0, 1],
            [1, 1, 1, 1]]
    seqs = [torch.tensor(s) for s in seqs]
    X = pack_list(seqs)
    Y = unpack_list(X)

    for i in range(len(Y)):
        assert Y[i].shape == seqs[i].shape
        assert torch.allclose(Y[i], seqs[i])
