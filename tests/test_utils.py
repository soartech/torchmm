import pytest
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from torchmm.utils import pack_list
from torchmm.utils import unpack_list
from torchmm.utils import kmeans_init


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


@pytest.mark.xfail
def test_kmeans_init():
    means = torch.tensor([0.0, 0.0])
    mv = MultivariateNormal(loc=means,
                            precision_matrix=torch.ones_like(means).diag())
    X = mv.sample((100,))
    centroids = kmeans_init(X, 3)

    # from matplotlib import pyplot as plt
    # plt.plot(X[:, 0], X[:, 1], 'b+')
    # plt.plot(centroids[:, 0], centroids[:, 1], 'ro')
    # plt.show()

    assert False
