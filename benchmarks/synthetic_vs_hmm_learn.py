import numpy as np
import torch
from functools import partial

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from hmmlearn.hmm import GaussianHMM

from torchmm.hmm import HiddenMarkovModel
from torchmm.base import DiagNormalModel
from torchmm.utils import kmeans
from torchmm.utils import kmeans_init


GaussianHMM = partial(GaussianHMM,
                      transmat_prior=2,
                      startprob_prior=2,
                      covars_prior=1,
                      covars_weight=1,
                      means_prior=0,
                      means_weight=0,
                      n_iter=100,
                      # min_covar=5,
                      tol=1e-3,
                      )


def fit_hmm_learn(seqs, n_states, axis):
    """
    Seqs is a list of numpy vectors
    """
    samples = np.concatenate(seqs)
    lengths = np.array([len(s) for s in seqs])
    if len(samples) < n_states:
        return float('inf'), float('-inf'), None, None
    # assert len(samples) >= n_states
    hmm = GaussianHMM(n_components=n_states)
    hmm.fit(samples, lengths)

    ll = hmm.score(samples, lengths)
    _, labels = hmm.decode(samples, lengths)

    axis.set_title("HMM Learn (ll=%0.2f)" % ll)
    # ax2.plot(means[:, 0], means[:, 1], 'ro')
    # ax2.plot(X[:, :, 0], X[:, :, 1], 'bo')

    possible_colors = ['orange', 'blue', 'green', 'red']
    colors = [possible_colors[e] for e in labels]
    axis.scatter(seqs[:100, :, 0], seqs[:100, :, 1], color=colors[:100],
                 marker='^')
    axis.scatter(seqs[100:200, :, 0], seqs[100:200, :, 1],
                 color=colors[100:200], marker='o')
    axis.scatter(seqs[200:, :, 0], seqs[200:, :, 1], color=colors[200:],
                 marker='s')


def fit_with_random_restarts(seqs, n_states, restarts=100):
    T0 = torch.zeros(n_states).softmax(0)
    T = torch.zeros((n_states, n_states)).softmax(1)
    centroids = torch.zeros((n_states, 2))
    # centroids = kmeans_init(X.squeeze(), n_states)

    states = []
    for s_idx in range(n_states):
        precisions = torch.ones(2)
        states.append(DiagNormalModel(centroids[s_idx], precisions))

    hmm = HiddenMarkovModel(states, T0=T0, T=T)
    hmm.fit(seqs, restarts=restarts, randomize_first=True)

    ll = hmm.log_prob(X) + hmm.log_parameters_prob()
    print('best_ll', ll)
    return hmm


def fit_torchmm(seqs, n_states, axis):
    """
    Seqs is a tensor
    """
    hmm = fit_with_random_restarts(seqs, n_states)
    score = hmm.log_prob(seqs) + hmm.log_parameters_prob()
    labels = hmm.decode(seqs)
    print('ll', score)
    print('ll (no prior)', hmm.log_prob(seqs))

    axis.set_title("Torchmm (ll=%0.2f)" % score)
    possible_colors = ['orange', 'blue', 'green', 'red']
    colors = [possible_colors[e[0]] for e in labels[0]]
    axis.scatter(seqs[:100, :, 0], seqs[:100, :, 1], color=colors[:100],
                 marker='^')
    axis.scatter(seqs[100:200, :, 0], seqs[100:200, :, 1],
                 color=colors[100:200], marker='o')
    axis.scatter(seqs[200:, :, 0], seqs[200:, :, 1], color=colors[200:],
                 marker='s')


if __name__ == "__main__":
    X = torch.zeros((100, 2)).float()
    X.normal_(0, 2)
    Y = torch.zeros((100, 2)).float()
    Y.normal_(0, 0.25)
    Z = torch.zeros((100, 2)).float()
    Z.normal_(1, 0.25)
    X = torch.cat((X, Y, Z))

    X = torch.tensor(StandardScaler().fit_transform(X.cpu().detach().numpy()))

    X = X.unsqueeze(1)
    print(X.shape)
    X_np = X.numpy()

    n_states = 3

    f, (ax1, ax2) = plt.subplots(1, 2)

    fit_hmm_learn(X_np, n_states, axis=ax1)
    fit_torchmm(X, n_states, axis=ax2)

    plt.show()
