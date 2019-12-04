import numpy as np
import torch
from functools import partial

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from hmmlearn.hmm import GaussianHMM

from torchmm.hmm import HiddenMarkovModel
from torchmm.base import DiagNormalModel
from torchmm.utils import kmeans_init
from torchmm.utils import timefun


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


def fit_hmm_learn(seqs, n_states):
    """
    Seqs is a list of numpy vectors
    """
    samples = np.concatenate(seqs)
    lengths = np.array([len(s) for s in seqs])
    if len(samples) < n_states:
        return float('inf'), float('-inf'), None, None
    # assert len(samples) >= n_states
    hmm = GaussianHMM(n_components=n_states)

    @timefun
    def fit():
        hmm.fit(samples, lengths)

    fit()

    ll = hmm.score(samples, lengths)

    print("Number of states = %i" % hmm.n_components)
    print()
    print("Start Probabilities")
    print(np.array2string(hmm.startprob_, max_line_width=np.inf))
    print()
    print("Transition Probabilities")
    print(np.array2string(hmm.transmat_, max_line_width=np.inf))
    print()
    print("Emmission Means")
    print(np.array2string(hmm.means_, max_line_width=np.inf))
    print()
    print("Emmission Covariances")
    print(np.array2string(hmm.covars_, max_line_width=np.inf))
    print()

    means = hmm.means_
    print('covars shape')
    print(hmm.covars_.shape)

    var = [np.diag(c) for c in hmm.covars_]
    std = np.sqrt(np.stack(var))
    print(std)

    fig2, ax2 = plt.subplots()
    ax2.set_title("HMM Learn (ll=%0.2f)" % ll)
    ax2.plot(means[:, 0], means[:, 1], 'ro')
    ax2.plot(X[:, :, 0], X[:, :, 1], 'bo')

    for i in range(std.shape[0]):
        ellipse = Ellipse(means[i], width=(
            2 * std[i, 0]), height=(2 * std[i, 1]), edgecolor="red")
        ax2.add_patch(ellipse)


def fit_torchmm(seqs, n_states):
    """
    Seqs is a tensor
    """
    T0 = torch.zeros(n_states).softmax(0)
    T = torch.zeros((n_states, n_states)).softmax(1)

    centroids = kmeans_init(X.squeeze(), n_states)

    states = []
    for s_idx in range(n_states):
        precisions = torch.ones(2)
        states.append(DiagNormalModel(centroids[s_idx], precisions))

    hmm = HiddenMarkovModel(states, T0=T0, T=T)

    @timefun
    def fit():
        hmm.fit(X)

    fit()
    score = hmm.log_prob(X)  # + hmm.log_parameters_prob()
    print('ll', score)
    print('ll (no prior)', hmm.log_prob(X))

    print("Pi Matrix: ")
    print(hmm.T0)

    print("Transition Matrix: ")
    print(hmm.T)
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission Matrix: ")
    for s in hmm.states:
        print("Means")
        print(list(s.parameters())[0])

        print("Variance")
        print(1/list(s.parameters())[1])

    means = torch.stack([list(s.parameters())[0] for s in states])
    means = means.detach().numpy()
    precs = torch.stack([list(s.parameters())[1] for s in states])
    # precs = precs.detach().numpy()
    std = (1/precs.sqrt()).detach().numpy()
    print('std', std)

    fig1, ax1 = plt.subplots()
    ax1.set_title("Torchmm (ll=%0.2f)" % score)
    ax1.plot(means[:, 0], means[:, 1], 'ro')
    ax1.plot(X[:, :, 0], X[:, :, 1], 'bo')

    for i in range(precs.shape[0]):
        ellipse = Ellipse(means[i], width=(
            2 * std[i, 0]), height=(2 * std[i, 1]), edgecolor="red")
        ax1.add_patch(ellipse)


if __name__ == "__main__":
    X = torch.zeros((100, 2)).float()
    X.normal_(0, 2)
    Y = torch.zeros((100, 2)).float()
    Y.normal_(5, 2)
    X = torch.cat((X, Y))

    X = X.unsqueeze(1)
    print(X.shape)
    X_np = X.numpy()

    n_states = 2

    fit_hmm_learn(X_np, n_states)
    fit_torchmm(X, n_states)

    plt.show()
