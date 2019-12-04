import torch

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from torchmm.hmm import HiddenMarkovModel
from torchmm.base import DiagNormalModel
from torchmm.utils import kmeans_init


if __name__ == "__main__":
    X = torch.zeros((100, 2)).float()
    X.normal_(0, 2)
    Y = torch.zeros((100, 2)).float()
    Y.normal_(5, 2)
    X = torch.cat((X, Y))
    print(X)

    # plt.plot(X[:, 0], X[:, 1], 'bo')
    # plt.show()

    n_states = 5
    T0 = torch.zeros(n_states).softmax(0)
    T = torch.zeros((n_states, n_states)).softmax(1)

    centroids = kmeans_init(X, n_states)
    print("CENTROIDS")
    print(centroids)

    states = []
    for s_idx in range(n_states):
        precisions = torch.ones(2)
        states.append(DiagNormalModel(centroids[s_idx], precisions))

    hmm = HiddenMarkovModel(states, T0=T0, T=T)

    X = X.unsqueeze(1)
    print(X.shape)

    hmm = HiddenMarkovModel(states, T0=T0, T=T)
    hmm.fit(X, alg="autograd", max_steps=100000, lr=1e-2, eps=1e-9)
    score = hmm.log_prob(X) + hmm.log_parameters_prob()
    print('ll', score)
    print('ll (no prior)', hmm.log_prob(X))

    means = torch.stack([list(s.parameters())[0] for s in states])
    means = means.detach().numpy()
    precs = torch.stack([list(s.parameters())[1] for s in states])
    # precs = precs.detach().numpy()
    std = (1/precs.sqrt()).detach().numpy()
    print('std', std)

    fig1, ax1 = plt.subplots()
    ax1.set_title("Gradient Descent (ll=%0.2f)" % score)
    ax1.plot(means[:, 0], means[:, 1], 'ro')
    ax1.plot(X[:, :, 0], X[:, :, 1], 'bo')

    for i in range(precs.shape[0]):
        ellipse = Ellipse(means[i], width=(
            2 * std[i, 0]), height=(2 * std[i, 1]), edgecolor="red")
        ax1.add_patch(ellipse)

    # plt.show()

    hmm = HiddenMarkovModel(states, T0=T0, T=T)
    hmm.fit(X, alg="viterbi", max_steps=1000)
    score = hmm.log_prob(X) + hmm.log_parameters_prob()
    print('ll', score)
    print('ll (no prior)', hmm.log_prob(X))

    means = torch.stack([list(s.parameters())[0] for s in states])
    means = means.detach().numpy()
    precs = torch.stack([list(s.parameters())[1] for s in states])
    # precs = precs.detach().numpy()
    std = (1/precs.sqrt()).detach().numpy()
    print('std', std)

    fig2, ax2 = plt.subplots()
    ax2.set_title("Viterbi (ll=%0.2f)" % score)
    ax2.plot(means[:, 0], means[:, 1], 'ro')
    ax2.plot(X[:, :, 0], X[:, :, 1], 'bo')

    for i in range(precs.shape[0]):
        ellipse = Ellipse(means[i], width=(
            2 * std[i, 0]), height=(2 * std[i, 1]), edgecolor="red")
        ax2.add_patch(ellipse)

    plt.show()
