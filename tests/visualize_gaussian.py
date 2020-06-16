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

    X = X.unsqueeze(1)
    print(X.shape)

    hmm = HiddenMarkovModel(states, T0=T0, T=T)
    hmm.fit(X, alg="viterbi", max_steps=1000)
    score = hmm.log_prob(X) + hmm.log_parameters_prob()
    states = hmm.decode(X)[0]
    print('ll', score)
    print('ll (no prior)', hmm.log_prob(X))

    print(states)

    fig2, ax2 = plt.subplots()
    ax2.set_title("Viterbi (ll=%0.2f)" % score)
    ax2.plot(X[:, :, 0], X[:, :, 1], 'bo')

    plt.show()
