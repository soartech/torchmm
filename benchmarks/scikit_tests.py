from itertools import cycle, islice
from functools import partial

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from hmmlearn.hmm import GaussianHMM

from torchmm.hmm import HiddenMarkovModel
from torchmm.base import DiagNormalModel
from torchmm.utils import kmeans_init
from torchmm.utils import kmeans

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


# np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (blobs, {}),
    (no_structure, {})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)
    # print(params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    n_states = params['n_clusters']

    samples = X
    lengths = np.array([1 for s in X])

    # assert len(samples) >= n_states
    hmm = GaussianHMM(n_components=n_states)
    hmm.fit(samples, lengths)
    y_pred = hmm.predict(samples, lengths)

    means = hmm.means_
    # print('covars shape')
    # print(hmm.covars_.shape)

    var = [np.diag(c) for c in hmm.covars_]
    std = np.sqrt(np.stack(var))
    # print(std)

    plt.subplot(len(datasets), 2, plot_num)
    if i_dataset == 0:
        plt.title('hmmlearn', size=18)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_pred) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=10,
                color=colors[y_pred])
    plt.scatter(means[:, 0], means[:, 1], s=20, color='red')

    ax = plt.gca()
    for i in range(std.shape[0]):
        ellipse = Ellipse(means[i], width=(
            2 * std[i, 0]), height=(2 * std[i, 1]), alpha=0.3, edgecolor="red")
        ax.add_patch(ellipse)

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

    # TORCHMM

    X = torch.from_numpy(X).float().unsqueeze(1)
    # print(X.shape)

    # fit
    T0 = torch.zeros(n_states).softmax(0)
    T = torch.zeros((n_states, n_states)).softmax(1)

    init_centroids = kmeans(X.squeeze(), n_states)
    centroids = init_centroids.clone()

    states = []
    for s_idx in range(n_states):
        precisions = torch.ones(2)
        states.append(DiagNormalModel(centroids[s_idx], precisions))

    hmm = HiddenMarkovModel(states, T0=T0, T=T)

    converged = hmm.fit(X, epsilon=1e-8)

    print()
    print('CONVERGED', converged)
    print()

    score = hmm.log_prob(X)  # + hmm.log_parameters_prob()
    print('ll', score)
    # print('ll (no prior)', hmm.log_prob(X))

    print("Pi Matrix: ")
    print(hmm.T0)

    # print("Transition Matrix: ")
    # print(hmm.T)
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    # print()
    # print("Emission Matrix: ")
    # for s in hmm.states:
    #     print("Means")
    #     print(list(s.parameters())[0])

    #     print("Variance")
    #     print(1/list(s.parameters())[1])

    means = torch.stack([list(s.parameters())[0] for s in states])
    means = means.detach().numpy()
    precs = torch.stack([list(s.parameters())[1] for s in states])
    # precs = precs.detach().numpy()
    std = (1/precs.sqrt()).detach().numpy()
    # print('std', std)

    y_pred, _ = hmm.decode(X)
    y_pred = y_pred.squeeze(1)
    # END FIT

    plt.subplot(len(datasets), 2, plot_num)
    if i_dataset == 0:
        plt.title('torchmm', size=18)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_pred) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(X.squeeze(1)[:, 0], X.squeeze(1)[:, 1], s=10,
                color=colors[y_pred])
    plt.scatter(init_centroids[:, 0], init_centroids[:, 1], s=20,
                color="black")
    plt.scatter(means[:, 0], means[:, 1], s=20, color='red')

    ax = plt.gca()
    for i in range(precs.shape[0]):
        ellipse = Ellipse(means[i], width=(
            2 * std[i, 0]), height=(2 * std[i, 1]), alpha=0.3, edgecolor="red")
        ax.add_patch(ellipse)

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

plt.show()
