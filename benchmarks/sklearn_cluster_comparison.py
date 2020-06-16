# -*- coding: utf-8 -*-
from torchmm.utils import kmeans_init, kmeans
from torchmm.hmm import HiddenMarkovModel
from torchmm.base import CategoricalModel, DiagNormalModel
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import cycle, islice
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
"""
Created on Mon Jun  1 21:12:06 2020

@author: robert.sheline
"""

print(__doc__)


seed = round(time.time())
np.random.seed(seed)
torch.manual_seed(seed)


class torchmm_transform(TransformerMixin, BaseEstimator):
    def __init__(self, k, use_kpp=True):
        # self.T0 = torch.tensor([1/k]*k)
        # self.T = torch.tensor([torch.tensor([1/k]*k) for _ in range(k)])
        self.n_states = k
        self.use_kpp = use_kpp

    def fit(self, X, restarts=20):
        best_ll = float('-inf')
        n_states = self.n_states
        X = torch.tensor([[_x] for _x in X]).float()
        for _ in range(restarts):
            if self.use_kpp:
                centroids = kmeans_init(X.squeeze(), n_states)

            T0 = torch.zeros(n_states)
            T0 = T0.softmax(0)

            T = torch.zeros((n_states, n_states))
            T = T.softmax(1)
            states = []
            for s_idx in range(n_states):
                if self.use_kpp:
                    means = centroids[s_idx]
                else:
                    means = torch.ones(2)
                precisions = torch.ones(2)

                states.append(DiagNormalModel(means, precisions))

            hmm = HiddenMarkovModel(states, T0=T0, T=T)
            hmm.init_params_random()
            hmm.fit(X, max_steps=5000, epsilon=1e-2)

            ll = hmm.log_prob(X) + hmm.log_parameters_prob()
            if ll > best_ll:
                best_ll = ll
                self.best_hmm = hmm

        if self.use_kpp:
            print("torchmm k++")
        else:
            print("torchmm rand init")
        print(best_ll)

        return self

    def predict(self, X):
        X = torch.tensor([[_x] for _x in X]).float()
        return self.best_hmm.decode(X)[0].squeeze()


class kmeans_transform(TransformerMixin, BaseEstimator):
    def __init__(self, k):
        # self.T0 = torch.tensor([1/k]*k)
        # self.T = torch.tensor([torch.tensor([1/k]*k) for _ in range(k)])
        self.n_states = k

    def fit(self, X, restarts=15):
        best_ll = float('-inf')
        n_states = self.n_states
        X = torch.tensor([[_x] for _x in X]).float()
        for r in range(restarts):
            centroids = kmeans(X.squeeze(), n_states)

            T0 = torch.zeros(n_states).normal_() / 10
            T0 = T0.softmax(0)

            T = torch.zeros((n_states, n_states)).normal_() / 10
            T = T.softmax(1)

            states = []
            for s_idx in range(n_states):
                means = centroids[s_idx]
                precisions = torch.ones(2)
                states.append(DiagNormalModel(means, precisions))

            hmm = HiddenMarkovModel(states, T0=T0, T=T)
            ll = hmm.log_prob(X) + hmm.log_parameters_prob()
            if ll > best_ll:
                best_ll = ll
                self.best_hmm = hmm

        print("torchmm kmeans")
        print(best_ll)
        return self

    def predict(self, X):
        X = torch.tensor([[_x] for _x in X]).float()
        return self.best_hmm.decode(X)[0].squeeze()


if __name__ == "__main__":

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
        print()
        print("NEW DATA")
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        optics = cluster.OPTICS(min_samples=params['min_samples'],
                                xi=params['xi'],
                                min_cluster_size=params['min_cluster_size'])
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='diag')

        torchmm_model_1 = torchmm_transform(params['n_clusters'], use_kpp=True)
        torchmm_model_2 = torchmm_transform(
            params['n_clusters'], use_kpp=False)
        kmeans_model = kmeans_transform(params['n_clusters'])
        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            # ('AffinityPropagation', affinity_propagation),
            # ('MeanShift', ms),
            # ('SpectralClustering', spectral),
            # ('Ward', ward),
            # ('AgglomerativeClustering', average_linkage),
            # ('DBSCAN', dbscan),
            # ('OPTICS', optics),
            # ('Birch', birch),
            ('GaussianMixture', gmm),
            ('Torchmm - Random Init', torchmm_model_2),
            ('Torchmm - K++ Init', torchmm_model_1),
            ('Torchmm - Kmeans Init', kmeans_model)
        )

        # clustering_algorithms = (('MiniBatchKMeans', two_means),
        # ('Torchmm', torchmm_model))

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
            # print('y_pred: ', type(y_pred), ' .. ', y_pred)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            # plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
            # transform=plt.gca().transAxes, size=15,
            # horizontalalignment='right')
            plot_num += 1

    plt.show()
