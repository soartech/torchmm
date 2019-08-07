"""
This will inherit from the discrete model and use the appropriate update
function.

This page has a nice review of the update equations for gaussian HMM:
https://sambaiga.github.io/ml/hmm/2017/06/12/hmm-gausian.html
"""
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

from torchmm.discrete import HiddenMarkovModel


class GaussianHMM(HiddenMarkovModel):

    def __init__(self, T, mu, sd, T0, epsilon=0.001, maxStep=10):
        if not np.allclose(np.sum(T, axis=1), 1):
            raise ValueError("Sum of T columns != 1.0")
        if not np.allclose(np.sum(T0), 1):
            raise ValueError("Sum of T0 != 1.0")

        if T.shape[0] != T.shape[1]:
            raise ValueError("T must be a square matrix")
        if T.shape[0] != mu.shape[0]:
            raise ValueError("mu has incorrect number of states")
        if T.shape[0] != sd.shape[0]:
            raise ValueError("sd has incorrect number of states")
        if mu.shape[1] != sd.shape[1]:
            raise ValueError("mu and sd must have same number of dimensions")
        if T.shape[0] != T0.shape[0]:
            raise ValueError("T0 has incorrect number of states")

        if epsilon <= 0:
            raise ValueError('Invalid value for epsilon, must be > 0')
        if maxStep <= 0:
            raise ValueError('Invalid value for maxStep, must be > 0')

        # Max number of iteration
        self.maxStep = maxStep
        # convergence criteria
        self.epsilon = epsilon
        # Number of possible states
        self.S = T.shape[0]

        # mean and sd of gaussian emissions
        self.mu = torch.tensor(mu)
        self.sd = torch.tensor(sd)
        self.mv = [MultivariateNormal(
            self.mu[i], torch.diag(self.sd[i].mul(self.sd[i])))
            for i in range(self.S)]

        # Transition matrix
        # self.T = torch.tensor(T)
        self.log_T = torch.log(torch.tensor(T))

        # Initial state vector
        # self.T0 = torch.tensor(T0)
        self.log_T0 = torch.log(torch.tensor(T0))

    def _emission_ll(self, X):
        """
        Returns a matrix of length N_data x N_states. Each entry contains the
        probability that a given datum would have been generated from a given
        state.
        """
        ll = torch.zeros([len(X.data), self.S], dtype=torch.float64)
        for i in range(self.S):
            ll[:, i] = self.mv[i].log_prob(X.data)
        return ll

    def _update_emissions_viterbi_training(self, states, obs_seq):
        new_mu = torch.zeros_like(self.mu)
        new_sd = torch.zeros_like(self.sd)
        for i, s in range(states):
            new_mu[i] = obs_seq[states == s].mean(0)
            new_sd[i] = obs_seq[states == s].std(0)

        self.mu = new_mu
        self.sd = new_sd
        self.mv = [MultivariateNormal(
            self.mu[i], torch.diag(self.sd[i].mul(self.sd[i])))
            for i in range(self.S)]
