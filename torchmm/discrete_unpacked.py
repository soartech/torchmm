import logging

import numpy as np
import torch
import torch.optim as optim
log = logging.getLogger(__name__)


class HiddenMarkovModel(object):
    """
    Hidden Markov self Class

    :param numpy.array T: Transition matrix of size S by S
    :param numpy.array E: Emission matrix of size N by S
    :param numpy.array T0: Initial state probabilities of size S.
    """

    def __init__(self, T, E, T0, epsilon=0.001, maxStep=100, alpha=.1):
        if not np.allclose(np.sum(T, axis=1), 1):
            raise ValueError("Sum of T rows != 1.0")
        if not np.allclose(np.sum(E, axis=0), 1):
            raise ValueError("Sum of E columns != 1.0")
        if not np.allclose(np.sum(T0), 1):
            raise ValueError("Sum of T0 != 1.0")

        if T.shape[0] != T.shape[1]:
            raise ValueError("T must be a square matrix")
        if T.shape[0] != E.shape[1]:
            raise ValueError("E has incorrect number of states")
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
        # Number of possible observations
        self.Obs = E.shape[0]

        # Emission probability
        # self.E = torch.tensor(E)
        self.log_E = torch.log(torch.tensor(E))

        # Transition matrix
        # self.T = torch.tensor(T)
        self.log_T = torch.log(torch.tensor(T))

        # Initial state vector
        # self.T0 = torch.tensor(T0)
        self.log_T0 = torch.log(torch.tensor(T0))

        self.alpha = alpha  # learning rate for autograd

    @property
    def T0(self):
        return self.log_T0.exp()

    @property
    def T(self):
        return self.log_T.exp()

    @property
    def E(self):
        return self.log_E.exp()

    def num_states(self):
        """
        Number of states.
        """
        return self.S

    def num_emissions(self):
        """
        Number of possible emissions.
        """
        return self.Obs

    def num_params(self):
        """
        The number of free parameters in the model.
        """
        init_params = self.T0.shape[0] - 1
        transition_params = self.T.shape[0] * (self.T.shape[1] - 1)
        emission_params = self.T.shape[0] * (self.E.shape[0] - 1)
        return init_params + transition_params + emission_params

    def fit(self, X, alg="autograd"):
        """
        Learn new model parameters from X using the specified alg.

        Alg can be either 'baum_welch' or 'viterbi'.
        """
        if alg == 'baum_welch':
            return self._baum_welch(X)
        elif alg == "autograd":
            return self._autograd(X)
        else:
            return self._viterbi_training(X)

    def decode(self, X):
        """
        Find the most likely state sequences corresponding to X.

        .. todo::
            Modify this doc comment based on how we decide to pack/pad X.
        """
        self._init_viterbi(X)
        state_seq, path_ll = self._viterbi_inference(X)
        return state_seq, path_ll

    def smooth(self, X):
        """
        Compute the smoothed posterior probability over each state for the
        sequences in X.

        .. todo::
            Update this doc comment and update to reflect packed and padded seq
        """
        self._init_forw_back(X)
        self.obs_ll_full = self._emission_ll(X)
        self._forward_backward_inference(X)
        return self.posterior_ll

    def filter(self, X):
        """
        Compute the log posterior distribution over the most recent state in
        each sequence-- given all the evidence to date for each.

        Filtering might also be referred to as state estimation.

        .. todo::
            Update this doc comment.
        """
        X = torch.tensor(X)
        self.forward_ll = torch.zeros([X.shape[0], X.shape[1], self.S],
                                      dtype=torch.float64)
        self.obs_ll_full = self._emission_ll(X)
        self._forward()
        return self.forward_ll[:, -1, :]

    def predict(self, X):
        """
        Compute the posterior distributions over the next (future) states for
        each sequence in X. Predicts 1 step into the future for each sequence.

        .. todo::
            Update this doc comment.

        .. todo::
            Update to accept a number of timesteps to project.
        """
        states = self.filter(X)
        return self._belief_prop_sum(states)

    def score(self, X):
        """
        Compute the log likelihood of the observations given the model.

        .. todo::
            Update this doc comment.
        """
        return self.filter(X).logsumexp(1).sum(0)

    def sample(self, n_seq, n_obs):
        """
        Draws a sample from the HMM of length num_obs.

        .. todo::
            could move obs outside loop.
        """
        obs = torch.zeros([n_seq, n_obs], dtype=torch.long)
        states = torch.zeros([n_seq, n_obs], dtype=torch.long)

        # sample initial states from T0/T0
        states[:, 0] = torch.multinomial(
            self.T0.unsqueeze(0).expand(n_seq, -1), 1).squeeze()

        obs[:, 0] = self._sample_states(states[:, 0]).squeeze()

        for t in range(1, n_obs):
            states[:, t] = torch.multinomial(
                self.T[states[:, t - 1], :], 1).squeeze()
            obs[:, t] = self._sample_states(states[:, t]).squeeze()

        return obs, states

    def _sample_states(self, s):
        """
        Given an array-like of states, randomly samples emissions.
        """
        return torch.multinomial(self.E[:, s].permute(1, 0), 1)

    def _belief_prop_max(self, scores):
        """
        Propagates the scores over transition matrix. Returns the indices and
        values of the max for each state.

        Scores should have shape N x S, where N is the num seq and S is the
        num states.
        """
        mv, mi = torch.max(scores.unsqueeze(2).expand(-1, -1,
                                                      self.log_T.shape[0]) +
                           self.log_T.unsqueeze(0).expand(scores.shape[0], -1,
                                                          -1), 1)
        return mv.squeeze(1), mi.squeeze(1)
        # return torch.max(scores.view(-1, 1) + self.log_T, 0)

    def _belief_prop_sum(self, scores):
        """
        Propagates the scores over transition matrix. Returns the indices and
        values of the max for each state.

        Scores should have shape N x S, where N is the num seq and S is the
        num states.
        """
        s = torch.logsumexp(scores.unsqueeze(2).expand(-1, -1,
                                                       self.log_T.shape[0]) +
                            self.log_T.unsqueeze(0).expand(scores.shape[0], -1,
                                                           -1), 1)

        return s.squeeze(1)
        # return torch.logsumexp(scores.view(-1, 1) + self.log_T, 0)

    def _emission_ll(self, X):
        """
        Log likelihood of emissions for each state.

        Takes a packedsequences object.

        Returns a tensor of shape (num samples in x, num states).
        """

        return self.log_E[X.long().data]

    def _init_viterbi(self, X):
        """
        Initialize the parameters needed for _viterbi_inference.

        Kept seperate so initialization can be called only once when repeated
        inference calls are needed.
        """
        # X = torch.tensor(X)
        shape = [X.shape[0], X.shape[1], self.S]

        # Init_viterbi_variables
        self.path_states = torch.zeros(shape, dtype=torch.float64)
        self.path_scores = torch.zeros_like(self.path_states)
        self.states_seq = torch.zeros_like(X, dtype=torch.int64)

    def _viterbi_inference(self, X):
        """
        Compute the most likely states given the current model and a sequence
        of observations (x).

        Returns the most likely state numbers and path score for each state.
        """
        # X = torch.tensor(X)

        # log probability of emission sequence
        obs_ll_full = self._emission_ll(X)

        # initialize with state starting log-priors
        self.path_scores[:, 0] = self.log_T0 + obs_ll_full[:, 0]

        for t in range(1, obs_ll_full.shape[1]):
            # propagate state belief
            max_vals, max_indices = (
                self._belief_prop_max(self.path_scores[:, t-1, :]))

            # the inferred state by maximizing global function
            self.path_states[:, t] = max_indices

            # and update state and score matrices
            self.path_scores[:, t] = max_vals + obs_ll_full[:, t, :]

        # infer most likely last state
        self.states_seq[:, X.shape[1]-1] = torch.argmax(
            self.path_scores[:, X.shape[1]-1, :], 1)

        for t in range(X.shape[1]-1, 0, -1):
            state = self.states_seq[:, t]
            state_prob = (
                self.path_states[:, t].gather(1, state.unsqueeze(1)).squeeze())
            self.states_seq[:, t-1] = state_prob

        return self.states_seq, self.path_scores

    def _forward(self):
        self.forward_ll[:, 0, :] = self.log_T0 + self.obs_ll_full[:, 0]
        for t in range(1, self.forward_ll.shape[1]):
            self.forward_ll[:, t, :] = (
                self._belief_prop_sum(self.forward_ll[:, t-1, :]) +
                self.obs_ll_full[:, t])

    def _backward(self):
        """
        Computes the backward values.
        """
        N = self.obs_ll_full.shape[0]
        T = self.obs_ll_full.shape[1]
        self.backward_ll[:, T-1] = torch.ones([N, self.S], dtype=torch.float64)
        for t in range(T-1, 0, -1):
            self.backward_ll[:, t-1] = self._belief_prop_sum(
                self.backward_ll[:, t, :] + self.obs_ll_full[:, t, :])

    def _forward_backward_inference(self, x):
        """
        Computes the expected probability of each state for each time.
        Returns the posterior over all states.

        Assumes the following have been initalized:
            - self.forward_ll
            - self.backward_ll
            - self.posterior_ll
            - self.obs_ll_full
        """
        # Forward
        self._forward()

        # Backward
        self._backward()

        # Posterior
        self.posterior_ll = self.forward_ll + self.backward_ll

        # Return posterior
        return self.posterior_ll

    def _init_forw_back(self, X):
        shape = [X.shape[0], X.shape[1], self.S]
        self.forward_ll = torch.zeros(shape, dtype=torch.float64)
        self.backward_ll = torch.zeros_like(self.forward_ll)
        self.posterior_ll = torch.zeros_like(self.forward_ll)

    @property
    def converged(self):
        return (len(self.ll_history) >= 2 and abs(self.ll_history[-2] -
                self.ll_history[-1]) < self.epsilon)

    def _viterbi_training_step(self, X):

        # for convergence testing
        self.obs_ll_full = self._emission_ll(X)
        self._forward()
        self.ll_history.append(
            self.forward_ll[:, -1, :].logsumexp(1).sum(0).item())

        # do the updating
        states, _ = self._viterbi_inference(X)

        # start prob
        s_counts = states[:, 0].bincount(minlength=self.S)
        # s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.log_T0 = torch.log(s_counts.float() /
                                s_counts.sum()).type(torch.float64)

        # transition
        t_counts = torch.zeros_like(self.log_T)

        for t in range(X.shape[1]-1):
            # print(states[:, t], states[:, t+1])
            t_counts[states[:, t], states[:, t+1]] += 1

        self.log_T = torch.log(t_counts /
                               t_counts.sum(1).view(-1, 1))

        # emission
        self._update_emissions_viterbi_training(states, X)

        return self.converged

    def _update_emissions_viterbi_training(self, states, obs_seq):
        emit_counts = torch.zeros_like(self.log_E)
        for i, s in enumerate(states):
            emit_counts[obs_seq.data[i], s] += 1
        self.log_E = torch.log(emit_counts / emit_counts.sum(0))

    def _viterbi_training(self, X):

        # initialize variables
        self._init_viterbi(X)

        # used for convergence testing.
        self.forward_ll = torch.zeros([X.shape[0], X.shape[1], self.S],
                                      dtype=torch.float64)
        self.ll_history = []

        converged = False

        for i in range(self.maxStep):
            converged = self._viterbi_training_step(X)
            if converged:
                print('converged at step {}'.format(i))
                break

        print('HISTORY!')
        print(self.ll_history)
        return self.log_T0, self.log_T, self.log_E, converged

    def _autograd(self, X):
        # X = torch.tensor(X)
        # self.forward_ll = torch.zeros([X.shape[0], X.shape[1], self.S],
        #                               dtype=torch.float64)
        # self.obs_ll_full = self._emission_ll(X)
        self.ll_history = []

        inner_T = self.log_T.softmax(1).log()
        inner_E = self.log_E.softmax(0).log()
        inner_T0 = self.log_T0.softmax(0).log()
        inner_T.requires_grad_(True)
        inner_E.requires_grad_(True)
        inner_T0.requires_grad_(True)
        self.log_T0 = inner_T0.softmax(0).log()
        self.log_T = inner_T.softmax(1).log()
        self.log_E = inner_E.softmax(0).log()
        # optimizer = optim.SGD([inner_T0, inner_E, inner_T], lr=1e-3)
        optimizer = optim.AdamW([inner_T0, inner_E, inner_T], lr=1e-1)

        # self.log_T0.requires_grad_(True)
        # self.log_E.requires_grad_(True)
        # self.log_T.requires_grad_(True)
        # optimizer = optim.SGD([self.log_T0, self.log_E, self.log_T], lr=1e-3)

        # optimizers = [optim.SGD([self.log_T0], lr=1e-1),
        # optim.SGD([self.log_E], lr=1e-1),
        # optim.SGD([self.log_T], lr=1e-1)]
        for i in range(self.maxStep):
            # print("CONVERGED", self.converged)
            if self.converged:
                break
            # print()
            print("STEP %i of %i" % (i, self.maxStep))
            # optimizer = np.random.choice(optimizers)
            self.forward_ll = torch.zeros([X.shape[0], X.shape[1], self.S],
                                          dtype=torch.float64)
            self.obs_ll_full = self._emission_ll(X)

            self._forward()
            ll = self.forward_ll[:, -1, :].logsumexp(1).sum(0)
            self.ll_history.append(ll.item())
            loss = -1 * ll

            # print(ll)
            print("loss: ", loss)
            # print("T0: ", self.log_T0, self.log_T0.exp())
            # # print("T: ", self.log_T, self.log_T.exp())
            # # print("E: ", self.log_E, self.log_E.exp())
            # print('GRAD T0: ', self.log_T0.grad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # self.log_T0.data -= self.log_T0.data.logsumexp(0)
            # self.log_T.data -= self.log_T.data.logsumexp(1)
            # self.log_E.data -= self.log_E.data.logsumexp(0)

            self.log_T0 = inner_T0.softmax(0).log()
            self.log_T = inner_T.softmax(1).log()
            self.log_E = inner_E.softmax(0).log()
        # print('LL HISTORY', self.ll_history[-3:])
        # from matplotlib import pyplot as plt
        # plt.plot(self.ll_history)
        # plt.show()

        return self.log_T0, self.log_T, self.log_E, self.converged


if __name__ == "__main__":
    True_T0 = np.array([0.75, 0.25])

    True_T = np.array([[0.85, 0.15],
                       [0.12, 0.88]])

    True_E = np.array([[0.99, 0.05],
                       [0.01, 0.95]])

    true_model = HiddenMarkovModel(True_T, True_E, True_T0)
    obs_seq, states = true_model.sample(100, 50)

    init_T0 = np.array([0.5, 0.5])

    init_T = np.array([[0.4, 0.6],
                       [0.5, 0.5]])

    init_E = np.array([[0.5, 0.5],
                       [0.5, 0.5]])

    model = HiddenMarkovModel(init_T, init_E, init_T0,
                              epsilon=1e-4, maxStep=1000)
    _, _, _, converge = model.fit(obs_seq, alg="autograd")

    # model = HiddenMarkovModel(transition.exp().data.numpy(),
    # emission.exp().data.numpy(), trans0.exp().data.numpy())

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_T0)
    print("T0 Matrix: ")
    print(model.log_T0.exp())

    print("Transition Matrix: ")
    print(model.log_T.exp())
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission Matrix: ")
    print(model.log_E.exp())
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

    # print(type(obs_seq))
    states_seq, _ = model.decode(obs_seq)

    print(states_seq)

    # print(type(obs_seq), obs_seq.shape, obs_seq)
    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = states_seq.data.numpy()
    true = states.data.numpy()
    accuracy = np.mean(np.abs(pred - true))
    print("Accuracy: ", accuracy)
    # assert accuracy >= 0.9 or accuracy <= 0.1
