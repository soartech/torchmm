import numpy as np
import torch
import torch.optim as optim

from torchmm.utils import pack_list
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence


class HiddenMarkovModel(object):
    """
    Hidden Markov self Class

    :param numpy.array T: Transition matrix of size S by S
    :param numpy.array E: Emission matrix of size N by S
    :param numpy.array T0: Initial state probabilities of size S.
    """

    def __init__(self, T, E, T0, epsilon=0.001, maxStep=10):

        if not np.allclose(np.sum(T, axis=1), 1):
            raise ValueError("Sum of T columns != 1.0")
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

    def fit(self, X, lengths=None, alg="viterbi"):
        """
        Learn new model parameters from X using the specified alg.

        Alg can be either 'baum_welch' or 'viterbi'.
        """
        packed_X = pack_list(X)

        if alg == 'baum_welch':
            return self._baum_welch(packed_X)
        elif alg == 'viterbi':
            return self._viterbi_training(packed_X)
        elif alg == 'autograd':
            return self._autograd(X)
        else:
            raise Exception("Unknown alg")

    def decode(self, X):
        """
        Find the most likely state sequences corresponding to X.

        .. todo::
            Modify this doc comment based on how we decide to pack/pad X.
        """
        X_packed = pack_list(X)
        self._init_viterbi(X_packed)
        state_seq, path_ll = self._viterbi_inference(X_packed)
        ss_packed = PackedSequence(
            data=state_seq, batch_sizes=X_packed.batch_sizes,
            sorted_indices=X_packed.sorted_indices,
            unsorted_indices=X_packed.unsorted_indices)
        ss_unpacked, ss_lengths = pad_packed_sequence(ss_packed,
                                                      batch_first=True)

        path_packed = PackedSequence(
            data=path_ll, batch_sizes=X_packed.batch_sizes,
            sorted_indices=X_packed.sorted_indices,
            unsorted_indices=X_packed.unsorted_indices)
        path_unpacked, path_lengths = pad_packed_sequence(path_packed,
                                                          batch_first=True)

        return ([ss_unpacked[i, :l] for i, l in enumerate(ss_lengths)],
                [path_unpacked[i, :l] for i, l in enumerate(path_lengths)])

    def smooth(self, X):
        """
        Compute the smoothed posterior probability over each state for the
        sequences in X.

        .. todo::
            Update this doc comment and update to reflect packed and padded seq
        """
        X_packed = pack_list(X)
        self._init_forw_back(X_packed)
        self.obs_ll_full = self._emission_ll(X_packed)
        self._forward_backward_inference(X_packed)

        post_packed = PackedSequence(
            data=self.posterior_ll, batch_sizes=X_packed.batch_sizes,
            sorted_indices=X_packed.sorted_indices,
            unsorted_indices=X_packed.unsorted_indices)
        post_unpacked, post_lengths = pad_packed_sequence(post_packed,
                                                          batch_first=True)
        return [post_unpacked[i, :l] for i, l in enumerate(post_lengths)]

    def filter(self, X):
        """
        Compute the log posterior distribution over the most recent state in
        each sequence-- given all the evidence to date for each.

        Filtering might also be referred to as state estimation.

        .. todo::
            Update this doc comment.
        """
        X_packed = pack_list(X)
        self.forward_ll = torch.zeros([len(X_packed.data), self.S],
                                      dtype=torch.float64)
        self.obs_ll_full = self._emission_ll(X_packed)
        self.batch_sizes = X_packed.batch_sizes
        self._forward()
        f_ll_packed = PackedSequence(
            data=self.forward_ll, batch_sizes=X_packed.batch_sizes,
            sorted_indices=X_packed.sorted_indices,
            unsorted_indices=X_packed.unsorted_indices)
        f_ll_unpacked, lengths = pad_packed_sequence(f_ll_packed,
                                                     batch_first=True)

        return f_ll_unpacked[torch.arange(f_ll_unpacked.size(0)), lengths-1]

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

        states[:, 0] = torch.multinomial(
            self.T0.unsqueeze(0).expand(n_seq, -1), 1).squeeze()
        obs[:, 0] = self._sample_states(states[:, 0]).squeeze()

        for t in range(1, n_obs):
            states[:, t] = torch.multinomial(
                self.T[states[:, t-1], :], 1).squeeze()
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
        return self.log_E[X.data]

    def _init_viterbi(self, X):
        """
        Initialize the parameters needed for _viterbi_inference.

        Kept seperate so initialization can be called only once when repeated
        inference calls are needed.
        """
        N = len(X.data)
        shape = [N, self.S]

        self.batch_sizes = X.batch_sizes

        # Init_viterbi_variables
        self.path_states = torch.zeros(shape, dtype=torch.float64)
        self.path_scores = torch.zeros_like(self.path_states)
        self.states_seq = torch.zeros([shape[0]], dtype=torch.int64)

    def _viterbi_inference(self, X):
        """
        Compute the most likely states given the current model and a sequence
        of observations (x).

        Returns the most likely state numbers and path score for each state.
        """
        N = len(X.batch_sizes)

        # log probability of emission sequence
        obs_ll_full = self._emission_ll(X)

        # initialize with state starting log-priors
        self.path_scores[0:self.batch_sizes[0]] = (
            self.log_T0 + obs_ll_full[0:self.batch_sizes[0]])

        idx = 0
        for step, prev_size in enumerate(self.batch_sizes[:-1]):
            start = idx
            mid = start + prev_size
            end = mid + self.batch_sizes[step+1]
            mv, mi = self._belief_prop_max(self.path_scores[start:mid])
            self.path_states[mid:end] = mi.squeeze(1)
            self.path_scores[mid:end] = mv + obs_ll_full[mid:end]
            idx = mid

        start = torch.zeros_like(self.batch_sizes)
        start[1:] = torch.cumsum(self.batch_sizes[:-1], 0)
        end = torch.cumsum(self.batch_sizes, 0)

        self.states_seq[start[N-1]:end[N-1]] = torch.argmax(
            self.path_scores[start[N-1]:end[N-1]], 1)

        for step in range(N-1, 0, -1):
            state = self.states_seq[start[step]:end[step]]
            state_prob = self.path_states[start[step]:end[step]]
            state_prob = state_prob.gather(
                1, state.unsqueeze(0).permute(1, 0)).squeeze(1)
            self.states_seq[start[step-1]:start[step-1] +
                            self.batch_sizes[step]] = state_prob

            # since we're doing packed sequencing, we need to check if
            # any new sequences are showing up for earlier times.
            # if so, initialize them
            if self.batch_sizes[step] > self.batch_sizes[step-1]:
                self.states_seq[
                    start[step-1] +
                    self.batch_sizes[step]:end[step-1]] = torch.argmax(
                        self.path_scores[start[step-1] +
                                         self.batch_sizes[step]:end[step-1]],
                        1)

        # # infer most likely last state
        # self.states_seq[N-1] = torch.argmax(self.path_scores[N-1, :], 0)
        # for step in range(N-1, 0, -1):
        #     # for every timestep retrieve inferred state
        #     state = self.states_seq[step]
        #     state_prob = self.path_states[step][state]
        #     self.states_seq[step - 1] = state_prob

        return self.states_seq, self.path_scores

    def _forward(self):
        """
        Computes the forward values.
        """
        self.forward_ll[0:self.batch_sizes[0]] = (
            self.log_T0 + self.obs_ll_full[0:self.batch_sizes[0]])

        idx = 0
        for step, prev_size in enumerate(self.batch_sizes[:-1]):
            start = idx
            mid = start + prev_size
            end = mid + self.batch_sizes[step+1]
            self.forward_ll[mid:end] = (
                self._belief_prop_sum(self.forward_ll[start:mid]) +
                self.obs_ll_full[mid:end])
            idx = mid

        # for step, obs_ll in enumerate(self.obs_ll_full[1:]):
        #     self.forward_ll[step+1] = (
        #         self._belief_prop_sum(self.forward_ll[step, :]) + obs_ll)

    def _backward(self):
        """
        Computes the backward values.
        """
        T = len(self.batch_sizes)
        start = torch.zeros_like(self.batch_sizes)
        start[1:] = torch.cumsum(self.batch_sizes[:-1], 0)
        end = torch.cumsum(self.batch_sizes, 0)

        self.backward_ll[start[T-1]:end[T-1]] = (
            torch.ones([self.batch_sizes[T-1], self.S], dtype=torch.float64))

        for t in range(T-1, 0, -1):
            self.backward_ll[start[t-1]:start[t-1] +
                             self.batch_sizes[t]] = self._belief_prop_sum(
                self.backward_ll[start[t]:end[t]] +
                self.obs_ll_full[start[t]:end[t]])

            if self.batch_sizes[t] < self.batch_sizes[t-1]:
                self.backward_ll[start[t-1] +
                                 self.batch_sizes[t]: end[t-1]] = (
                    torch.ones([self.batch_sizes[t-1] - self.batch_sizes[t],
                                self.S], dtype=torch.float64))

        # self.backward_ll[0] = torch.ones([self.S], dtype=torch.float64)
        # for step, obs_prob in enumerate(self.obs_ll_full.flip([0, 1])[:-1]):
        #     self.backward_ll[step+1] = self._belief_prop_sum(
        #         self.backward_ll[step, :] + obs_prob)
        # self.backward_ll = self.backward_ll.flip([0, 1])

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

    def _init_forw_back(self, obs_seq):
        N = len(obs_seq.data)
        shape = [N, self.S]
        self.batch_sizes = obs_seq.batch_sizes
        self.forward_ll = torch.zeros(shape, dtype=torch.float64)
        self.backward_ll = torch.zeros_like(self.forward_ll)
        self.posterior_ll = torch.zeros_like(self.forward_ll)

    def _re_estimate_transition_ll(self, X):
        """
        Updates the transmission probabilities, given a sequence.

        Assumes the forward backward step has already been run and
        self.forward_ll, self.backward_ll, and self.posterior_ll are set.

        Great reference: https://web.stanford.edu/~jurafsky/slp3/A.pdf
        """
        T = len(self.batch_sizes)
        start = torch.zeros_like(self.batch_sizes)
        start[1:] = torch.cumsum(self.batch_sizes[:-1], 0)
        end = torch.cumsum(self.batch_sizes, 0)

        # Compute T0, using normalized forward.
        log_T0_new = (self.forward_ll[start[0]:end[0]] -
                      torch.logsumexp(self.forward_ll[start[0]:end[0]], 0))
        print('T0')
        print(log_T0_new)

        # log_T0_new = (self.forward_ll[0] -
        #               torch.logsumexp(self.forward_ll[0], 0))

        # Compute expected transitions
        M_ll = (self.forward_ll[:-1].unsqueeze(2).expand(-1, -1, 2) +
                self.log_T.expand(T-1, -1, -1) +
                self.obs_ll_full[1:].unsqueeze(2).expand(-1, -1, 2) +
                self.backward_ll[1:].unsqueeze(2).expand(-1, -1, 2))

        denom = self.posterior_ll[:-1].logsumexp(1)
        M_ll -= denom.unsqueeze(1).unsqueeze(2).expand(-1, self.S, self.S)

        # Update log_T
        log_T_new = (torch.logsumexp(M_ll, 0) -
                     torch.logsumexp(M_ll, (0, 2)).view(-1, 1))

        return log_T0_new, log_T_new

    def _re_estimate_emission_ll(self, X):
        """
        Updates the emission probabilities, given an X.

        Assumes the forward-backward step has already been run and
        self.posterior_ll is updated.

        TODO:
            - Rewrite emission score in terms of log prob
        """
        m = torch.logsumexp(self.posterior_ll, 1)
        gamma_ll = self.posterior_ll - m.view(-1, 1)
        gamma = torch.exp(gamma_ll)
        states_marginal = gamma.sum(0)

        # One hot encoding buffer that you create out of the loop and just keep
        # reusing
        seq_one_hot = torch.zeros([len(X.data), self.Obs], dtype=torch.float64)
        seq_one_hot.scatter_(1, X.data.unsqueeze(1), 1)
        emission_score = torch.matmul(seq_one_hot.transpose_(1, 0), gamma)

        new_E = torch.log(emission_score / states_marginal)
        # print(new_E.exp())
        return new_E

    @property
    def converged(self):
        return (len(self.ll_history) >= 2 and
                abs(self.ll_history[-2] - self.ll_history[-1]) < self.epsilon)

    def _expectation_maximization_step(self, obs_seq):

        self.obs_ll_full = self._emission_ll(obs_seq)
        self._forward_backward_inference(obs_seq)

        # TODO Refactor this bit -- appears in at least 3 places.
        f_ll_packed = PackedSequence(
            data=self.forward_ll, batch_sizes=obs_seq.batch_sizes,
            sorted_indices=obs_seq.sorted_indices,
            unsorted_indices=obs_seq.unsorted_indices)
        f_ll_unpacked, lengths = pad_packed_sequence(f_ll_packed,
                                                     batch_first=True)
        self.ll_history.append(f_ll_unpacked[:, lengths-1].squeeze(1).sum())
        # self.ll_history.append(self.forward_ll[-1].sum())

        log_T0_new, log_T_new = self._re_estimate_transition_ll(obs_seq)
        log_E_new = self._re_estimate_emission_ll(obs_seq)

        self.log_T0 = log_T0_new
        self.log_E = log_E_new
        self.log_T = log_T_new

        return self.converged

    def _viterbi_training_step(self, obs_seq):

        # for convergence testing
        self.obs_ll_full = self._emission_ll(obs_seq)
        self._forward()

        # TODO Refactor this bit -- appears in at least 3 places.
        f_ll_packed = PackedSequence(
            data=self.forward_ll, batch_sizes=obs_seq.batch_sizes,
            sorted_indices=obs_seq.sorted_indices,
            unsorted_indices=obs_seq.unsorted_indices)
        f_ll_unpacked, lengths = pad_packed_sequence(f_ll_packed,
                                                     batch_first=True)
        self.ll_history.append(
            f_ll_unpacked[torch.arange(f_ll_unpacked.size(0)),
                          lengths-1].logsumexp(1).sum(0).item())
        # self.ll_history.append(self.forward_ll[-1].sum())

        # do the updating
        states, _ = self._viterbi_inference(obs_seq)

        # start prob
        s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.log_T0 = torch.log(s_counts.float() /
                                s_counts.sum()).type(torch.float64)

        # transition
        t_counts = torch.zeros_like(self.log_T)

        idx = 0
        for t, prev_size in enumerate(self.batch_sizes[:-1]):
            start = idx
            mid = start + prev_size
            # end = mid + self.batch_sizes[t+1]
            for i in range(self.batch_sizes[t+1]):
                t_counts[states[start:start+i+1], states[mid:mid+i+1]] += 1
            idx = mid

        self.log_T = torch.log(t_counts /
                               t_counts.sum(1).view(-1, 1))

        # emission
        self._update_emissions_viterbi_training(states, obs_seq)

        return self.converged

    def _update_emissions_viterbi_training(self, states, obs_seq):
        emit_counts = torch.zeros_like(self.log_E)
        for i, s in enumerate(states):
            emit_counts[obs_seq.data[i], s] += 1
        self.log_E = torch.log(emit_counts / emit_counts.sum(0))

    def _viterbi_training(self, obs_seq):

        # initialize variables
        self._init_viterbi(obs_seq)

        # used for convergence testing.
        self.forward_ll = torch.zeros([len(obs_seq.data), self.S],
                                      dtype=torch.float64)
        self.ll_history = []

        converged = False

        for i in range(self.maxStep):
            converged = self._viterbi_training_step(obs_seq)
            if converged:
                print('converged at step {}'.format(i))
                break

        print('HISTORY!')
        print(self.ll_history)
        return self.log_T0, self.log_T, self.log_E, converged

    def _baum_welch(self, obs_seq):
        # initialize variables
        self._init_forw_back(obs_seq)

        # used for convergence testing.
        self.ll_history = []

        converged = False

        for i in range(self.maxStep):
            converged = self._expectation_maximization_step(obs_seq)
            if converged:
                print('converged at step {}'.format(i))
                break
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
        optimizer = optim.AdamW([inner_T0, inner_E, inner_T], lr=1)

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
            ll = self.score(X)
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
    True_pi = np.array([0.75, 0.25])

    True_T = np.array([[0.85, 0.15],
                       [0.12, 0.88]])

    True_E = np.array([[0.99, 0.05],
                       [0.01, 0.95]])

    true_model = HiddenMarkovModel(True_T, True_E, True_pi)
    obs_seq, states = true_model.sample(100, 50)

    print("First 5 Obersvations:  ", obs_seq[0, :5])
    print("First 5 Hidden States: ", states[0, :5])

    init_pi = np.array([0.5, 0.5])

    init_T = np.array([[0.6, 0.4],
                       [0.5, 0.5]])

    init_E = np.array([[0.5, 0.5],
                       [0.5, 0.5]])

    model = HiddenMarkovModel(init_T, init_E, init_pi,
                              epsilon=1e-2, maxStep=500)

    trans0, transition, emission, converge = model.fit(obs_seq, alg="autograd")

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_pi)
    print("Pi Matrix: ")
    print(trans0.exp())

    print("Transition Matrix: ")
    print(transition.exp())
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission Matrix: ")
    print(emission.exp())
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

    states_seq, _ = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = torch.stack(states_seq)
    # pred = states_seq
    true = states
    accuracy = torch.mean(torch.abs(pred - true).float())
    print("Accuracy: ", accuracy)
