import torch

from torchmm.base import Model
from torchmm.hmm_packed import HiddenMarkovModel
from torchmm.base import CategoricalModel


class HiddenMarkovModel(HiddenMarkovModel):
    """
    A Hidden Markov Model, defined by states (defined by other models that
    determine the likelihood of observation), initial start probabilities, and
    transition probabilities.

    Note, this model requires that all observation sequences are the same
    length. If the sequences have different lengths, then look at the Hidden
    Markov Model from the hmm_packed module.

    .. todo::
        Consider removing this model and only supporting packed sequences.
    """

    def sample(self, n_seq, n_obs):
        """
        Draws n_seq samples from the HMM. All sequences will have length
        num_obs.

        :param n_seq: Number of sequences in generated sample
        :type n_seq: int
        :param n_obs: Number of observations per sequence in generated sample
        :type n_obs: int
        :returns: two tensors of shape n_seq x n_obs. The first contains
            observations of length F, where F is the number of emissions
            defined by the HMM's state models. The second contains state
            labels.
        """
        test_sample = self.states[0].sample()
        shape = [n_seq, n_obs] + list(test_sample.shape)[1:]
        obs = torch.zeros(shape, device=self.device).type(test_sample.type())
        states = torch.zeros([n_seq, n_obs], device=self.device).long()

        # Sample the states
        states[:, 0] = torch.multinomial(
            self.T0.unsqueeze(0).expand(n_seq, -1), 1).squeeze()
        for t in range(1, n_obs):
            states[:, t] = torch.multinomial(
                self.T[states[:, t-1], :], 1).squeeze()

        # Sample the emissions
        for i, s in enumerate(self.states):
            idx = states == i
            obs[idx] = s.sample(idx.sum().unsqueeze(0))

        return obs, states

    def _emission_ll(self, X):
        """
        Compute the log-likelihood of each X from each state.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: tensor with shape N x O x F
        """
        ll = torch.zeros([X.shape[0], X.shape[1], len(self.states)],
                         device=self.device).float()
        for i, s in enumerate(self.states):
            ll[:, :, i] = s.log_prob(X)
        return ll

    def filter(self, X):
        """
        Compute the log posterior distribution over the last state in
        each sequence--given all the data in each sequence.

        Filtering might also be referred to as state estimation.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: tensor with shape N x S, where S is the number of states
        """
        self.forward_ll = torch.zeros([X.shape[0], X.shape[1],
                                       len(self.states)],
                                      device=self.device).float()
        self.obs_ll_full = self._emission_ll(X)
        self._forward()
        return self.forward_ll[:, -1, :]

    def _forward(self):
        """
        The forward algorithm.
        """
        self.forward_ll[:, 0, :] = self.log_T0 + self.obs_ll_full[:, 0]
        for t in range(1, self.forward_ll.shape[1]):
            self.forward_ll[:, t, :] = (
                self._belief_prop_sum(self.forward_ll[:, t-1, :]) +
                self.obs_ll_full[:, t])

    def _backward(self):
        """
        The backward algorithm.
        """
        N = self.obs_ll_full.shape[0]
        T = self.obs_ll_full.shape[1]
        self.backward_ll[:, T-1] = torch.ones([N, len(self.states)],
                                              device=self.device).float()
        for t in range(T-1, 0, -1):
            self.backward_ll[:, t-1] = self._belief_prop_sum(
                self.backward_ll[:, t, :] + self.obs_ll_full[:, t, :])


    def decode(self, X):
        """
        Find the most likely state sequences corresponding to each observation
        in X. Note the state assignments within a sequence are not independent
        and this gives the best joint set of state assignments for each
        sequence.

        This essentially finds the state assignments for each observation.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: two tensors, first is N x O and contains the state labels,
            the second is N x O and contains the previous state label
        """
        self._init_viterbi(X)
        state_seq, path_ll = self._viterbi_inference(X)
        return state_seq, path_ll

    def smooth(self, X):
        """
        Compute the smoothed posterior probability over each state for the
        sequences in X. Unlike decode, this computes the best state assignment
        for each observation independent of the other assignments.

        Note, using this to compute the state state labels for the observations
        in a sequence is incorrect! Use decode instead.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: N x O x S
        """
        self._init_forw_back(X)
        self.obs_ll_full = self._emission_ll(X)
        self._forward_backward_inference(X)
        return self.posterior_ll


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

    def _init_forw_back(self, X):
        """
        Forward backward algorithm.
        """
        shape = [X.shape[0], X.shape[1], len(self.states)]
        self.forward_ll = torch.zeros(shape, device=self.device).float()
        self.backward_ll = torch.zeros_like(self.forward_ll)
        self.posterior_ll = torch.zeros_like(self.forward_ll)

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

    def _init_viterbi(self, X):
        """
        Initialize the parameters needed for _viterbi_inference.

        Kept seperate so initialization can be called only once when repeated
        inference calls are needed.
        """
        # X = torch.tensor(X)
        shape = [X.shape[0], X.shape[1], len(self.states)]

        # Init_viterbi_variables
        self.path_states = torch.zeros(shape, device=self.device).float()
        self.path_scores = torch.zeros_like(self.path_states)
        self.states_seq = torch.zeros([X.shape[0], X.shape[1]],
                                      device=self.device).long()

    def _viterbi_training_step(self, X):
        """
        The inner viterbi training loop.

        Perform one step of viterbi training. Note this is different from
        viterbi inference.

        Viterbi training performes one expectation maximization. It computes
        the most likely states (using viterbi inference), then based on these
        hard state assignments it updates all the parameters using maximum
        likelihood or maximum a posteri if the respective models have priors.
        """
        # for convergence testing
        self.obs_ll_full = self._emission_ll(X)
        self._forward()
        self.ll_history.append(
            self.forward_ll[:, -1, :].logsumexp(1).sum(0) +
            self.log_parameters_prob())

        # do the updating
        states, _ = self._viterbi_inference(X)

        # start prob
        s_counts = states[:, 0].bincount(minlength=len(self.states)).float()
        # s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.log_T0 = torch.log((s_counts + self.T0_prior) /
                                (s_counts.sum() + self.T0_prior.sum()))

        # transition
        t_counts = torch.zeros_like(self.log_T).float()

        M = self.log_T.shape[0]

        for t in range(X.shape[1]-1):
            pairs = (states[:, t] * M +
                     states[:, t+1]).bincount(minlength=M*M).float()
            t_counts += pairs.reshape((M, M))

            # for from_state in states[:, t].unique():
            #     for to_state in states[:, t+1].unique():
            #         t_counts[from_state, to_state] += (
            #             (states[:, t] == from_state) *
            #             (states[:, t+1] == to_state)).sum()

        # for t in range(X.shape[1]-1):
        #     for from_state in states[:, t].unique():
        #         for to_state in states[:, t+1].unique():
        #             t_counts[from_state, to_state] += (
        #                 (states[:, t] == from_state) *
        #                 (states[:, t+1] == to_state)).sum()

        self.log_T = torch.log((t_counts + self.T_prior) /
                               (t_counts.sum(1) +
                                self.T_prior.sum(1)).view(-1, 1))

        # emission
        self._update_emissions_viterbi_training(states, X)

    def _update_emissions_viterbi_training(self, states, obs_seq):
        """
        Given the states assignments and the observations, update the state
        emission models to better represent the assigned observations.
        """
        for i, s in enumerate(self.states):
            s.fit(obs_seq[states == i])

    def _viterbi_training(self, X, max_steps=500, epsilon=1e-5):
        """
        The outer viterbi training loop. This iteratively executes viterbi
        training steps until the model has converged or the maximum number of
        iterations has been reached.
        """
        # initialize variables
        self._init_viterbi(X)

        # used for convergence testing.
        self.forward_ll = torch.zeros([X.shape[0], X.shape[1],
                                       len(self.states)],
                                      device=self.device).float()
        self.ll_history = []

        self.epsilon = epsilon

        for i in range(max_steps):
            self._viterbi_training_step(X)
            if self.converged:
                # print('converged at step {}'.format(i))
                break

        # print('HISTORY!')
        # print(self.ll_history)

        return self.converged


if __name__ == "__main__":
    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_orig = torch.tensor([1.0, 0.0])
    s2_orig = torch.tensor([0.0, 1.0])
    s1 = CategoricalModel(probs=s1_orig, prior=torch.zeros_like(s1_orig))
    s2 = CategoricalModel(probs=s2_orig, prior=torch.zeros_like(s2_orig))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T,
                              T0_prior=torch.tensor([0., 0.]),
                              T_prior=torch.tensor([[0., 0.], [0., 0.]]))
    obs_seq, states = model.sample(10, 1)
    # print(obs_seq[:, 0].sum())

    print("First 50 Obersvations of seq 0:  ", obs_seq[0, :50])
    print("First 5 Hidden States of seq 0: ", states[0, :5])

    T0 = torch.tensor([0.49, 0.51])
    T = torch.tensor([[0.6, 0.4],
                      [0.48, 0.52]])
    s1_orig = torch.tensor([0.9, 0.1])
    s2_orig = torch.tensor([0.2, 0.8])
    s1 = CategoricalModel(probs=s1_orig)
    s2 = CategoricalModel(probs=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    converge = model.fit(obs_seq, max_steps=1, epsilon=1e-2)

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_pi)
    print("Pi Matrix: ")
    print(model.T0)

    print("Transition Matrix: ")
    print(model.T)
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission Matrix: ")
    for s in model.states:
        print([p for p in s.parameters()])
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
    pred = states_seq
    true = states
    error = torch.mean(torch.abs(pred - true).float())
    print("Error: ", error)
    assert error >= 0.9 or error <= 0.1
