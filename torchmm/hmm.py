import torch
from torch.distributions.dirichlet import Dirichlet

from torchmm.base import Model
from torchmm.base import CategoricalModel


class HiddenMarkovModel(Model):
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

    def __init__(self, states, T0=None, T=None, T0_prior=None, T_prior=None):
        """
        Constructor for the HMM accepts.

        This model requires a list of states, which define the emission models
        (e.g., categorical model, diagnormal model).

        Additionally, it is typical to specify start probabilities (T0) for
        each state as well as transition probabilities (T) from state to state.
        If these are missing, then the model assumes that the start and
        transition probabilites are uniform and equal (e.g., for a 3 state
        model, the start probabilities would be 0.33, 0.33, and 0.33).

        The model also accepts Dirchlet priors over the start and transition
        probabilites. Both priors have the same length as their respective
        start and transition parameters (priors can be specified for each state
        and state to state transtion). Dirchlet priors correspond intuitively
        to having previously observed counts over the number of times each
        start/transition was observed. If none are provided, then the are
        assumed to be ones (i.e., add-one laplace smoothing).

        Note, internally the model converts probabilities into
        log-probabilities to prevent underflows.
        """
        for s in states:
            if not isinstance(s, Model):
                raise ValueError("States must be models")

        if T0 is not None and not isinstance(T0, torch.Tensor):
            raise ValueError("T0 must be a torch tensor.")
        if T0 is not None and not torch.allclose(T0.sum(0),
                                                 torch.tensor([1.])):
            raise ValueError("Sum of T0 != 1.0")

        if T is not None and not isinstance(T, torch.Tensor):
            raise ValueError("T must be a torch tensor.")
        if T is not None and not torch.allclose(T.sum(1), torch.tensor([1.])):
            raise ValueError("Sum of T rows != 1.0")

        if T is not None and T.shape[0] != T.shape[1]:
            raise ValueError("T must be a square matrix")
        if T is not None and T.shape[0] != len(states):
            raise ValueError("E has incorrect number of states")
        if T0 is not None and T0.shape[0] != len(states):
            raise ValueError("T0 has incorrect number of states")

        self.states = states
        self.device = "cpu"

        if T0 is not None:
            self.log_T0 = T0.log()
        else:
            self.log_T0 = torch.zeros([len(states)]).float()

        if T is not None:
            self.log_T = T.log()
        else:
            self.log_T = torch.zeros([len(states), len(states)]).float()

        if T0_prior is None:
            self.T0_prior = torch.ones_like(self.log_T0)
        else:
            self.T0_prior = T0_prior

        if T_prior is None:
            self.T_prior = torch.ones_like(self.log_T)
        else:
            self.T_prior = T_prior

    def to(self, device):
        """
        Moves the model's parameters / tensors to the specified pytorch device.
        """
        self.device = device
        self.log_T0 = self.log_T0.to(device)
        self.log_T = self.log_T.to(device)
        self.T0_prior = self.T0_prior.to(device)
        self.T_prior = self.T_prior.to(device)

        for s in self.states:
            s.to(device)

    def log_parameters_prob(self):
        """
        Computes the log probability of the parameters given priors.
        """
        ll = Dirichlet(self.T0_prior).log_prob(self.T0)
        ll += Dirichlet(self.T_prior).log_prob(self.T).sum(0)
        for s in self.states:
            ll += s.log_parameters_prob()
        return ll

    def init_params_random(self):
        """
        Randomly sets the parameters of the model using the dirchlet priors.
        """
        self.log_T0 = Dirichlet(self.T0_prior).sample().log()
        self.log_T = Dirichlet(self.T_prior).sample().log()
        for s in self.states:
            s.init_params_random()

    def parameters(self):
        """
        Returns the set of parameters for optimization.

        This makes it possible to nest models.

        .. todo::
            Need to rewrite to somehow check for parameters already returned.
            For example, if states share the same emission parameters somehow,
            then we only want them returned once.
        """
        yield self.log_T0
        yield self.log_T
        for s in self.states:
            for p in s.parameters():
                yield p

    @property
    def T0(self):
        """
        Returns the start probabilites (convert from log-probs to probs).
        """
        return self.log_T0.exp()

    @property
    def T(self):
        """
        Returns the transition probabilites (convert from log-probs to probs).
        """
        return self.log_T.exp()

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

    def log_prob(self, X):
        """
        Computes the log likelihood of the data X given the model.

        X should have format N x O, where N is number of sequences and O is
        number of observations in that sequence. Note, this requires all
        sequences to have the same number of observations.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: tensor(1).
        """
        return self.filter(X).logsumexp(1).sum(0)

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

    def predict(self, X):
        """
        Compute the posterior distributions over the next (future) states for
        each sequence in X. Predicts 1 step into the future for each sequence.

        .. todo::
            Update to accept a number of timesteps to project into the future.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: tensor with shape N x S, where S is the number of states
        """
        states = self.filter(X)
        return self._belief_prop_sum(states)

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

    def fit(self, X, max_steps=500, epsilon=1e-2, randomize_first=False,
            **kwargs):
        """
        Learn new model parameters from X using the specified alg.

        :param X: sequence/observation data
        :type X: tensor with shape N x O x F, where N is number of sequences, O
            is number of observations, and F is number of emission/features
        :returns: None
        """
        if randomize_first:
            self.init_params_random()

        return self._viterbi_training(X, max_steps=max_steps,
                                      epsilon=epsilon, **kwargs)

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

    @property
    def converged(self):
        """
        Specifies the convergence test for training.
        """
        return (len(self.ll_history) >= 2 and
                (self.ll_history[-2] - self.ll_history[-1]).abs() <
                self.epsilon)


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
