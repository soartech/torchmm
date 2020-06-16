import torch

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions.dirichlet import Dirichlet

from torchmm.base import Model
# from torchmm.hmm import HiddenMarkovModel
from torchmm.base import CategoricalModel
from torchmm.utils import unpack_list


class HiddenMarkovModel(Model):
    """
    A Hidden Markov Model, defined by states (defined by other models that
    determine the likelihood of observation), initial start probabilities, and
    transition probabilities.

    Note, this model requires packed sequence data. If all the sequences have
    the same length and you don't want to have to deal with packing, then you
    can use the Hidden Markov Model from the hmm module instead.
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

    def sample(self, n_seq, n_obs):
        """
        Draws n_seq samples from the HMM. All sequences will have length
        num_obs. The returned samples are packed.

        :param n_seq: Number of sequences in generated sample
        :type n_seq: int
        :param n_obs: Number of observations per sequence in generated sample
        :type n_obs: int
        :returns: Two packed sequence tensors. The first contains the
            observations and the second contains state labels.
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

        packed_obs = pack_padded_sequence(
            obs, torch.tensor(obs.shape[1]).unsqueeze(0).expand(obs.shape[0]),
            batch_first=True)
        packed_states = pack_padded_sequence(
            states, torch.tensor(
                states.shape[1]).unsqueeze(0).expand(states.shape[0]),
            batch_first=True)
        return packed_obs, packed_states

    def _emission_ll(self, X):
        """
        Computes the log-probability of every observation given the states.

        This is different than unpacked equivelent because there is one less
        dimension.
        """
        ll = torch.zeros([X.data.shape[0], len(self.states)],
                         device=self.device).float()
        for i, s in enumerate(self.states):
            ll[:, i] = s.log_prob(X.data)
        return ll

    def filter(self, X):
        """
        Compute the log posterior distribution over the last state in
        each sequence--given all the data in each sequence.

        Filtering might also be referred to as state estimation.

        .. todo::
            Remove the need to unpack the forward_ll var. Just read the packed
            representation directly.

        :param X: packed sequence/observation data
        :type X: packed sequence
        :returns: tensor with shape N x S, where N is number of sequences and S
            is the number of states.
        """
        self.forward_ll = torch.zeros([X.data.shape[0],
                                       len(self.states)],
                                      device=self.device).float()
        self.obs_ll_full = self._emission_ll(X)
        self.batch_sizes = X.batch_sizes
        self._forward()

        f_ll_packed = PackedSequence(
            data=self.forward_ll, batch_sizes=X.batch_sizes,
            sorted_indices=X.sorted_indices,
            unsorted_indices=X.unsorted_indices)
        f_ll_unpacked, lengths = pad_packed_sequence(f_ll_packed,
                                                     batch_first=True)

        return f_ll_unpacked[torch.arange(f_ll_unpacked.size(0)), lengths-1]

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
        The forward algorithm applied to packed sequences.
        """
        self.forward_ll[0:self.batch_sizes[0]] = (
            self.log_T0 + self.obs_ll_full[0:self.batch_sizes[0]])

        idx = 0
        for step, prev_size in enumerate(self.batch_sizes[:-1]):
            start = idx
            mid = start + prev_size
            mid_sub = start + self.batch_sizes[step+1]
            end = mid + self.batch_sizes[step+1]
            self.forward_ll[mid:end] = (
                self._belief_prop_sum(self.forward_ll[start:mid_sub]) +
                self.obs_ll_full[mid:end])
            idx = mid

    def _backward(self):
        """
        The backward algorithm applied to packed sequences.
        """
        T = len(self.batch_sizes)
        start = torch.zeros_like(self.batch_sizes)
        start[1:] = torch.cumsum(self.batch_sizes[:-1], 0)
        end = torch.cumsum(self.batch_sizes, 0)

        self.backward_ll[start[T-1]:end[T-1]] = (
            torch.ones([self.batch_sizes[T-1], len(self.states)],
                       device=self.device).float())

        for t in range(T-1, 0, -1):
            self.backward_ll[start[t-1]:start[t-1] +
                             self.batch_sizes[t]] = self._belief_prop_sum(
                self.backward_ll[start[t]:end[t]] +
                self.obs_ll_full[start[t]:end[t]])

            if self.batch_sizes[t] < self.batch_sizes[t-1]:
                self.backward_ll[start[t-1] +
                                 self.batch_sizes[t]: end[t-1]] = (
                    torch.ones([self.batch_sizes[t-1] - self.batch_sizes[t],
                                len(self.states)],
                               device=self.device).float())

    def decode(self, X):
        """
        Find the most likely state sequences corresponding to each observation
        in X. Note the state assignments within a sequence are not independent
        and this gives the best joint set of state assignments for each
        sequence.

        This essentially finds the state assignments for each observation.

        :param X: packed sequence/observation data
        :type X: packed sequence
        :returns: two packed sequence tensors. The first contains state labels,
            the second contains the previous state label (i.e., the path).
        """
        self._init_viterbi(X)
        state_seq, path_ll = self._viterbi_inference(X)

        ss_packed = PackedSequence(
            data=state_seq, batch_sizes=X.batch_sizes,
            sorted_indices=X.sorted_indices,
            unsorted_indices=X.unsorted_indices)
        path_packed = PackedSequence(
            data=path_ll, batch_sizes=X.batch_sizes,
            sorted_indices=X.sorted_indices,
            unsorted_indices=X.unsorted_indices)

        return ss_packed, path_packed

    def smooth(self, X):
        """
        Compute the smoothed posterior probability over each state for the
        sequences in X. Unlike decode, this computes the best state assignment
        for each observation independent of the other assignments.

        Note, using this to compute the state state labels for the observations
        in a sequence is incorrect! Use decode instead.

        :param X: packed sequence/observation data
        :type X: packed sequence
        :returns: a packed sequence tensors.
        """
        self._init_forw_back(X)
        self.obs_ll_full = self._emission_ll(X)
        self._forward_backward_inference(X)
        post_packed = PackedSequence(
            data=self.posterior_ll, batch_sizes=X.batch_sizes,
            sorted_indices=X.sorted_indices,
            unsorted_indices=X.unsorted_indices)

        return post_packed

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

    def fit(self, X, max_steps=500, epsilon=1e-3, randomize_first=False,
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
            mid_sub = start + self.batch_sizes[step+1]
            end = mid + self.batch_sizes[step+1]
            mv, mi = self._belief_prop_max(self.path_scores[start:mid_sub])
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

        return self.states_seq, self.path_scores

    def _init_forw_back(self, X):
        """
        Initialization for the forward backward algorithm.
        """
        N = len(X.data)
        shape = [N, len(self.states)]
        self.batch_sizes = X.batch_sizes
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
        shape = [X.data.shape[0], len(self.states)]

        self.batch_sizes = X.batch_sizes

        # Init_viterbi_variables
        self.path_states = torch.zeros(shape, device=self.device).float()
        self.path_scores = torch.zeros_like(self.path_states)
        self.states_seq = torch.zeros(X.data.shape[0],
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
        self.ll_history.append(self.log_prob(X) + self.log_parameters_prob())
        states, _ = self.decode(X)

        # start prob
        s_counts = states.data[0:states.batch_sizes[0]].bincount(
            minlength=len(self.states)).float()
        # s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.log_T0 = torch.log((s_counts + self.T0_prior) /
                                (s_counts.sum() + self.T0_prior.sum()))

        # transition
        t_counts = torch.zeros_like(self.log_T).float()

        M = self.log_T.shape[0]

        idx = 0
        for step, prev_size in enumerate(self.batch_sizes[:-1]):
            next_size = self.batch_sizes[step+1]
            start = idx
            mid = start + prev_size
            end = mid + next_size
            # t_counts[states.data[start:start+next_size],
            #          states.data[mid:end]] += 1
            pairs = (states.data[start:start+next_size] * M +
                     states.data[mid:end]).bincount(minlength=M*M).float()
            t_counts += pairs.reshape((M, M))

            idx = mid

        self.log_T = torch.log((t_counts + self.T_prior) /
                               (t_counts.sum(1).view(-1, 1) +
                                self.T_prior.sum(1).view(-1, 1)))

        # emission
        self._update_emissions_viterbi_training(states.data, X.data)

    def _update_emissions_viterbi_training(self, states, obs_seq):
        """
        Given the states assignments and the observations, update the state
        emission models to better represent the assigned observations.
        """
        for i, s in enumerate(self.states):
            s.fit(obs_seq[states == i])

    def _viterbi_training(self, X, max_steps=500, epsilon=1e-2):
        """
        The outer viterbi training loop. This iteratively executes viterbi
        training steps until the model has converged or the maximum number of
        iterations has been reached.
        """
        # initialize variables
        self._init_viterbi(X)

        # used for convergence testing.
        self.forward_ll = torch.zeros([X.data.shape[0],
                                       len(self.states)],
                                      device=self.device).float()
        self.ll_history = []

        self.epsilon = epsilon

        for i in range(max_steps):
            self._viterbi_training_step(X)
            if self.converged:
                print('converged at step %i' % i)
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
    T0 = torch.tensor([0.98, 0.02])
    T = torch.tensor([[0.9, 0.1],
                      [0.09, 0.91]])
    s1_orig = torch.tensor([0.99, 0.01])
    s2_orig = torch.tensor([0.02, 0.98])
    s1 = CategoricalModel(probs=s1_orig)
    s2 = CategoricalModel(probs=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(3000, 30)

    obs_seq_unpacked = unpack_list(obs_seq)
    print(torch.stack(obs_seq_unpacked))
    states_unpacked = unpack_list(states)

    print("First 50 Obersvations of seq 0:  ", obs_seq_unpacked[0][:50])
    print("First 5 Hidden States of seq 0: ", states_unpacked[0][:5])

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.6, 0.4],
                      [0.5, 0.5]])
    s1_orig = torch.tensor([0.6, 0.4])
    s2_orig = torch.tensor([0.5, 0.5])
    s1 = CategoricalModel(probs=s1_orig)
    s2 = CategoricalModel(probs=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    converge = model.fit(obs_seq, max_steps=500, epsilon=1e-2)

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
        # print([p.softmax(0) for p in s.parameters()])
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

    states_seq, _ = model.decode(obs_seq)

    states_seq_unpacked = unpack_list(states_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = torch.stack(unpack_list(states_seq))
    true = torch.stack(states_unpacked)
    accuracy = torch.mean(torch.abs(pred - true).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1
