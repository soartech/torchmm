import torch

from torchmm.hmm import HiddenMarkovModel
from torchmm.base import CategoricalModel
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class HiddenMarkovModel(HiddenMarkovModel):
    """
    A Hidden Markov Model, defined by states (defined by other models that
    determine the likelihood of observation), initial start probabilities, and
    transition probabilities.

    Note, this model requires packed sequence data. If all the sequences have
    the same length and you don't want to have to deal with packing, then you
    can use the Hidden Markov Model from the hmm module instead.
    """

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
        self.update_log_params()

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
        self.update_log_params()

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
        self.update_log_params()
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
        self.update_log_params()
        self._init_forw_back(X)
        self.obs_ll_full = self._emission_ll(X)
        self._forward_backward_inference(X)
        post_packed = PackedSequence(
            data=self.posterior_ll, batch_sizes=X.batch_sizes,
            sorted_indices=X.sorted_indices,
            unsorted_indices=X.unsorted_indices)

        return post_packed

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
        self.update_log_params()
        self.ll_history.append(self.log_prob(X))
        states, _ = self.decode(X)

        # start prob
        s_counts = states.data[0:states.batch_sizes[0]].bincount(
            minlength=len(self.states)).float()
        # s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.logit_T0 = torch.log((s_counts + self.T0_prior) /
                                  (s_counts.sum() + self.T0_prior.sum()))

        # transition
        t_counts = torch.zeros_like(self.log_T).float()

        idx = 0
        for step, prev_size in enumerate(self.batch_sizes[:-1]):
            next_size = self.batch_sizes[step+1]
            start = idx
            mid = start + prev_size
            end = mid + next_size
            t_counts[states.data[start:start+next_size],
                     states.data[mid:end]] += 1
            idx = mid

        self.logit_T = torch.log((t_counts + self.T_prior) /
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
                print('converged at step {}'.format(i))
                break

        # print('HISTORY!')
        # print(self.ll_history)
        return self.converged


if __name__ == "__main__":
    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_orig = torch.tensor([0.99, 0.01]).log()
    s2_orig = torch.tensor([0.05, 0.95]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(50, 100)

    print("First 50 Obersvations:  ", obs_seq[0, :50])
    print("First 5 Hidden States: ", states[0, :5])

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.6, 0.4],
                      [0.5, 0.5]])
    s1_orig = torch.tensor([0.6, 0.4]).log()
    s2_orig = torch.tensor([0.5, 0.5]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    converge = model.fit(obs_seq, max_steps=500, epsilon=1e-2, alg="autograd")

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
        print([p.softmax(0) for p in s.parameters()])
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
    accuracy = torch.mean(torch.abs(pred - true).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1
