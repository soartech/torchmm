import torch

from torchmm.hmm import HiddenMarkovModel
from torchmm.base import CategoricalModel
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class HiddenMarkovModel(HiddenMarkovModel):
    """
    Hidden Markov Model
    """

    def sample(self, n_seq, n_obs):
        """
        Draws n_seq samples from the HMM of length num_obs.
        """
        self.update_log_params()
        obs = torch.zeros([n_seq, n_obs], dtype=torch.long)
        states = torch.zeros([n_seq, n_obs], dtype=torch.long)

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
        ll = torch.zeros([X.data.shape[0], len(self.states)]).float()
        for i, s in enumerate(self.states):
            ll[:, i] = s.log_prob(X.data)
        return ll

    def filter(self, X):
        """
        Compute the log posterior distribution over the most recent state in
        each sequence-- given all the evidence to date for each.

        Filtering might also be referred to as state estimation.

        .. todo::
            Remove the need to unpack the forward_ll var. Just read the packed
            representation directly.
        """
        self.update_log_params()

        self.forward_ll = torch.zeros([X.data.shape[0],
                                       len(self.states)]).float()
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

    def _backward(self):
        """
        Computes the backward values.
        """
        T = len(self.batch_sizes)
        start = torch.zeros_like(self.batch_sizes)
        start[1:] = torch.cumsum(self.batch_sizes[:-1], 0)
        end = torch.cumsum(self.batch_sizes, 0)

        self.backward_ll[start[T-1]:end[T-1]] = (
            torch.ones([self.batch_sizes[T-1], len(self.states)]).float())

        for t in range(T-1, 0, -1):
            self.backward_ll[start[t-1]:start[t-1] +
                             self.batch_sizes[t]] = self._belief_prop_sum(
                self.backward_ll[start[t]:end[t]] +
                self.obs_ll_full[start[t]:end[t]])

            if self.batch_sizes[t] < self.batch_sizes[t-1]:
                self.backward_ll[start[t-1] +
                                 self.batch_sizes[t]: end[t-1]] = (
                    torch.ones([self.batch_sizes[t-1] - self.batch_sizes[t],
                                len(self.states)]).float())

    def decode(self, X):
        """
        Find the most likely state sequences corresponding to X.

        .. todo::
            Modify this doc comment based on how we decide to pack/pad X.
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
        sequences in X.

        .. todo::
            Update this doc comment and update to reflect packed and padded seq
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

        return self.states_seq, self.path_scores

    def _init_forw_back(self, X):
        N = len(X.data)
        shape = [N, len(self.states)]
        self.batch_sizes = X.batch_sizes
        self.forward_ll = torch.zeros(shape).float()
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
        self.path_states = torch.zeros(shape).float()
        self.path_scores = torch.zeros_like(self.path_states)
        self.states_seq = torch.zeros_like(X.data).long()

    def _viterbi_training_step(self, X):
        self.update_log_params()
        self.ll_history.append(self.log_prob(X).item())
        states, _ = self.decode(X)

        # start prob
        s_counts = states.data[0:states.batch_sizes[0]].bincount(
            minlength=len(self.states)).float()
        # s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.logit_T0 = torch.log(s_counts / s_counts.sum())

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

        self.logit_T = torch.log(t_counts /
                                 t_counts.sum(1).view(-1, 1))

        # emission
        self._update_emissions_viterbi_training(states.data, X.data)

    def _update_emissions_viterbi_training(self, states, obs_seq):
        for i, s in enumerate(self.states):
            s.fit(obs_seq[states == i])

    def _viterbi_training(self, X, max_steps=500, epsilon=1e-2):

        # initialize variables
        self._init_viterbi(X)

        # used for convergence testing.
        self.forward_ll = torch.zeros([X.data.shape[0],
                                       len(self.states)]).float()
        self.ll_history = []

        self.epsilon = epsilon

        for i in range(max_steps):
            self._viterbi_training_step(X)
            if self.converged:
                print('converged at step {}'.format(i))
                break

        print('HISTORY!')
        print(self.ll_history)
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
