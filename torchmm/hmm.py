import torch
import torch.optim as optim

from torchmm.base import Model
from torchmm.base import CategoricalModel


class HiddenMarkovModel(Model):
    """
    Hidden Markov Model
    """

    def __init__(self, states, T0=None, T=None):
        """
        states -> a list of emission models.
        Pi -> a tensor of start probs
        T -> a tensor of transition probs
        """
        self.states = states

        if T0 is not None:
            self.logit_T0 = T0.log()
            self.log_T0 = T0.log()
        if T is not None:
            self.logit_T = T.log()
            self.log_T = T.log()

        # self.update_log_params()

    def parameters(self):
        yield self.logit_T0
        yield self.logit_T
        for s in self.states:
            for p in s.parameters():
                yield p

    def update_log_params(self):
        self.log_T0 = self.logit_T0.softmax(0).log()
        self.log_T = self.logit_T.softmax(1).log()
        # self.log_T0 = self.logit_T0 - self.logit_T0.logsumexp(0)
        # self.log_T = self.logit_T - self.logit_T.logsumexp(0)
        # print(self.logit_T0.softmax(0))
        # self.log_T0 = self.logit_T0.softmax(0).log()
        # self.log_T = self.logit_T.softmax(0).log()

    # @property
    # def log_T0(self):
    #     print(self.logit_T0.softmax(0).log())
    #     print(self.logit_T0 - self.logit_T0.logsumexp(0))
    #     assert torch.allclose(self.logit_T0.softmax(0).log(), self.logit_T0 -
    #                           self.logit_T0.logsumexp(0))
    #     return self.logit_T0 - self.logit_T0.logsumexp(0)

    # @property
    # def log_T(self):
    #     return self.logit_T.softmax(1).log()
    #     # return self.logit_T - self.logit_T.logsumexp(1)

    @property
    def T0(self):
        return self.log_T0.softmax(0)

    @property
    def T(self):
        return self.log_T.softmax(1)

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

        return obs, states

    def log_prob(self, X):
        """
        Computes the log likelihood of the data X given the model.

        X should have format N x O, where N is number of sequences and O is
        number of observations in that sequence.
        """
        return self.filter(X).logsumexp(1).sum(0)

    def _emission_ll(self, X):
        ll = torch.zeros([X.shape[0], X.shape[1], len(self.states)]).float()
        for i, s in enumerate(self.states):
            ll[:, :, i] = s.log_prob(X)
        return ll

    def filter(self, X):
        """
        Compute the log posterior distribution over the most recent state in
        each sequence-- given all the evidence to date for each.

        Filtering might also be referred to as state estimation.

        .. todo::
            Update this doc comment.
        """
        self.update_log_params()
        self.forward_ll = torch.zeros([X.shape[0], X.shape[1],
                                       len(self.states)]).float()
        self.obs_ll_full = self._emission_ll(X)
        self._forward()
        return self.forward_ll[:, -1, :]

    def _forward(self):
        self.forward_ll[:, 0, :] = self.log_T0 + self.obs_ll_full[:, 0]
        for t in range(1, self.forward_ll.shape[1]):
            self.forward_ll[:, t, :] = (
                self._belief_prop_sum(self.forward_ll[:, t-1, :]) +
                self.obs_ll_full[:, t])

    def fit(self, X, max_steps=500, epsilon=1e-2, alg="viterbi"):
        """
        Learn new model parameters from X using the specified alg.

        Alg can be either 'baum_welch' or 'viterbi'.
        """
        if alg == 'viterbi':
            return self._viterbi_training(X, max_steps=max_steps,
                                          epsilon=epsilon)
        elif alg == 'autograd':
            return self._autograd(X, max_steps=max_steps, epsilon=epsilon)
        else:
            raise Exception("Unknown alg")

    def decode(self, X):
        """
        Find the most likely state sequences corresponding to X.

        .. todo::
            Modify this doc comment based on how we decide to pack/pad X.
        """
        self.update_log_params()
        self._init_viterbi(X)
        state_seq, path_ll = self._viterbi_inference(X)
        return state_seq, path_ll

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

    def _init_viterbi(self, X):
        """
        Initialize the parameters needed for _viterbi_inference.

        Kept seperate so initialization can be called only once when repeated
        inference calls are needed.
        """
        # X = torch.tensor(X)
        shape = [X.shape[0], X.shape[1], len(self.states)]

        # Init_viterbi_variables
        self.path_states = torch.zeros(shape).float()
        self.path_scores = torch.zeros_like(self.path_states)
        self.states_seq = torch.zeros_like(X).long()

    def _viterbi_training_step(self, X):

        # self.update_log_params()

        # for convergence testing
        self.obs_ll_full = self._emission_ll(X)
        self._forward()
        self.ll_history.append(
            self.forward_ll[:, -1, :].logsumexp(1).sum(0).item())

        # do the updating
        states, _ = self._viterbi_inference(X)

        # start prob
        s_counts = states[:, 0].bincount(minlength=len(self.states)).float()
        # s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.logit_T0 = torch.log(s_counts / s_counts.sum())

        # transition
        t_counts = torch.zeros_like(self.log_T).float()

        for t in range(X.shape[1]-1):
            # print(states[:, t], states[:, t+1])
            t_counts[states[:, t], states[:, t+1]] += 1

        self.logit_T = torch.log(t_counts /
                                 t_counts.sum(1).view(-1, 1))

        # emission
        self._update_emissions_viterbi_training(states, X)

    def _update_emissions_viterbi_training(self, states, obs_seq):
        for i, s in enumerate(self.states):
            s.fit(obs_seq[states == i])

    def _viterbi_training(self, X, max_steps=500, epsilon=1e-2):

        # initialize variables
        self._init_viterbi(X)

        # used for convergence testing.
        self.forward_ll = torch.zeros([X.shape[0], X.shape[1],
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

    def _autograd(self, X, max_steps=500, epsilon=1e-2):
        self.ll_history = []
        self.epsilon = epsilon

        for p in self.parameters():
            p.requires_grad_(True)
        optimizer = optim.AdamW(self.parameters(), lr=1)

        for i in range(max_steps):

            # print("CONVERGED", self.converged)
            if self.converged:
                break
            # print()
            print("STEP %i of %i" % (i, max_steps))
            # optimizer = np.random.choice(optimizers)
            ll = self.log_prob(X)
            self.ll_history.append(ll.item())
            loss = -1 * ll

            # print(ll)
            print("loss: ", loss)
            # print("T0: ", self.log_T0, self.T0)
            # print("T: ", self.log_T, self.T)
            # # print("E: ", self.log_E, self.log_E.exp())
            # print('GRAD T0: ', self.log_T0.grad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('LL HISTORY', self.ll_history[-3:])
        # from matplotlib import pyplot as plt
        # plt.plot(self.ll_history)
        # plt.show()

        return self.converged

    @property
    def converged(self):
        return (len(self.ll_history) >= 2 and
                abs(self.ll_history[-2] - self.ll_history[-1]) < self.epsilon)


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
