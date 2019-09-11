import torch
import torch.optim as optim
from torch.distributions.dirichlet import Dirichlet
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
            self.logit_T0 = T0.log()
        else:
            self.logit_T0 = torch.zeros([len(states)]).float()

        if T is not None:
            self.logit_T = T.log()
        else:
            self.logit_T = torch.zeros([len(states), len(states)]).float()

        self.T0_prior = torch.ones_like(self.logit_T0)
        self.T_prior = torch.ones_like(self.logit_T)

    def to(self, device):
        self.device = device
        self.logit_T0 = self.logit_T0.to(device)
        self.logit_T = self.logit_T.to(device)
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
        self.logit_T0 = Dirichlet(self.T0_prior).sample().log()
        self.logit_T = Dirichlet(self.T_prior).sample().log()
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
        yield self.logit_T0
        yield self.logit_T
        for s in self.states:
            for p in s.parameters():
                yield p

    def update_log_params(self):
        self.log_T0 = self.logit_T0.softmax(0).log()
        self.log_T = self.logit_T.softmax(1).log()

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
        number of observations in that sequence.
        """
        return self.filter(X).logsumexp(1).sum(0)

    def _emission_ll(self, X):
        ll = torch.zeros([X.shape[0], X.shape[1], len(self.states)],
                         device=self.device).float()
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
            Update this doc comment.

        .. todo::
            Update to accept a number of timesteps to project.
        """
        states = self.filter(X)
        return self._belief_prop_sum(states)

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
        self.backward_ll[:, T-1] = torch.ones([N, len(self.states)],
                                              device=self.device).float()
        for t in range(T-1, 0, -1):
            self.backward_ll[:, t-1] = self._belief_prop_sum(
                self.backward_ll[:, t, :] + self.obs_ll_full[:, t, :])

    def fit(self, X, max_steps=500, epsilon=1e-2, alg="viterbi",
            randomize_first=False, **kwargs):
        """
        Learn new model parameters from X using the specified alg.

        Alg can be either 'baum_welch' or 'viterbi'.
        """
        if randomize_first:
            self.init_params_random()

        if alg == 'viterbi':
            return self._viterbi_training(X, max_steps=max_steps,
                                          epsilon=epsilon, **kwargs)
        elif alg == 'autograd':
            return self._autograd(X, max_steps=max_steps, epsilon=epsilon,
                                  **kwargs)
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

        self.update_log_params()

        # for convergence testing
        self.obs_ll_full = self._emission_ll(X)
        self._forward()
        self.ll_history.append(
            self.forward_ll[:, -1, :].logsumexp(1).sum(0))

        # do the updating
        states, _ = self._viterbi_inference(X)

        # start prob
        s_counts = states[:, 0].bincount(minlength=len(self.states)).float()
        # s_counts = states[:self.batch_sizes[0]].bincount(minlength=self.S)
        self.logit_T0 = torch.log(s_counts + self.T0_prior /
                                  (s_counts.sum() + self.T0_prior.sum()))

        # transition
        t_counts = torch.zeros_like(self.log_T).float()

        for t in range(X.shape[1]-1):
            # print(states[:, t], states[:, t+1])
            t_counts[states[:, t], states[:, t+1]] += 1

        self.logit_T = torch.log(t_counts + self.T_prior /
                                 (t_counts.sum(1) +
                                  self.T_prior.sum(1)).view(-1, 1))

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
                                       len(self.states)],
                                      device=self.device).float()
        self.ll_history = []

        self.epsilon = epsilon

        for i in range(max_steps):
            self._viterbi_training_step(X)
            if self.converged:
                print('converged at step {}'.format(i))
                break

        return self.converged

    def _autograd(self, X, max_steps=500, epsilon=1e-2, **kwargs):

        self.ll_history = []
        self.epsilon = epsilon

        for p in self.parameters():
            p.requires_grad_(True)

        # TODO Come up with a better way to translate kwargs into optimizer
        # maybe just pass the whole thing in.
        if 'lr' in kwargs:
            learning_rate = kwargs['lr']
        else:
            learning_rate = 1e-3

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

        for i in range(max_steps):

            # print("CONVERGED", self.converged)
            if self.converged:
                break
            # print()
            # print("STEP %i of %i" % (i, max_steps))
            # optimizer = np.random.choice(optimizers)
            ll = self.log_prob(X) + self.log_parameters_prob()
            self.ll_history.append(ll)
            loss = -1 * ll

            if torch.isnan(loss):
                from pprint import pprint
                pprint(list(self.parameters()))

            # print(ll)
            print("loss: ", loss, "(%i of %i)" % (i, max_steps))
            # print("T0: ", self.log_T0, self.T0)
            # print("T: ", self.log_T, self.T)
            # # print("E: ", self.log_E, self.log_E.exp())
            # print('GRAD T0: ', self.log_T0.grad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        # print('LL HISTORY', self.ll_history[-3:])
        # from matplotlib import pyplot as plt
        # plt.plot(self.ll_history)
        # plt.show()

        return self.converged

    @property
    def converged(self):
        return (len(self.ll_history) >= 2 and
                (self.ll_history[-2] - self.ll_history[-1]).abs() <
                self.epsilon)


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
