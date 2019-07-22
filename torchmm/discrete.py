import numpy as np
import torch


class HiddenMarkovModel(object):
    """
    Hidden Markov self Class

    Parameters:
    -----------
    - T: numpy.array Transition matrix of size S by S stores probability from
    state i to state j.
    - E: numpy.array Emission matrix of size N (number of observations) by S
    (number of states) stores the probability of observing  O from each state
    - T0: numpy.array Initial state probabilities of size S.
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

        self.prob_state_1 = []

        # Emission probability
        # self.E = torch.tensor(E)
        self.log_E = torch.log(torch.tensor(E))

        # Transition matrix
        # self.T = torch.tensor(T)
        self.log_T = torch.log(torch.tensor(T))

        # Initial state vector
        # self.T0 = torch.tensor(T0)
        self.log_T0 = torch.log(torch.tensor(T0))

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

    def sample(self, num_obs):
        """
        Draws a sample from the HMM of length num_obs.

        TODO:
            - refactor to use arbitrary emission PDF
        """
        def drawFrom(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        obs = np.zeros(num_obs)
        states = np.zeros(num_obs)
        states[0] = drawFrom(torch.exp(self.log_T0))
        obs[0] = drawFrom(torch.exp(self.log_E[:, int(states[0])]))
        for t in range(1, num_obs):
            states[t] = drawFrom(torch.exp(self.log_T[int(states[t-1]), :]))
            obs[t] = self.sample_emission(int(states[t]))
            # obs[t] = drawFrom(self.E[:, int(states[t])])
        return np.int64(obs), states

    def sample_emission(self, s):
        """
        Given a state, randomly samples an emission.
        """
        probs = torch.exp(self.log_E[:, s])
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    def belief_max(self, scores):
        """
        Propagates the scores over transition matrix. Returns the indices and
        values of the max for each state.
        """
        return torch.max(scores.view(-1, 1) + self.log_T, 0)

    def belief_sum(self, scores):
        return torch.logsumexp(scores.view(-1, 1) + self.log_T, 0)

    def emission_ll(self, x):
        """
        Log likelihood of emissions for each state.

        Returns a tensor of shape (num samples in x, num states).
        """
        return self.log_E[x]

    def viterbi_inference(self, x):
        """
        Compute the most likely states given the current model and a sequence
        of observations (x).

        Returns the most likely state numbers and path score for each state.
        """
        N = len(x)
        shape = [N, self.S]

        # Init_viterbi_variables
        path_states = torch.zeros(shape, dtype=torch.float64)
        path_scores = torch.zeros_like(path_states)
        states_seq = torch.zeros([shape[0]], dtype=torch.int64)

        # log probability of emission sequence
        obs_ll_full = self.emission_ll(x)

        # initialize with state starting log-priors
        path_scores[0] = self.log_T0 + obs_ll_full[0]

        for step, obs_prob in enumerate(obs_ll_full[1:]):
            # propagate state belief
            max_vals, max_indices = self.belief_max(path_scores[step, :])

            # the inferred state by maximizing global function
            path_states[step + 1] = max_indices

            # and update state and score matrices
            path_scores[step + 1] = max_vals + obs_prob

        # infer most likely last state
        states_seq[N-1] = torch.argmax(path_scores[N-1, :], 0)
        for step in range(N-1, 0, -1):
            # for every timestep retrieve inferred state
            state = states_seq[step]
            state_prob = path_states[step][state]
            states_seq[step - 1] = state_prob

        return states_seq, path_scores

    def forward_backward_inference(self, x):
        """
        Computes the expected probability of each state for each time.
        Returns the posterior over all states.
        """
        shape = [len(x), self.S]

        # Init
        self.forward_ll = torch.zeros(shape, dtype=torch.float64)
        self.backward_ll = torch.zeros_like(self.forward_ll)
        self.obs_ll_full = self.emission_ll(x)

        # Forward
        self.forward_ll[0] = self.log_T0 + self.obs_ll_full[0]
        for step, obs_ll in enumerate(self.obs_ll_full[1:]):
            self.forward_ll[step+1] = (
                self.belief_sum(self.forward_ll[step, :]) + obs_ll)

        # Backward
        self.backward_ll[0] = torch.ones([self.S], dtype=torch.float64)
        for step, obs_prob in enumerate(self.obs_ll_full.flip([0, 1])[:-1]):
            self.backward_ll[step+1] = self.belief_sum(
                self.backward_ll[step, :] + obs_prob)
        self.backward_ll = self.backward_ll.flip([0, 1])
        self.posterior_ll = self.forward_ll + self.backward_ll

        # Return posterior
        return self.posterior_ll

    def initialize_forw_back_variables(self, shape):
        self.forward_ll = torch.zeros(shape, dtype=torch.float64)
        self.backward_ll = torch.zeros_like(self.forward_ll)
        self.posterior_ll = torch.zeros_like(self.forward_ll)

    def re_estimate_transition_ll(self, x):
        """
        Updates the transmission probabilities, given a sequence.

        Assumes the forward backward step has already been run and
        self.forward_ll, self.backward_ll, and self.posterior_ll are set.

        Great reference: https://web.stanford.edu/~jurafsky/slp3/A.pdf
        """
        N = len(x)
        obs_ll_full = self.emission_ll(x)

        # Compute T0, using normalized forward.
        log_T0_new = (self.forward_ll[0] -
                      torch.logsumexp(self.forward_ll[0], 0))

        # Compute expected transitions
        M_ll = (self.forward_ll[:-1].unsqueeze(2).expand(-1, -1, 2) +
                self.log_T.expand(N-1, -1, -1) +
                obs_ll_full[1:].unsqueeze(2).expand(-1, -1, 2) +
                self.backward_ll[1:].unsqueeze(2).expand(-1, -1, 2))

        denom = self.posterior_ll[:-1].logsumexp(1)
        M_ll -= denom.unsqueeze(1).unsqueeze(2).expand(-1, self.S, self.S)

        # Estimate new transition matrix
        print(torch.logsumexp(M_ll, 0).shape)
        print(torch.logsumexp(M_ll, (0, 2)).shape)

        log_T_new = (torch.logsumexp(M_ll, 0) -
                     torch.logsumexp(M_ll, (0, 2)).view(-1, 1))

        print(torch.exp(log_T0_new))
        print(torch.exp(log_T_new))

        return log_T0_new, log_T_new

    def re_estimate_emission_ll(self, x):
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
        seq_one_hot = torch.zeros([len(x), self.Obs], dtype=torch.float64)
        seq_one_hot.scatter_(1, torch.tensor(x).unsqueeze(1), 1)
        emission_score = torch.matmul(seq_one_hot.transpose_(1, 0), gamma)

        new_E = torch.log(emission_score / states_marginal)
        print(new_E.exp())
        return new_E

    @property
    def converged(self):
        return (len(self.ll_history) == 2 and self.ll_history[-2] -
                self.ll_history[-1] < self.epsilon)

    # def check_convergence(self, new_log_T0, new_log_transition,
    #                       new_log_emission):
    #     delta_T0 = (torch.max(torch.abs(self.log_T0 - new_log_T0)).item() <
    #                 self.epsilon)
    #     delta_T = (torch.max(torch.abs(self.log_T -
    #     new_log_transition)).item() < self.epsilon)
    #     delta_E = (torch.max(torch.abs(self.log_E - new_log_emission)).item()
    #                < self.epsilon)
    #     print(delta_T0, delta_T, delta_E)
    #     return delta_T0 and delta_T and delta_E

    def expectation_maximization_step(self, obs_seq):

        self.forward_backward_inference(obs_seq)
        self.ll_history.append(self.forward_ll[-1].sum())
        print('LL: ', self.ll_history[-1])

        log_T0_new, log_T_new = self.re_estimate_transition_ll(obs_seq)
        log_E_new = self.re_estimate_emission_ll(obs_seq)

        # converged = self.check_convergence(
        #     log_T0_new, log_T_new, log_E_new)

        self.log_T0 = log_T0_new
        self.log_E = log_E_new
        self.log_T = log_T_new

        return self.converged

    def Baum_Welch_EM(self, obs_seq):
        # length of observed sequence
        N = len(obs_seq)

        # shape of Variables
        shape = [N, self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)
        self.ll_history = []

        converged = False

        for i in range(self.maxStep):
            converged = self.expectation_maximization_step(obs_seq)
            if converged:
                print('converged at step {}'.format(i))
                break
        return self.log_T0, self.log_T, self.log_E, converged
