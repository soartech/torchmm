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
        self.E = torch.tensor(E)
        self.log_E = torch.log(self.E)

        # Transition matrix
        self.T = torch.tensor(T)
        self.log_T = torch.log(self.T)

        # Initial state vector
        self.T0 = torch.tensor(T0)
        self.log_T0 = torch.log(self.T0)

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
        states[0] = drawFrom(self.T0)
        obs[0] = drawFrom(self.E[:, int(states[0])])
        for t in range(1, num_obs):
            states[t] = drawFrom(self.T[int(states[t-1]), :])
            obs[t] = drawFrom(self.E[:, int(states[t])])
        return np.int64(obs), states

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

        Returns the most likely state numbers and log likelihood of each state.
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
        self.forward = torch.zeros(shape, dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward)
        self.obs_ll_full = self.emission_ll(x)

        # Forward
        self.forward[0] = self.log_T0 + self.obs_ll_full[0]
        for step, obs_ll in enumerate(self.obs_ll_full[1:]):
            self.forward[step+1] = (self.belief_sum(self.forward[step, :]) +
                                    obs_ll)

        # Backward
        self.backward[0] = torch.ones([self.S], dtype=torch.float64)
        for step, obs_prob in enumerate(self.obs_ll_full.flip([0, 1])[:-1]):
            self.backward[step+1] = self.belief_sum(self.backward[step, :] +
                                                    obs_prob)
        self.backward = self.backward.flip([0, 1])
        self.posterior = self.forward + self.backward

        # normalizing constant, assuming forward and backwards have been
        # normalized?
        # - self.forward.sum(0).view(-1, 1))

        # Return posterior
        return self.posterior

    def reestimate_trans_ll(self):
        M = 0
        numerators = self.forward + self.log_T + M + self.backward[1:]
        self.log_T = numerators.sum(0) - self.posterior.sum(0)

    def initialize_forw_back_variables(self, shape):
        self.forward = torch.zeros(shape, dtype=torch.float64)
        self.backward = torch.zeros_like(self.forward)

    def _forward(model, obs_prob_seq):
        model.scale = torch.zeros(
            [model.N], dtype=torch.float64)  # scale factors
        # initialize with state starting priors
        init_prob = model.T0 * obs_prob_seq[0]
        # scaling factor at t=0
        model.scale[0] = 1.0 / init_prob.sum()
        # scaled belief at t=0
        model.forward[0] = model.scale[0] * init_prob
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq[1:]):
            # previous state probability
            prev_prob = model.forward[step].unsqueeze(0)
            # transition prior
            prior_prob = torch.matmul(prev_prob, model.T)
            # forward belief propagation
            forward_score = prior_prob * obs_prob
            forward_prob = torch.squeeze(forward_score)
            # scaling factor
            model.scale[step + 1] = 1 / forward_prob.sum()
            # Update forward matrix
            model.forward[step + 1] = model.scale[step + 1] * forward_prob

    def _backward(self, obs_prob_seq_rev):
        # initialize with state ending priors
        self.backward[0] = self.scale[self.N - 1] * \
            torch.ones([self.S], dtype=torch.float64)
        # propagate belief
        for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
            # next state probability
            next_prob = self.backward[step, :].unsqueeze(1)
            # observation emission probabilities
            obs_prob_d = torch.diag(obs_prob)
            # transition prior
            prior_prob = torch.matmul(self.T, obs_prob_d)
            # backward belief propagation
            backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
            # Update backward matrix
            self.backward[step + 1] = self.scale[self.N -
                                                 2 - step] * backward_prob
        self.backward = torch.flip(self.backward, [0, 1])

    def forward_backward(self, obs_prob_seq):
        """
        Runs forward backward algorithm on observation sequence

        Arguments
        - obs_prob_seq : matrix of size N by S, where N is number of timesteps
        and S is the number of states

        Returns
        - forward : matrix of size N by S representing the forward probability
        of each state at each time step
        - backward : matrix of size N by S representing the backward
        probability of each state at each time step
        - posterior : matrix of size N by S representing the posterior
        probability of each state at each time step
        """
        self._forward(obs_prob_seq)
        obs_prob_seq_rev = torch.flip(obs_prob_seq, [0, 1])
        self._backward(obs_prob_seq_rev)

    def re_estimate_transition(self, x):
        self.M = torch.zeros([self.N - 1, self.S, self.S], dtype=torch.float64)

        for t in range(self.N - 1):
            tmp_0 = torch.matmul(self.forward[t].unsqueeze(0), self.T)
            tmp_1 = tmp_0 * self.E[x[t + 1]].unsqueeze(0)
            denom = torch.matmul(
                tmp_1, self.backward[t + 1].unsqueeze(1)).squeeze()

            trans_re_estimate = torch.zeros(
                [self.S, self.S], dtype=torch.float64)

            for i in range(self.S):
                numer = self.forward[t, i] * self.T[i, :] * \
                    self.E[x[t+1]] * self.backward[t+1]
                trans_re_estimate[i] = numer / denom

            self.M[t] = trans_re_estimate

        self.gamma = self.M.sum(2).squeeze()
        T_new = self.M.sum(0) / self.gamma.sum(0).unsqueeze(1)

        T0_new = self.gamma[0, :]

        prod = (self.forward[self.N-1] * self.backward[self.N-1]).unsqueeze(0)
        s = prod / prod.sum()
        self.gamma = torch.cat([self.gamma, s], 0)
        self.prob_state_1.append(self.gamma[:, 0])
        return T0_new, T_new

    def re_estimate_emission(self, x):
        states_marginal = self.gamma.sum(0)
        # One hot encoding buffer that you create out of the loop and just keep
        # reusing
        seq_one_hot = torch.zeros([len(x), self.Obs], dtype=torch.float64)
        seq_one_hot.scatter_(1, torch.tensor(x).unsqueeze(1), 1)
        emission_score = torch.matmul(seq_one_hot.transpose_(1, 0), self.gamma)
        return emission_score / states_marginal

    def re_estimate_transition_ll(self, x):
        """
        Great reference: https://web.stanford.edu/~jurafsky/slp3/A.pdf
        """
        obs_ll_full = self.emission_ll(x)

        # Compute T0, using normalized forward.
        log_T0_new = (self.forward_ll[0] -
                      torch.logsumexp(self.forward_ll[0], 0))
        T0_new = torch.exp(log_T0_new)

        # Compute expected transitions
        M_ll = (self.forward_ll[:-1].expand(2, -1, -1).permute(1, 2, 0) +
                self.log_T.expand(self.N-1, -1, -1) +
                obs_ll_full[1:].expand(2, -1, -1).permute(1, 2, 0) +
                self.backward_ll[1:].expand(2, -1, -1).permute(1, 2, 0))

        denom = self.posterior_ll[:-1].logsumexp(1)
        M_ll -= denom.unsqueeze(1).unsqueeze(1).expand(-1, self.S, self.S)

        # Estimate new transition matrix
        log_T_new = (torch.logsumexp(M_ll, 0) -
                     torch.logsumexp(torch.logsumexp(M_ll, 2), 0))
        T_new = torch.exp(log_T_new)

        m = torch.logsumexp(self.posterior_ll, 1)
        gamma_ll = self.posterior_ll - m.view(-1, 1)
        gamma = torch.exp(gamma_ll)

        prod = (self.forward[self.N-1] * self.backward[self.N-1]).unsqueeze(0)
        s = prod / prod.sum()
        self.gamma = torch.cat([gamma, s], 0)
        self.prob_state_1.append(gamma[:, 0])

        return T0_new, T_new

    def re_estimate_emission_ll(self, x):
        # posterior_ll = torch.log(self.posterior)
        posterior_ll = self.posterior_ll

        m = torch.logsumexp(posterior_ll, 1)
        gamma_ll = posterior_ll - m.view(-1, 1)
        gamma = torch.exp(gamma_ll)
        states_marginal = gamma.sum(0)

        # One hot encoding buffer that you create out of the loop and just keep
        # reusing
        seq_one_hot = torch.zeros([len(x), self.Obs], dtype=torch.float64)
        seq_one_hot.scatter_(1, torch.tensor(x).unsqueeze(1), 1)
        emission_score = torch.matmul(seq_one_hot.transpose_(1, 0), gamma)

        return emission_score / states_marginal

    def check_convergence(self, new_T0, new_transition, new_emission):

        delta_T0 = torch.max(torch.abs(self.T0 - new_T0)).item() < self.epsilon
        delta_T = torch.max(
            torch.abs(self.T - new_transition)).item() < self.epsilon
        delta_E = torch.max(torch.abs(self.E - new_emission)
                            ).item() < self.epsilon

        return delta_T0 and delta_T and delta_E

    def expectation_maximization_step(self, obs_seq):
        # probability of emission sequence
        # obs_prob_seq = self.E[obs_seq]
        # self.forward_backward(obs_prob_seq)

        self.forward_backward_inference(obs_seq)
        self.forward_ll = self.forward
        self.backward_ll = self.backward
        self.posterior_ll = self.posterior

        self.forward = torch.exp(self.forward)
        self.backward = torch.exp(self.backward)
        self.posterior = torch.exp(self.posterior)

        new_T0, new_transition = self.re_estimate_transition_ll(obs_seq)
        new_T01, new_transition = self.re_estimate_transition(obs_seq)

        print('transition')
        print(new_T0)
        print(new_T01)
        # assert np.allclose(new_T01, new_T0)

        new_emission = self.re_estimate_emission_ll(obs_seq)
        new_emission1 = self.re_estimate_emission(obs_seq)

        print('emission')
        print(new_emission)
        print(new_emission1)
        # assert np.allclose(new_emission, new_emission1)

        converged = self.check_convergence(
            new_T0, new_transition, new_emission)

        self.T0 = new_T0
        self.log_T0 = torch.log(self.T0)
        self.E = new_emission
        self.log_E = torch.log(self.E)
        self.T = new_transition
        self.log_T = torch.log(self.T)

        return converged

    def Baum_Welch_EM(self, obs_seq):
        # length of observed sequence
        self.N = len(obs_seq)

        # shape of Variables
        shape = [self.N, self.S]

        # initialize variables
        self.initialize_forw_back_variables(shape)

        converged = False

        for i in range(self.maxStep):
            converged = self.expectation_maximization_step(obs_seq)
            if converged:
                print('converged at step {}'.format(i))
                break
        return self.T0, self.T, self.E, converged
