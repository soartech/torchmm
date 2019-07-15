import torch
import numpy as np
# import pandas as pd

from hmm_torch.discrete import HiddenMarkovModel
from hmm_torch.utils import generate_HMM_observation


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def test_viterbi():
    states = {0: 'Healthy', 1: 'Fever'}
    # obs = {0: 'normal', 1: 'cold', 2: 'dizzy'}

    p0 = np.array([0.6, 0.4])
    emi = np.array([[0.5, 0.1],
                    [0.4, 0.3],
                    [0.1, 0.6]])
    trans = np.array([[0.7, 0.3],
                      [0.4, 0.6]])
    model = HiddenMarkovModel(trans, emi, p0)

    obs_seq = np.array([0, 0, 1, 2, 2])
    states_seq, state_prob = model.viterbi_inference(obs_seq)

    most_likely_states = [states[s.item()] for s in states_seq]
    assert most_likely_states == ['Healthy', 'Healthy', 'Healthy', 'Fever',
                                  'Fever']


def test_forward_backward():
    states = {0: 'rain', 1: 'no_rain'}
    # obs = {0: 'umbrella', 1: 'no_umbrella'}

    p0 = np.array([0.5, 0.5])
    emi = np.array([[0.9, 0.2],
                    [0.1, 0.8]])
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]])

    model = HiddenMarkovModel(trans, emi, p0)
    obs_seq = np.array([1, 1, 0, 0, 0, 1])

    model.N = len(obs_seq)
    shape = [model.N, model.S]

    model.initialize_forw_back_variables(shape)
    obs_prob_seq = model.E[obs_seq]
    model.forward_backward(obs_prob_seq)
    posterior = model.forward * model.backward

    # marginal per timestep
    marginal = torch.sum(posterior, 1)
    # Normalize porsterior into probabilities
    posterior = posterior / marginal.view(-1, 1)

    results = [model.forward.cpu().numpy(), model.backward.cpu().numpy(),
               posterior.cpu().numpy()]
    result_list = ["Forward", "Backward", "Posterior"]

    for state_prob, path in zip(results, result_list):
        inferred_states = np.argmax(state_prob, axis=1)
        # print()
        # print(path)
        # # dptable(state_prob)
        # print(state_prob)
        # print()

    assert states[inferred_states[-1]] == 'no_rain'


def test_baum_welch():

    True_pi = np.array([0.5, 0.5])

    True_T = np.array([[0.85, 0.15],
                       [0.12, 0.88]])

    True_E = np.array([[0.8, 0.0],
                       [0.1, 0.0],
                       [0.1, 1.0]])

    obs_seq, states = generate_HMM_observation(200, True_pi, True_T, True_E)

    print("First 10 Obersvations:  ", obs_seq[:10])
    print("First 10 Hidden States: ", states[:10])

    init_pi = np.array([0.5, 0.5])

    init_T = np.array([[0.5, 0.5],
                       [0.5, 0.5]])

    init_E = np.array([[0.3, 0.2],
                       [0.3, 0.5],
                       [0.4, 0.3]])

    model = HiddenMarkovModel(init_T, init_E, init_pi,
                              epsilon=0.0001, maxStep=100)

    trans0, transition, emission, converge = model.Baum_Welch_EM(obs_seq)

    print("Transition Matrix: ")
    print(transition)
    print()
    print("Emission Matrix: ")
    print(emission)
    print()
    print("Reached Convergence: ")
    print(converge)

    state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
                              range(len(model.prob_state_1))])

    pred = (1 - state_summary[-2]) > 0.5
    print("Accuracy: ", np.mean(pred == states))
