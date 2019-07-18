import pytest
import torch
import numpy as np
# import pandas as pd

from torchmm.discrete import HiddenMarkovModel


def test_init():
    good_pi = np.array([1.0, 0.0])
    good_T = np.array([[1.0, 0.0], [0.0, 1.0]])
    good_E = np.array([[1.0, 0.0], [0.0, 1.0]])

    bad1_pi = np.array([1.0, 1.0])
    bad1_T = np.array([[1.0, 1.0], [1.0, 1.0]])
    bad1_E = np.array([[1.0, 1.0], [1.0, 1.0]])

    bad2_pi = np.array([1.0])
    bad2_T = np.array([[1.0], [1.0]])
    bad2_E = np.array([[1.0], [1.0]])

    HiddenMarkovModel(good_T, good_E, good_pi)

    with pytest.raises(ValueError):
        HiddenMarkovModel(bad1_T, good_E, good_pi)
    with pytest.raises(ValueError):
        HiddenMarkovModel(good_T, bad1_E, good_pi)
    with pytest.raises(ValueError):
        HiddenMarkovModel(good_T, good_E, bad1_pi)

    with pytest.raises(ValueError):
        HiddenMarkovModel(bad2_T, good_E, good_pi)
    with pytest.raises(ValueError):
        HiddenMarkovModel(good_T, bad2_E, good_pi)
    with pytest.raises(ValueError):
        HiddenMarkovModel(good_T, good_E, bad2_pi)

    with pytest.raises(ValueError):
        HiddenMarkovModel(good_T, good_E, good_pi, epsilon=0)
    with pytest.raises(ValueError):
        HiddenMarkovModel(good_T, good_E, good_pi, maxStep=0)


def test_sample():

    True_pi = np.array([1.0, 0.0])

    True_T = np.array([[1.0, 0.0],
                       [0.0, 1.0]])

    True_E = np.array([[1.0, 0.0],
                       [0.0, 1.0]])

    true_model = HiddenMarkovModel(True_T, True_E, True_pi)
    obs_seq, states = true_model.sample(10)

    assert len(obs_seq) == 10
    assert len(states) == 10
    assert 1 not in obs_seq
    assert 1 not in states

    True_pi = np.array([0.5, 0.5])

    True_T = np.array([[0.5, 0.5],
                       [0.5, 0.5]])

    True_E = np.array([[1.0, 0.0],
                       [0.0, 1.0]])

    true_model = HiddenMarkovModel(True_T, True_E, True_pi)
    obs_seq, states = true_model.sample(20)

    assert len(obs_seq) == 20
    assert len(states) == 20
    assert 1 in obs_seq and 0 in obs_seq
    assert 1 in states and 0 in states

    True_pi = np.array([0.5, 0.5])

    True_T = np.array([[0.9, 0.1],
                       [0.5, 0.5]])

    True_E = np.array([[1.0, 0.0],
                       [0.0, 1.0]])

    true_model = HiddenMarkovModel(True_T, True_E, True_pi)
    obs_seq, states = true_model.sample(20)

    assert len(obs_seq) == 20
    assert len(states) == 20
    assert (states == 0).sum() > (states == 1).sum()


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
    states_seq, states_ll = model.viterbi_inference(obs_seq)

    print(states_seq)
    print(states_ll)
    probs = torch.exp(states_ll).data.numpy()
    print(np.sum(probs, 0))
    print(np.sum(probs, 1))
    # assert False

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

    inferred_states = [states[s] for s in inferred_states]
    print(inferred_states)
    assert inferred_states == ['no_rain', 'no_rain', 'rain', 'rain', 'rain',
                               'no_rain']


def test_baum_welch():

    True_pi = np.array([0.5, 0.5])

    True_T = np.array([[0.85, 0.15],
                       [0.12, 0.88]])

    True_E = np.array([[0.95, 0.05],
                       [0.05, 0.95]])

    true_model = HiddenMarkovModel(True_T, True_E, True_pi)
    obs_seq, states = true_model.sample(200)

    print("First 10 Obersvations:  ", obs_seq[:100])
    print("First 10 Hidden States: ", states[:10])

    init_pi = np.array([0.5, 0.5])

    init_T = np.array([[0.5, 0.5],
                       [0.5, 0.5]])

    init_E = np.array([[0.6, 0.3],
                       [0.4, 0.7]])

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
    accuracy = np.mean(pred == states)
    print("Accuracy: ", np.mean(pred == states))
    assert accuracy > 0.9 or accuracy < 0.1
