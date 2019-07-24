import pytest
import torch
import numpy as np
# import pandas as pd
from scipy.special import logsumexp

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


def test_belief_propagation():
    n_states = 4
    n_obs = 3

    T0 = np.random.random([n_states, 1])
    T0 /= np.sum(T0)

    T = np.random.random([n_states, n_states])
    T = T / T.sum(axis=1)[:, np.newaxis]
    E = np.random.random([n_obs, n_states])
    E = E / E.sum(axis=0)[np.newaxis, :]

    # T0 = np.array([0.5, 0.5])
    # T = np.array([[1, 0], [0, 1]])
    # E = np.array([[1, 0], [0, 1]])

    model = HiddenMarkovModel(T, E, T0)

    # compute max path from each node to each node
    # will give a matrix, row is the source col is the dest
    res = np.array([np.log(T0[i] * T[i, :]) for i in range(len(T0))])

    T0_t = torch.tensor(T0)
    log_T0 = torch.log(T0_t)
    max_b, _ = model._belief_prop_max(log_T0)
    assert np.allclose(res.max(0), max_b.data.numpy())

    sum_b = model._belief_prop_sum(log_T0)
    assert np.allclose(logsumexp(res, 0), sum_b.data.numpy())


def test_decode():
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
    states_seq, states_ll = model.decode(obs_seq)

    most_likely_states = [states[s.item()] for s in states_seq]
    assert most_likely_states == ['Healthy', 'Healthy', 'Healthy', 'Fever',
                                  'Fever']


def test_decode_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The state sequence comes from figure 15.5(b) on page 577 of the third
    edition. The correct values were manually compared to the normalized values
    from the figure 15.5(b).
    """

    states = {0: 'No Rain', 1: 'Rain'}
    # obs = {0: 'No Umbrella', 1: 'Umbrella'}

    p0 = np.array([0.5, 0.5])
    emi = np.array([[0.8, 0.1],
                    [0.2, 0.9]])
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
    model = HiddenMarkovModel(trans, emi, p0)

    obs_seq = np.array([1, 1, 0, 1, 1])
    states_seq, path_ll = model.decode(obs_seq)

    most_likely_states = [states[s.item()] for s in states_seq]
    assert most_likely_states == ['Rain', 'Rain', 'No Rain', 'Rain', 'Rain']

    probs = torch.exp(path_ll).data.numpy()
    normalized = probs / probs.sum(axis=1)[:, np.newaxis]

    correct = np.array([[0.18181818, 0.81818182],
                        [0.08695652, 0.91304348],
                        [0.77419355, 0.22580645],
                        [0.34146341, 0.65853659],
                        [0.10332103, 0.89667897]])

    assert np.allclose(normalized, correct)


def test_filter_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The example comes from page 572, filtering and prediction section. The
    correct values were manually compared to the normalized values from this
    example.
    """
    # states = {0: 'No Rain', 1: 'Rain'}
    # obs = {0: 'No Umbrella', 1: 'Umbrella'}

    p0 = np.array([0.5, 0.5])
    emi = np.array([[0.8, 0.1],
                    [0.2, 0.9]])
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
    model = HiddenMarkovModel(trans, emi, p0)

    obs_seq = np.array([1])
    posterior = model.filter(obs_seq)
    probs = torch.exp(posterior).data.numpy()
    normalized = probs / probs.sum(axis=0)
    correct = np.array([0.18181818, 0.81818182])
    assert np.allclose(normalized, correct)

    obs_seq = np.array([1, 1])
    posterior = model.filter(obs_seq)
    probs = torch.exp(posterior).data.numpy()
    normalized = probs / probs.sum(axis=0)
    correct = np.array([0.11664296, 0.88335704])
    assert np.allclose(normalized, correct)


def test_score_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The example comes from page 572, filtering and prediction section. The
    correct values were manually computed by summing the filtered posterior.
    """
    # states = {0: 'No Rain', 1: 'Rain'}
    # obs = {0: 'No Umbrella', 1: 'Umbrella'}

    p0 = np.array([0.5, 0.5])
    emi = np.array([[0.8, 0.1],
                    [0.2, 0.9]])
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
    model = HiddenMarkovModel(trans, emi, p0)

    obs_seq = np.array([1])
    ll_score = model.score(obs_seq)
    print(ll_score)
    assert ll_score - -3.101092789211817 < 0.001

    obs_seq = np.array([1, 1])
    ll_score = model.score(obs_seq)
    assert ll_score - -3.101092789211817 < 0.001


def test_predict_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The example comes from page 572, filtering and prediction section. The
    correct values were manually compared propagated one step into future from
    the filtering example.
    """
    # states = {0: 'No Rain', 1: 'Rain'}
    # obs = {0: 'No Umbrella', 1: 'Umbrella'}

    p0 = np.array([0.5, 0.5])
    emi = np.array([[0.8, 0.1],
                    [0.2, 0.9]])
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
    model = HiddenMarkovModel(trans, emi, p0)

    obs_seq = np.array([1])
    posterior = model.predict(obs_seq)
    probs = torch.exp(posterior).data.numpy()
    normalized = probs / probs.sum(axis=0)
    correct = np.array([0.37272727, 0.62727273])
    assert np.allclose(normalized, correct)

    obs_seq = np.array([1, 1])
    posterior = model.predict(obs_seq)
    probs = torch.exp(posterior).data.numpy()
    normalized = probs / probs.sum(axis=0)
    correct = np.array([0.34665718, 0.65334282])
    assert np.allclose(normalized, correct)


def test_smooth():

    p0 = np.array([0.5, 0.5])
    emi = np.array([[0.8, 0.1],
                    [0.2, 0.9]])
    trans = np.array([[0.7, 0.3],
                      [0.3, 0.7]])

    model = HiddenMarkovModel(trans, emi, p0)
    obs_seq = np.array([1, 1])

    # New approach
    posterior_ll = model.smooth(obs_seq)
    posterior_prob = torch.exp(posterior_ll)
    m = torch.sum(posterior_prob, 1)
    posterior_prob = posterior_prob / m.view(-1, 1)

    first_correct = np.array([[0.11664296, 0.88335704]])
    assert np.allclose(posterior_prob.data.numpy()[0], first_correct)


def test_baum_welch():

    True_pi = np.array([0.5, 0.5])

    True_T = np.array([[0.85, 0.15],
                       [0.12, 0.88]])

    True_E = np.array([[0.95, 0.05],
                       [0.05, 0.95]])

    true_model = HiddenMarkovModel(True_T, True_E, True_pi)
    obs_seq, states = true_model.sample(500)

    print("First 10 Obersvations:  ", obs_seq[:10])
    print("First 10 Hidden States: ", states[:10])

    init_pi = np.array([0.5, 0.5])

    init_T = np.array([[0.6, 0.4],
                       [0.3, 0.7]])

    init_E = np.array([[0.6, 0.3],
                       [0.4, 0.7]])

    model = HiddenMarkovModel(init_T, init_E, init_pi,
                              epsilon=0.1, maxStep=100)

    trans0, transition, emission, converge = model.fit(obs_seq,
                                                       alg="baum_welch")

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_pi)
    print("Pi Matrix: ")
    print(trans0.exp())

    print("Transition Matrix: ")
    print(transition.exp())
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission Matrix: ")
    print(emission.exp())
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

    assert converge

    states_seq, path_ll = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    pred = states_seq.data.numpy()
    print(pred)
    print(states)
    accuracy = np.mean(pred == states)
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1


def test_viterbi_training():

    True_pi = np.array([0.5, 0.5])

    True_T = np.array([[0.85, 0.15],
                       [0.12, 0.88]])

    True_E = np.array([[0.95, 0.05],
                       [0.05, 0.95]])

    true_model = HiddenMarkovModel(True_T, True_E, True_pi)
    obs_seq, states = true_model.sample(500)

    print("First 10 Obersvations:  ", obs_seq[:10])
    print("First 10 Hidden States: ", states[:10])

    init_pi = np.array([0.5, 0.5])

    init_T = np.array([[0.6, 0.4],
                       [0.3, 0.7]])

    init_E = np.array([[0.6, 0.3],
                       [0.4, 0.7]])

    model = HiddenMarkovModel(init_T, init_E, init_pi,
                              epsilon=0.1, maxStep=10)

    trans0, transition, emission, converge = model.fit(obs_seq, alg="viterbi")

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_pi)
    print("Pi Matrix: ")
    print(trans0.exp())

    print("Transition Matrix: ")
    print(transition.exp())
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission Matrix: ")
    print(emission.exp())
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

    assert converge

    states_seq, path_ll = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    pred = states_seq.data.numpy()
    print(pred)
    print(states)
    accuracy = np.mean(pred == states)
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1
