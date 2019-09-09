import pytest
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence

from torchmm.base import CategoricalModel
from torchmm.base import DiagNormalModel
from torchmm.hmm_packed import HiddenMarkovModel
from torchmm.utils import pack_list


def test_init():
    good_T0 = torch.tensor([1.0, 0.0])
    good_T = torch.tensor([[1.0, 0.0],
                           [0.0, 1.0]])
    good_E1 = CategoricalModel(probs=torch.tensor([1.0, 0.0]))
    good_E2 = CategoricalModel(probs=torch.tensor([0.0, 1.0]))

    bad1_T0 = torch.tensor([1.0, 1.0])
    bad1_T = torch.tensor([[1.0, 1.0],
                           [1.0, 1.0]])
    bad_E1 = torch.tensor([1.0, 0.0])

    bad2_T0 = torch.tensor([1.0])
    bad2_T = torch.tensor([[1.0], [1.0]])

    HiddenMarkovModel([good_E1, good_E2], T0=good_T0, T=good_T)

    with pytest.raises(ValueError):
        HiddenMarkovModel([bad_E1, good_E2], T0=bad1_T0, T=good_T)

    with pytest.raises(ValueError):
        HiddenMarkovModel([good_E1, good_E2], T0=bad1_T0, T=good_T)
    with pytest.raises(ValueError):
        HiddenMarkovModel([good_E1, good_E2], T0=good_T0, T=bad1_T)

    with pytest.raises(ValueError):
        HiddenMarkovModel([good_E1, good_E2], T0=bad2_T0, T=good_T)
    with pytest.raises(ValueError):
        HiddenMarkovModel([good_E1, good_E2], T0=good_T0, T=bad2_T)


def test_hmm_sample():

    # create categorical models for the states
    T0 = torch.tensor([1.0, 0.0])
    T = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])
    s1 = CategoricalModel(probs=torch.tensor([1.0, 0.0]))
    s2 = CategoricalModel(probs=torch.tensor([0.0, 1.0]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(3, 10)

    assert isinstance(obs_seq, PackedSequence)
    assert obs_seq.data.shape[0] == 30

    assert isinstance(states, PackedSequence)
    assert states.data.shape[0] == 30
    assert 1 not in obs_seq.data
    assert 1 not in states.data

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.5, 0.5],
                      [0.5, 0.5]])
    s1 = CategoricalModel(probs=torch.tensor([1.0, 0.0]))
    s2 = CategoricalModel(probs=torch.tensor([0.0, 1.0]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(4, 20)

    assert isinstance(obs_seq, PackedSequence)
    assert obs_seq.data.shape[0] == 80

    assert isinstance(states, PackedSequence)
    assert states.data.shape[0] == 80
    assert 1 in obs_seq.data and 0 in obs_seq.data
    assert 1 in states.data and 0 in states.data

    T0 = torch.tensor([0.9, 0.1])
    T = torch.tensor([[0.9, 0.1],
                      [0.5, 0.5]])
    s1 = CategoricalModel(probs=torch.tensor([1.0, 0.0]))
    s2 = CategoricalModel(probs=torch.tensor([0.0, 1.0]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(1, 20)

    assert isinstance(obs_seq, PackedSequence)
    assert obs_seq.data.shape[0] == 20

    assert isinstance(states, PackedSequence)
    assert states.data.shape[0] == 20
    assert (states.data == 0).sum() > (states.data == 1).sum()


def test_belief_propagation():
    n_states = 4
    n_obs = 3

    T0 = torch.rand([n_states]).softmax(0)
    T = torch.rand([n_states, n_states]).softmax(1)

    states = [CategoricalModel(probs=torch.rand([n_obs]).softmax(0)) for i in
              range(n_states)]

    model = HiddenMarkovModel(states, T0=T0, T=T)
    model.update_log_params()

    # compute max path from each node to each node
    # will give a matrix, row is the source col is the dest
    # res = torch.log(T0 * T)
    res = torch.stack([torch.log(T0[i] * T[i, :]) for i in range(len(T0))])

    log_T0 = T0.log()
    max_b, _ = model._belief_prop_max(log_T0.unsqueeze(0))
    assert torch.allclose(res.max(0).values, max_b[0])

    sum_b = model._belief_prop_sum(log_T0.unsqueeze(0))
    assert torch.allclose(res.logsumexp(0), sum_b[0])


def test_hmm_decode():
    states = {0: 'Healthy', 1: 'Fever'}
    # obs = {0: 'normal', 1: 'cold', 2: 'dizzy'}

    T0 = torch.tensor([0.6, 0.4])
    T = torch.tensor([[0.7, 0.3],
                      [0.4, 0.6]])
    s1 = CategoricalModel(probs=torch.tensor([0.5, 0.4, 0.1]))
    s2 = CategoricalModel(probs=torch.tensor([0.1, 0.3, 0.6]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    obs_seq = pack_list([torch.tensor([1, 0, 1, 2, 2])])
    states_seq, _ = model.decode(obs_seq)
    ss_unpacked, _ = pad_packed_sequence(states_seq, batch_first=True)

    most_likely_states = [states[s.item()] for s in ss_unpacked[0]]
    assert most_likely_states == ['Healthy', 'Healthy', 'Healthy', 'Fever',
                                  'Fever']


def test_hmm_decode_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The state sequence comes from figure 15.5(b) on page 577 of the third
    edition. The correct values were manually compared to the normalized values
    from the figure 15.5(b).
    """

    states = {0: 'No Rain', 1: 'Rain'}
    # obs = {0: 'No Umbrella', 1: 'Umbrella'}

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.7, 0.3],
                      [0.3, 0.7]])
    s1 = CategoricalModel(probs=torch.tensor([0.8, 0.2]))
    s2 = CategoricalModel(probs=torch.tensor([0.1, 0.9]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    obs_seq = pack_list([torch.tensor([1, 1, 0, 1, 1])])
    states_seq, path_ll = model.decode(obs_seq)
    ss_unpacked, _ = pad_packed_sequence(states_seq, batch_first=True)
    path_unpacked, _ = pad_packed_sequence(path_ll, batch_first=True)

    most_likely_states = [states[s.item()] for s in ss_unpacked[0]]
    assert most_likely_states == ['Rain', 'Rain', 'No Rain', 'Rain', 'Rain']

    normalized = path_unpacked[0].softmax(1)

    correct = torch.tensor([[0.18181818, 0.81818182],
                            [0.08695652, 0.91304348],
                            [0.77419355, 0.22580645],
                            [0.34146341, 0.65853659],
                            [0.10332103, 0.89667897]])

    assert torch.allclose(normalized, correct)


def test_hmm_filter_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The example comes from page 572, filtering and prediction section. The
    correct values were manually compared to the normalized values from this
    example.
    """
    # states = {0: 'No Rain', 1: 'Rain'}
    # obs = {0: 'No Umbrella', 1: 'Umbrella'}
    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.7, 0.3],
                      [0.3, 0.7]])
    s1 = CategoricalModel(probs=torch.tensor([0.8, 0.2]))
    s2 = CategoricalModel(probs=torch.tensor([0.1, 0.9]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    obs_seq = pack_list([torch.tensor([1])])
    posterior = model.filter(obs_seq)
    normalized = posterior.softmax(1)
    correct = torch.tensor([[0.18181818, 0.81818182]])
    assert torch.allclose(normalized, correct)

    obs_seq = pack_list([torch.tensor([1, 1])])
    posterior = model.filter(obs_seq)
    normalized = posterior.softmax(1)
    correct = torch.tensor([[0.11664296, 0.88335704]])
    assert torch.allclose(normalized, correct)


def test_log_prob_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The example comes from page 572, filtering and prediction section. The
    correct values were manually computed by summing the filtered posterior.
    """
    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.7, 0.3],
                      [0.3, 0.7]])
    s1 = CategoricalModel(probs=torch.tensor([0.8, 0.2]))
    s2 = CategoricalModel(probs=torch.tensor([0.1, 0.9]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    obs_seq = pack_list([torch.tensor([1])])
    ll_score = model.log_prob(obs_seq).item()
    assert abs(ll_score - -0.5978) < 0.001

    obs_seq = pack_list([torch.tensor([1, 1])])
    ll_score = model.log_prob(obs_seq).item()
    assert abs(ll_score - -1.045545) < 0.001

    obs_seq = pack_list([torch.tensor([1]),
                         torch.tensor([1])])
    ll_score = model.log_prob(obs_seq)
    assert abs(ll_score - (2 * -0.5978)) < 0.001


def test_hmm_parameters():
    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.7, 0.3],
                      [0.3, 0.7]])
    s1_orig = torch.tensor([0.8, 0.2]).log()
    s2_orig = torch.tensor([0.1, 0.9]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    actual = [T0.log(), T.log(), s1_orig, s2_orig]
    for i, p in enumerate(model.parameters()):
        assert torch.isclose(actual[i], p).all()


def test_hmm_smooth():
    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.7, 0.3],
                      [0.3, 0.7]])
    s1_orig = torch.tensor([0.8, 0.2]).log()
    s2_orig = torch.tensor([0.1, 0.9]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    obs_seq = pack_list([torch.tensor([1, 1])])

    # New approach
    posterior_ll = model.smooth(obs_seq)
    post_unpacked, post_lengths = pad_packed_sequence(posterior_ll,
                                                      batch_first=True)

    posterior_prob = post_unpacked[0].softmax(1)

    first_correct = torch.tensor([[0.11664296, 0.88335704]])
    assert torch.allclose(posterior_prob[0], first_correct)


def test_hmm_predict_aima_umbrella_example():
    """
    This example was taken from AI a Modern Approach

    The example comes from page 572, filtering and prediction section. The
    correct values were manually compared propagated one step into future from
    the filtering example.
    """
    # states = {0: 'No Rain', 1: 'Rain'}
    # obs = {0: 'No Umbrella', 1: 'Umbrella'}

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.7, 0.3],
                      [0.3, 0.7]])
    s1_orig = torch.tensor([0.8, 0.2]).log()
    s2_orig = torch.tensor([0.1, 0.9]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    obs_seq = pack_list([torch.tensor([1])])
    posterior = model.predict(obs_seq)
    normalized = posterior.softmax(1)
    correct = torch.tensor([[0.37272727, 0.62727273]])
    assert torch.allclose(normalized, correct)

    obs_seq = pack_list([torch.tensor([1, 1])])
    posterior = model.predict(obs_seq)
    normalized = posterior.softmax(1)
    correct = torch.tensor([[0.34665718, 0.65334282]])
    assert torch.allclose(normalized, correct)


def test_hmm_fit_viterbi():

    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_orig = torch.tensor([0.99, 0.01]).log()
    s2_orig = torch.tensor([0.05, 0.95]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(50, 100)

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.6, 0.4],
                      [0.5, 0.5]])
    s1_orig = torch.tensor([0.6, 0.4]).log()
    s2_orig = torch.tensor([0.5, 0.5]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    converge = model.fit(obs_seq, max_steps=500,
                         epsilon=1e-2, alg="viterbi")

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

    assert converge

    states_seq, _ = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = states_seq
    true = states
    accuracy = torch.mean(torch.abs(pred.data - true.data).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1


def test_hmm_fit_autograd():

    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_orig = torch.tensor([0.99, 0.01]).log()
    s2_orig = torch.tensor([0.05, 0.95]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(50, 100)

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.6, 0.4],
                      [0.5, 0.5]])
    s1_orig = torch.tensor([0.6, 0.4]).log()
    s2_orig = torch.tensor([0.5, 0.5]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    import time
    t0 = time.time()
    converge = model.fit(obs_seq, max_steps=500,
                         epsilon=1e-2, alg="autograd", lr=1e-1)
    t1 = time.time()
    total = t1-t0
    print("CPU runtime", total)

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

    assert converge

    states_seq, _ = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = states_seq
    true = states
    accuracy = torch.mean(torch.abs(pred.data - true.data).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1


def test_hmm_fit_autograd_diagnormal():

    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_means = torch.tensor([0.0, 0.0, 0.0])
    s1_precs = torch.tensor([1.0, 1.0, 1.0])
    s2_means = torch.tensor([10.0, 10.0, 10.0])
    s2_precs = torch.tensor([1.0, 1.0, 1.0])
    s1 = DiagNormalModel(means=s1_means, precs=s1_precs)
    s2 = DiagNormalModel(means=s2_means, precs=s2_precs)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(50, 100)

    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_means = torch.tensor([3.0, 3.0, 3.0])
    s1_precs = torch.tensor([1.0, 1.0, 1.0])
    s2_means = torch.tensor([6.0, 6.0, 6.0])
    s2_precs = torch.tensor([1.0, 1.0, 1.0])
    s1 = DiagNormalModel(means=s1_means, precs=s1_precs)
    s2 = DiagNormalModel(means=s2_means, precs=s2_precs)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    converge = model.fit(obs_seq, max_steps=500,
                         epsilon=1e-2, alg="autograd", lr=1e-1)

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_pi)
    print("Pi Matrix: ")
    print(model.T0)

    print("Transition Matrix: ")
    print(model.T)
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission: ")
    for s in model.states:
        p = list(s.parameters())
        print("Means", p[0])
        print("Cov", p[1].abs())
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

    assert converge

    states_seq, _ = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = states_seq
    true = states
    accuracy = torch.mean(torch.abs(pred.data - true.data).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1


def test_hmm_fit_viterbi_diagnormal():

    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_means = torch.tensor([0.0, 0.0, 0.0])
    s1_precs = torch.tensor([1.0, 1.0, 1.0])
    s2_means = torch.tensor([10.0, 10.0, 10.0])
    s2_precs = torch.tensor([1.0, 1.0, 1.0])
    s1 = DiagNormalModel(means=s1_means, precs=s1_precs)
    s2 = DiagNormalModel(means=s2_means, precs=s2_precs)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(50, 100)

    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_means = torch.tensor([3.0, 3.0, 3.0])
    s1_precs = torch.tensor([1.0, 1.0, 1.0])
    s2_means = torch.tensor([6.0, 6.0, 6.0])
    s2_precs = torch.tensor([1.0, 1.0, 1.0])
    s1 = DiagNormalModel(means=s1_means, precs=s1_precs)
    s2 = DiagNormalModel(means=s2_means, precs=s2_precs)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    converge = model.fit(obs_seq, max_steps=500,
                         epsilon=1e-2, alg="viterbi")

    # Not enough samples (only 1) to test
    # assert np.allclose(trans0.data.numpy(), True_pi)
    print("Pi Matrix: ")
    print(model.T0)

    print("Transition Matrix: ")
    print(model.T)
    # assert np.allclose(transition.exp().data.numpy(), True_T, atol=0.1)
    print()
    print("Emission: ")
    for s in model.states:
        p = list(s.parameters())
        print("Means", p[0])
        print("Cov", p[1].abs())
    # assert np.allclose(emission.exp().data.numpy(), True_E, atol=0.1)
    print()
    print("Reached Convergence: ")
    print(converge)

    assert converge

    states_seq, _ = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = states_seq
    true = states
    accuracy = torch.mean(torch.abs(pred.data - true.data).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Requires CUDA Device")
def test_hmm_fit_autograd_gpu():

    device = torch.device('cuda:0')

    T0 = torch.tensor([0.75, 0.25])
    T = torch.tensor([[0.85, 0.15],
                      [0.12, 0.88]])
    s1_orig = torch.tensor([0.99, 0.01]).log()
    s2_orig = torch.tensor([0.05, 0.95]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)

    obs_seq, states = model.sample(50, 100)

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.6, 0.4],
                      [0.5, 0.5]])
    s1_orig = torch.tensor([0.6, 0.4]).log()
    s2_orig = torch.tensor([0.5, 0.5]).log()
    s1 = CategoricalModel(logits=s1_orig)
    s2 = CategoricalModel(logits=s2_orig)
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    model.to(device)

    import time
    obs_seq = obs_seq.to(device)
    t0 = time.time()
    converge = model.fit(obs_seq, max_steps=500,
                         epsilon=1e-2, alg="autograd")
    t1 = time.time()
    total = t1-t0
    print("GPU runtime", total)

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

    assert converge

    states_seq, _ = model.decode(obs_seq)

    # state_summary = np.array([model.prob_state_1[i].cpu().numpy() for i in
    #                           range(len(model.prob_state_1))])

    # pred = (1 - state_summary[-2]) > 0.5
    # pred = torch.cat(states_seq, 0).data.numpy()
    # true = np.concatenate(states, 0)
    pred = states_seq.to("cpu")
    true = states
    accuracy = torch.mean(torch.abs(pred.data - true.data).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1
