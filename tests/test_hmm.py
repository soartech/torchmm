import torch
from torchmm.base import CategoricalModel
from torchmm.hmm import HiddenMarkovModel


def test_hmm_sample():

    # create categorical models for the states
    T0 = torch.tensor([1.0, 0.0])
    T = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])
    s1 = CategoricalModel(probs=torch.tensor([1.0, 0.0]))
    s2 = CategoricalModel(probs=torch.tensor([0.0, 1.0]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(3, 10)

    assert obs_seq.shape == (3, 10)
    assert states.shape == (3, 10)
    assert 1 not in obs_seq
    assert 1 not in states

    T0 = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.5, 0.5],
                      [0.5, 0.5]])
    s1 = CategoricalModel(probs=torch.tensor([1.0, 0.0]))
    s2 = CategoricalModel(probs=torch.tensor([0.0, 1.0]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(4, 20)

    assert obs_seq.shape == (4, 20)
    assert states.shape == (4, 20)
    assert 1 in obs_seq and 0 in obs_seq
    assert 1 in states and 0 in states

    T0 = torch.tensor([0.9, 0.1])
    T = torch.tensor([[0.9, 0.1],
                      [0.5, 0.5]])
    s1 = CategoricalModel(probs=torch.tensor([1.0, 0.0]))
    s2 = CategoricalModel(probs=torch.tensor([0.0, 1.0]))
    model = HiddenMarkovModel([s1, s2], T0=T0, T=T)
    obs_seq, states = model.sample(1, 20)

    assert obs_seq.shape == (1, 20)
    assert states.shape == (1, 20)
    assert (states == 0).sum() > (states == 1).sum()


def test_hmm_log_prob():
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

    obs_seq = torch.tensor([[1]])
    ll_score = model.log_prob(obs_seq).item()
    assert abs(ll_score - -0.5978) < 0.001

    obs_seq = torch.tensor([[1, 1]])
    ll_score = model.log_prob(obs_seq).item()
    assert abs(ll_score - -1.045545) < 0.001

    obs_seq = torch.tensor([[1], [1]])
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

    converge = model.fit(obs_seq, max_steps=500, epsilon=1e-2, alg="viterbi")

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
    accuracy = torch.mean(torch.abs(pred - true).float())
    print("Accuracy: ", accuracy)
    assert accuracy >= 0.9 or accuracy <= 0.1


def test_autograd_training():

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

    assert converge

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
