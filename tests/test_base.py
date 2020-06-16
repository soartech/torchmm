import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from torchmm.base import CategoricalModel
from torchmm.base import DiagNormalModel


def test_categorical_model_sample():
    m = CategoricalModel(probs=torch.tensor([0., 0., 1.]))
    assert m.sample().item() == 2

    m = CategoricalModel(probs=torch.tensor([0., 1., 0.]))
    assert m.sample().item() == 1

    m = CategoricalModel(probs=torch.tensor([1., 0., 0.]))
    assert m.sample().item() == 0


def test_categorical_model_log_prob():
    m = CategoricalModel(probs=torch.tensor([0.1, 0.2, 0.7]))
    assert torch.isclose(m.log_prob(
        torch.tensor([2])), torch.tensor([0.7]).log())


def test_categorical_model_fit():
    probs = torch.tensor([0., 0., 0.]).softmax(0)
    m = CategoricalModel(probs=probs)
    data = m.sample([1000])
    actual_counts = data.bincount(minlength=probs.shape[0]).float()
    actual_probs = actual_counts / actual_counts.sum()
    m.fit(data)
    assert torch.allclose(m.probs, actual_probs, atol=1e-2)

    m = CategoricalModel(probs=probs)
    m.fit(torch.tensor([0]))

    # with a prior of 1 obs for each output and only a single data
    # point (0),
    # probs should be 0.5, 0.25, 0.25
    assert torch.allclose(m.probs,
                          torch.tensor([0.5, 0.25, 0.25]))


def test_diagnormalmodel_sample():
    m = DiagNormalModel(means=torch.tensor([0., 5., 10.]),
                        precs=torch.tensor([10000, 10000, 10000]))
    s = m.sample()
    assert abs(s[0][0] - 0) <= 0.1
    assert abs(s[0][1] - 5) <= 0.1
    assert abs(s[0][2] - 10) <= 0.1


def test_diagnormalmodel_log_prob():
    m = DiagNormalModel(means=torch.tensor([0.]),
                        precs=torch.tensor([1.]))
    assert torch.allclose(m.log_prob(
        torch.tensor([[0.]])), torch.tensor([-0.9189]), atol=1e-4)


def test_diagnormalmodel_parmeters():
    means = torch.tensor([0.])
    precs = torch.tensor([1.])
    m = DiagNormalModel(means=means,
                        precs=precs)
    p = list(m.parameters())
    assert torch.isclose(means, torch.tensor([0.])).all()
    assert torch.isclose(precs, torch.tensor([1.])).all()
    p[0] += 1
    p[1] += 1
    assert torch.isclose(means, torch.tensor([0.])).all()
    assert torch.isclose(precs, torch.tensor([1.])).all()


def test_diagnomralmodel_fit_no_data():
    m = DiagNormalModel(means=torch.tensor([0.]),
                        precs=torch.tensor([1.]))
    X = torch.empty((0, 1))
    m.fit(X)
    assert not torch.isnan(m.means).any()
    assert not torch.isnan(m.precs).any()


def test_diagnormalmodel_fit_2d():
    m = DiagNormalModel(means=torch.tensor([0., 0.]),
                        precs=torch.tensor([1., 1.]))
    generator = MultivariateNormal(
        loc=torch.tensor([0., 100.]),
        precision_matrix=torch.tensor([4., 0.5]).diag())
    data = generator.sample((100000,))

    m.fit(data)
    p = list(m.parameters())

    assert torch.allclose(p[0], data.mean(0), atol=1e-1)
    assert torch.allclose(p[1], 1./data.std(0).pow(2), atol=1e-1)
