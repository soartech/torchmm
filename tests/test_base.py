import torch
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
    m = CategoricalModel(probs=torch.tensor([1., 2., 7.]))
    assert torch.isclose(m.log_prob(
        torch.tensor([2])), torch.tensor([0.7]).log())


def test_categorical_model_parmeters():
    original = torch.tensor([1, 1, 1])
    m = CategoricalModel(logits=original)
    p = list(m.parameters())
    assert torch.isclose(original, torch.tensor([1, 1, 1])).all()
    p[0] += 1
    assert torch.isclose(original, torch.tensor([2, 2, 2])).all()


def test_categorical_model_fit():
    m = CategoricalModel(logits=torch.tensor([0, 0, 0]))
    m.fit(torch.tensor([0, 0, 1, 1, 2]))
    assert torch.isclose(list(m.parameters())[0].softmax(0),
                         torch.tensor([0.4, 0.4, 0.2])).all()


def test_diagnormalmodel_sample():
    m = DiagNormalModel(means=torch.tensor([0., 5., 10.]),
                        covs=torch.tensor([0.00001, 0.00001, 0.00001]))
    s = m.sample()
    assert abs(s[0][0] - 0) <= 0.1
    assert abs(s[0][1] - 5) <= 0.1
    assert abs(s[0][2] - 10) <= 0.1


def test_diagnormalmodel_log_prob():
    m = DiagNormalModel(means=torch.tensor([0.]),
                        covs=torch.tensor([1.]))
    assert torch.allclose(m.log_prob(
        torch.tensor([[0.]])), torch.tensor([-0.9189]), atol=1e-4)


def test_diagnormalmodel_parmeters():
    means = torch.tensor([0.])
    covs = torch.tensor([1.])
    m = DiagNormalModel(means=means,
                        covs=covs)
    p = list(m.parameters())
    assert torch.isclose(means, torch.tensor([0.])).all()
    assert torch.isclose(covs, torch.tensor([1.])).all()
    p[0] += 1
    p[1] += 1
    assert torch.isclose(means, torch.tensor([1.])).all()
    assert torch.isclose(covs, torch.tensor([2.])).all()


def test_diagnormalmodel_fit():
    m = DiagNormalModel(means=torch.tensor([0.]),
                        covs=torch.tensor([1.]))
    m.fit(torch.tensor([0., 0., 0., 2., 2., 2.]))
    p = list(m.parameters())
    assert torch.allclose(p[0], torch.tensor(1.))
    assert torch.allclose(p[1], torch.tensor(1.2))
