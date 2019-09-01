import torch
from torchmm.base import CategoricalModel


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
    p = m.parameters()
    assert torch.isclose(original, torch.tensor([1, 1, 1])).all()
    p += 1
    assert torch.isclose(original, torch.tensor([2, 2, 2])).all()


def test_categorical_model_fit():
    m = CategoricalModel(logits=torch.tensor([0, 0, 0]))
    m.fit(torch.tensor([0, 0, 1, 1, 2]))
    assert torch.isclose(m.parameters().softmax(0),
                         torch.tensor([0.4, 0.4, 0.2])).all()
