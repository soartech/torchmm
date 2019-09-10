import torch
from torch.distributions import Categorical
import torch.optim as optim
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
    m = CategoricalModel(probs=torch.tensor([1., 2., 7.]))
    assert torch.isclose(m.log_prob(
        torch.tensor([2])), torch.tensor([0.7]).log())


def test_categorical_model_parmeters():
    original = torch.tensor([1, 1, 1])
    m = CategoricalModel(logits=original)
    p = list(m.parameters())
    assert torch.isclose(m.logits, torch.tensor([1., 1., 1.])).all()
    p[0] += 1
    assert torch.isclose(m.logits, torch.tensor([2., 2., 2.])).all()


def test_categorical_model_fit():
    logits = torch.tensor([0., 0., 0.])
    m = CategoricalModel(logits=logits)
    data = Categorical(logits=logits).sample([1000])
    actual_counts = data.bincount(minlength=logits.shape[0]).float()
    actual_probs = actual_counts / actual_counts.sum()
    m.fit(data)
    assert torch.allclose(list(m.parameters())[0].softmax(0),
                          actual_probs, atol=1e-2)

    m = CategoricalModel(probs=logits)
    m.fit(torch.tensor([0]))
    assert torch.allclose(list(m.parameters())[0].softmax(0),
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
    assert torch.isclose(means, torch.tensor([1.])).all()
    assert torch.isclose(precs, torch.tensor([2.])).all()


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


def test_diagnormalmodel_fit():
    m = DiagNormalModel(means=torch.tensor([0.]),
                        precs=torch.tensor([1.]))
    data = torch.zeros([10000, 1])

    true_mean = 0.
    true_std = 10.
    data.normal_(true_mean, true_std)

    m.fit(data)
    p = list(m.parameters())

    from pprint import pprint
    pprint(p)

    assert torch.allclose(p[0], data.mean(), atol=1e-2)
    assert torch.allclose(p[1], 1/data.std().pow(2), atol=1e-2)

    data = torch.zeros([2, 1])

    true_mean = 0.
    true_std = 100.
    data.normal_(true_mean, true_std)

    m.fit(data)
    p = list(m.parameters())

    from pprint import pprint
    pprint(p)

    print('obs', data.mean(), data.std())
    assert torch.allclose(p[0], (2 * data.mean()) / 3, atol=1e-2)

    data = data - data.mean()

    m.fit(data)
    p = list(m.parameters())

    from pprint import pprint
    pprint(p)

    assert torch.all(p[1] > 1/data.std().pow(2))


def test_diagnormalmodel_fit_autograd():

    m = DiagNormalModel(means=torch.tensor([0.]),
                        precs=torch.tensor([1.]))
    # p = list(m.parameters())
    # p[0].requires_grad_(True)

    for p in m.parameters():
        p.requires_grad_(True)

    true_mean = 0.
    true_std = 10
    data = torch.zeros([10000, 1])
    data.normal_(true_mean, true_std)

    optimizer = optim.AdamW(m.parameters(), lr=1e-3)
    # optimizer = optim.AdamW([p[0]], lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    max_steps = 1000
    for i in range(max_steps):
        ll = m.log_prob(data).sum() + m.log_parameters_prob()
        loss = -1 * ll
        print("loss: ", loss, "(%i of %i)" % (i, max_steps))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

    p = list(m.parameters())
    from pprint import pprint
    pprint(p)

    print('actual')
    print(data.mean(), 1./data.std().pow(2))

    assert torch.allclose(p[0][0], data.mean(), atol=1e-2)
    assert torch.allclose(p[1][0].abs(), 1/(data.std().pow(2)), atol=1e-2)

    data = torch.zeros([2, 1])

    true_mean = 0.
    true_std = 0.5
    data.normal_(true_mean, true_std)

    optimizer = optim.AdamW(m.parameters(), lr=1e-3)
    # optimizer = optim.AdamW([p[0]], lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    max_steps = 1000
    for i in range(max_steps):
        ll = m.log_prob(data).sum() + m.log_parameters_prob()
        loss = -1 * ll
        print("loss: ", loss, "(%i of %i)" % (i, max_steps))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

    p = list(m.parameters())

    from pprint import pprint
    pprint(p)

    print('obs', data.mean(), 1/data.std().pow(2))
    assert torch.allclose(p[0][0], (2 * data.mean()) / 3, atol=1e-2)

    true_mean = 0.
    true_std = 100
    data.normal_(true_mean, true_std)
    data = data - data.mean()

    optimizer = optim.AdamW(m.parameters(), lr=1e-3)
    # optimizer = optim.AdamW([p[0]], lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    max_steps = 1000
    for i in range(max_steps):
        ll = m.log_prob(data).sum() + m.log_parameters_prob()
        loss = -1 * ll
        print("loss: ", loss, "(%i of %i)" % (i, max_steps))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
    p = list(m.parameters())

    from pprint import pprint
    pprint(p)

    assert torch.all(p[1][0].abs() > 1/data.std().pow(2))
