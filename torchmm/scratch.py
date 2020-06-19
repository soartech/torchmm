import torch
from torchmm.hmm import HiddenMarkovModel
from torchmm.base import DiagNormalModel

good_T0 = torch.tensor([1.0, 0.0])
good_T = torch.tensor([[1.0, 0.0], [0.0, 1.0]])


m1 = DiagNormalModel(2)
m2 = DiagNormalModel(2)
hmm = HiddenMarkovModel([m1, m2])
hmm.sample(2, 2)
