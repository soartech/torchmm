"""
This module includes the base Model class used throughout TorCHmm. It also
includes some very basic models for discrete and continuous emissions.
"""
import torch
from torch.distributions import Categorical


class Model(object):
    """
    This is an unsupervised model base class. It specifies the methods that
    should be implemented.
    """

    def sample(self, *args, **kargs):
        """
        Draws a sample from the model. This method might take additional
        arguments for specifying things like how many samples to draw.
        """
        raise NotImplementedError("sample method not implemented")

    def log_prob(self, X):
        """
        Computes the log likelihood of the data X given the model.
        """
        raise NotImplementedError("log_likelihood method not implemented")

    def fit(self, X, *args, **kargs):
        """
        Updates the model parameters to best fit the data X. This might accept
        additional arguments that govern how the model is updated.
        """
        raise NotImplementedError("fit method not implemented")

    def parameters(self):
        """
        Returns the models parameters, as a list of tensors. This is is so
        other models can optimize this model. For example, in the case of
        nested models.
        """
        raise NotImplementedError("parameters method not implemented")


class CategoricalModel(Model):

    def __init__(self, probs=None, logits=None):
        """
        Accepts a set of probabilites OR logits for the model, NOT both.
        """
        if probs is not None and logits is not None:
            raise ValueError("Both probs and logits provided; only one should"
                             " be used.")
        elif probs is not None:
            self.logits = probs.log()
        elif logits is not None:
            self.logits = logits
        else:
            raise ValueError("Neither probs or logits provided; one must be.")

    def sample(self, sample_shape=None):
        """
        Draws n samples from this model.
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1])
        print("logits", self.logits)
        print("SAMPLE SHAPE", sample_shape)
        return Categorical(logits=self.logits).sample(sample_shape)

    def log_prob(self, value):
        """
        Returns the loglikelihood of x given the current categorical
        distribution.
        """
        return Categorical(logits=self.logits).log_prob(value)

    def parameters(self):
        """
        Returns the model parameters for optimization.
        """
        yield self.logits

    def fit(self, X):
        """
        Update the logit vector based on the observed counts.

        .. todo::
            Maybe could be modified with weights to support baum welch?
        """
        counts = X.bincount(minlength=self.logits.shape[0]).float()
        prob = counts / counts.sum()
        self.logits = prob.log()
