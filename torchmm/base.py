"""
This module includes the base Model class used throughout TorCHmm. It also
includes some very basic models for discrete and continuous emissions.
"""
import torch
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma


class Model(object):
    """
    This is an unsupervised model base class. It specifies the methods that
    should be implemented.
    """

    def init_params_random(self):
        """
        Randomly sets the parameters of the model; used for model fitting.
        """
        raise NotImplementedError("init_params_random method not implemented")

    def log_parameters_prob(self):
        """
        Returns the loglikelihood of the parameter estimates given the prior.
        """
        raise NotImplementedError("log_parameters_prob method not implemented")

    def to(self, device):
        """
        Moves the model's parameters / tensors to the specified device.
        """
        raise NotImplementedError("to method not implemented")

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

    def set_parameters(self, params):
        """
        Used to set the value of a models parameters
        """
        raise NotImplementedError("set_parameters method not implemented")


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

        self.device = "cpu"

        self.prior = torch.ones_like(self.logits)

    def to(self, device):
        self.logits = self.logits.to(device)
        self.prior = self.prior.to(device)
        self.device = device

    def log_parameters_prob(self):
        return Dirichlet(self.prior).log_prob(self.logits.softmax(0))

    def init_params_random(self):
        self.logits = torch.rand_like(self.logits).softmax(0).log()

    def sample(self, sample_shape=None):
        """
        Draws n samples from this model.
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
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

    def set_parameters(self, params):
        """
        Returns the model parameters for optimization.
        """
        self.logits = params[0]

    def fit(self, X):
        """
        Update the logit vector based on the observed counts.

        .. todo::
            Maybe could be modified with weights to support baum welch?
        """
        counts = X.bincount(minlength=self.logits.shape[0]).float()
        prob = counts / counts.sum()
        self.logits = prob.log()


class DiagNormalModel(Model):

    def __init__(self, means, covs):
        """
        Accepts a set of mean and cov for each dimension.

        Currently assumes all dimensions are independent.
        """
        if not isinstance(means, torch.Tensor):
            raise ValueError("Means must be a tensor.")
        if not isinstance(covs, torch.Tensor):
            raise ValueError("Covs must be a tensor.")
        if means.shape != covs.shape:
            raise ValueError("Means and covs must have same shape!")

        self.means = means
        self.means_mean_prior = torch.zeros_like(self.means)
        self.means_std_prior = torch.ones_like(self.means)

        self.covs = covs
        self.cov_alpha_prior = 100 * torch.ones_like(self.covs)
        self.cov_beta_prior = 100 * torch.ones_like(self.covs)

        self.device = "cpu"

    def to(self, device):
        self.means = self.means.to(device)
        self.means_mean_prior = self.means_mean_prior.to(device)
        self.means_std_prior = self.means_std_prior.to(device)
        self.covs = self.covs.to(device)
        self.cov_alpha_prior = self.cov_alpha_prior.to(device)
        self.cov_beta_prior = self.cov_alpha_prior.to(device)
        self.device = device

    def init_params_random(self):
        self.means.normal_()
        self.covs.log_normal_()

    def sample(self, sample_shape=None):
        """
        Draws n samples from this model.
        """
        if sample_shape is None:
            sample_shape = torch.tensor([1], device=self.device)
        return MultivariateNormal(
            loc=self.means,
            covariance_matrix=self.covs.abs().diag()).sample(sample_shape)

    def log_parameters_prob(self):
        cov_prior = self.means_std_prior.pow(2).diag()
        means_ll = MultivariateNormal(
            loc=self.means_mean_prior,
            covariance_matrix=cov_prior).log_prob(self.means)
        cov_ll = Gamma(self.cov_alpha_prior,
                       self.cov_beta_prior).log_prob(self.covs.abs()).sum()
        return means_ll + cov_ll

    def log_prob(self, value):
        """
        Returns the loglikelihood of x given the current parameters.
        """
        return MultivariateNormal(
            loc=self.means,
            covariance_matrix=(1e-5 + self.covs.abs()).diag()
        ).log_prob(value)

    def parameters(self):
        """
        Returns the model parameters for optimization.
        """
        yield self.means
        yield self.covs

    def set_parameters(self, params):
        """
        Sets the model parameters.
        """
        print("SETTING", params)
        self.means = params[0]
        self.covs = params[1]

    def fit(self, X):
        """
        Update the logit vector based on the observed counts.

        .. todo::
            - Utilize the priors to do a MAP estimator. See:
                https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
        """
        self.means = X.mean(0)
        self.covs = X.std(0).pow(2)
